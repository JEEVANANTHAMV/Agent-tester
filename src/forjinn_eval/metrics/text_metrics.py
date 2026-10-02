"""Deterministic text-quality metrics (ragas/deepeval, no LLM required).

Implements the non-LLM string/reference metrics the reference libraries ship:

* :class:`PatternMatch`       - regex present/absent in the output (deepeval).
* :class:`BleuScore`          - corpus BLEU vs a reference (ragas, sacrebleu or
  a pure-Python fallback).
* :class:`RougeScore`         - ROUGE-1 / ROUGE-L vs a reference (pure-Python).
* :class:`ChrfScore`          - character-n-gram F-score vs a reference.
* :class:`SemanticSimilarity` - cosine similarity via the shared embedding
  backend (ragas ``AnswerSimilarity``); :class:`AnswerSimilarity` is an alias.

All read ``run.text`` and an optional ``reference`` (defaulted to
``run.reference``). They are cheap, fully offline, and stable - the backbone of
CI that must never depend on a live judge.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from itertools import combinations
from typing import List, Optional

from ..capture import AgentRun
from ..judge import Embeddings
from .base import CheckResult, Evaluator
from ._common import _cr, _nan_or_skip


# ---------------------------------------------------------------------------
# Shared scoring primitives
# ---------------------------------------------------------------------------
def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def _ngrams(tokens: List[str], n: int) -> List[tuple]:
    if n < 1:
        n = 1
    if len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _count_ngrams(tokens: List[str]) -> Counter:
    c: Counter = Counter()
    for n in range(1, 5):
        c.update(_ngrams(tokens, n))
    return c


def bleu(references: List[str], hypothesis: str, max_n: int = 4, smoothing: bool = True) -> float:
    """Corpus/segment BLEU in [0,1] (higher = more n-gram overlap with ref)."""
    hyp = _tokens(hypothesis)
    ref = _tokens(references[0]) if references else []
    if not hyp or not ref:
        return 0.0
    precisions: List[float] = []
    ref_counts: List[Counter] = []
    for n in range(1, max_n + 1):
        ref_n = _ngrams(ref, n)
        ref_counts.append(Counter(ref_n))
        hyp_n = _ngrams(hyp, n)
        matches = 0
        total = 0
        hc = Counter(hyp_n)
        rc = ref_counts[-1]
        for gram, cnt in hc.items():
            total += cnt
            matches += min(cnt, rc.get(gram, 0))
        p = matches / total if total else 0.0
        precisions.append(p)
    if any(p == 0.0 for p in precisions):
        return 0.0
    bp = 1.0 if len(hyp) > len(ref) else math.exp(1 - len(ref) / max(len(hyp), 1))
    log_prec = sum(math.log(p) for p in precisions) / len(precisions)
    return max(0.0, min(1.0, math.exp(log_prec) * bp))


def rouge_l(reference: str, hypothesis: str) -> float:
    """ROUGE-L = F1 over the longest common subsequence (F-measure default)."""
    r_tokens = _tokens(reference)
    h_tokens = _tokens(hypothesis)
    if not r_tokens or not h_tokens:
        return 0.0
    # LCS length via DP (O(n*m)); guard large inputs.
    if len(r_tokens) * len(h_tokens) > 4_000_000:
        # fall back to n-gram-overlap ratio (cheap)
        rc = _count_ngrams(r_tokens)
        hc = _count_ngrams(h_tokens)
        inter = sum((rc & hc).values())
        union = sum((rc | hc).values())
        return inter / union if union else 0.0
    prev = [0] * (len(h_tokens) + 1)
    for i in range(1, len(r_tokens) + 1):
        curr = [0] * (len(h_tokens) + 1)
        for j in range(1, len(h_tokens) + 1):
            if r_tokens[i - 1] == h_tokens[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    lcs = prev[len(h_tokens)]
    p = lcs / len(h_tokens)
    rc_ = lcs / len(r_tokens)
    f = (2 * p * rc_ / (p + rc_)) if (p + rc_) else 0.0
    return max(0.0, min(1.0, f))


def chrf(reference: str, hypothesis: str, char_order: int = 6, word_order: int = 0, beta: float = 2.0) -> float:
    """Character-n-gram (and optional word-n-gram) F-score in [0,1]."""
    text = (hypothesis if hypothesis else "")
    ref = reference or ""
    score = 0.0
    matched_char = 0
    total_char = 0
    ref_char = 0
    for i in range(0, min(char_order, 4) + 1):
        # char n-grams on the raw text (keeps subword signal)
        chars_ref = list(ref)
        chars_hyp = list(text)
        if i > 0:
            n_ref = [tuple(chars_ref[k:k + i]) for k in range(len(chars_ref) - i + 1)]
            n_hyp = [tuple(chars_hyp[k:k + i]) for k in range(len(chars_hyp) - i + 1)]
        else:
            n_ref = chars_ref
            n_hyp = chars_hyp
        c_ref = Counter(n_ref)
        c_hyp = Counter(n_hyp)
        matched = sum((c_ref & c_hyp).values())
        matched_char += matched
        total_char += sum(c_hyp.values())
        ref_char += sum(c_ref.values())
    if total_char == 0 or ref_char == 0:
        return 0.0
    p = 1.0 - (total_char - matched_char) / total_char
    r = 1.0 - (ref_char - matched_char) / ref_char
    b2 = beta * beta
    f = (1 + b2) * p * r / (b2 * p + r) if (b2 * p + r) else 0.0
    return max(0.0, min(1.0, f))


def _cosine_from_embed(emb: Optional[Embeddings], a: str, b: str) -> float:
    from ..judge import _cosine

    return _cosine(emb.embed(a), emb.embed(b))


# ---------------------------------------------------------------------------
# Metric classes
# ---------------------------------------------------------------------------
class PatternMatch(Evaluator):
    """deepeval PatternMatch: regex present in (or absent with ``invert_match``)."""

    name = "pattern_match"
    kind = "rag"

    def __init__(self, pattern: str, invert_match: bool = False, search: bool = True,
                 threshold: float = 1.0, name: Optional[str] = None):
        self.pattern = pattern
        self.invert_match = invert_match
        self.search = search
        self.threshold = threshold
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        text = run.text or ""
        found = re.search(self.pattern, text) is not None if self.search else re.fullmatch(self.pattern, text) is not None
        ok = (not found) if self.invert_match else found
        score = 1.0 if ok else 0.0
        return _cr(self.name, score, self.threshold,
                   details={"pattern": self.pattern, "found": found, "inverted": self.invert_match})


class BleuScore(Evaluator):
    """Corpus BLEU of the answer vs a reference (deterministic)."""

    name = "bleu_score"
    kind = "rag"

    def __init__(self, reference: Optional[str] = None, threshold: float = 0.5,
                 max_n: int = 4, name: Optional[str] = None):
        self.reference = reference
        self.threshold = threshold
        self.max_n = max_n
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        ref = self.reference if self.reference is not None else run.reference
        if ref is None:
            return _nan_or_skip(self.name, "no reference")
        score = bleu([str(ref)], run.text or "", max_n=self.max_n)
        return _cr(self.name, score, self.threshold, details={"bleu": score})


class RougeScore(Evaluator):
    """ROUGE (default F-measure / ROUGE-L) of the answer vs a reference."""

    name = "rouge_score"
    kind = "rag"

    def __init__(self, reference: Optional[str] = None, threshold: float = 0.5,
                 rouge_type: str = "rougeL", name: Optional[str] = None):
        self.reference = reference
        self.threshold = threshold
        self.rouge_type = rouge_type
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        ref = self.reference if self.reference is not None else run.reference
        if ref is None:
            return _nan_or_skip(self.name, "no reference")
        if self.rouge_type == "rouge1":
            rc = Counter(_tokens(ref))
            hc = Counter(_tokens(run.text or ""))
            inter = sum((rc & hc).values())
            p = inter / sum(hc.values()) if hc else 0.0
            r = inter / sum(rc.values()) if rc else 0.0
            score = (2 * p * r / (p + r)) if (p + r) else 0.0
        else:
            score = rouge_l(str(ref), run.text or "")
        return _cr(self.name, score, self.threshold, details={"type": self.rouge_type})


class ChrfScore(Evaluator):
    """CHRF (character n-gram F-score) of the answer vs a reference."""

    name = "chrf_score"
    kind = "rag"

    def __init__(self, reference: Optional[str] = None, threshold: float = 0.5,
                 char_order: int = 6, beta: float = 2.0, name: Optional[str] = None):
        self.reference = reference
        self.threshold = threshold
        self.char_order = char_order
        self.beta = beta
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        ref = self.reference if self.reference is not None else run.reference
        if ref is None:
            return _nan_or_skip(self.name, "no reference")
        score = chrf(str(ref), run.text or "", char_order=self.char_order, beta=self.beta)
        return _cr(self.name, score, self.threshold, details={"chrf": score})


class SemanticSimilarity(Evaluator):
    """ragas ``SemanticSimilarity`` / legacy ``AnswerSimilarity``: cosine of the
    answer vs the reference embeddings. Uses the shared offline-first
    :class:`~forjinn_eval.judge.Embeddings` backend, so it runs with no network."""

    name = "semantic_similarity"
    kind = "rag"

    def __init__(self, reference: Optional[str] = None, threshold: float = 0.5,
                 embeddings: Optional[Embeddings] = None, name: Optional[str] = None,
                 kind_: Optional[str] = None):
        self.reference = reference
        self.threshold = threshold
        self.embeddings = embeddings or Embeddings()
        if name:
            self.name = name
        if kind_:
            self.kind = kind_

    def evaluate(self, run: AgentRun) -> CheckResult:
        ref = self.reference if self.reference is not None else run.reference
        if ref is None:
            return _nan_or_skip(self.name, "no reference")
        score = _cosine_from_embed(self.embeddings, run.text or "", str(ref))
        return _cr(self.name, score, self.threshold, details={"cosine": score})


# Alias to the legacy ragas name so downstream code importing
# ``from forjinn_eval import AnswerSimilarity`` works unchanged.
class AnswerSimilarity(SemanticSimilarity):
    name = "answer_similarity"


__all__ = [
    "PatternMatch",
    "BleuScore",
    "RougeScore",
    "ChrfScore",
    "SemanticSimilarity",
    "AnswerSimilarity",
    "bleu",
    "rouge_l",
    "chrf",
]
