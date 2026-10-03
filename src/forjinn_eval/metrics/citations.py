"""Citation metrics (ragas ``QuotedSpansAlignment`` + deepeval
``CitationFaithfulness``).

* :class:`QuotedSpansAlignment` - deterministic: fraction of double-quoted
  spans in the answer that appear verbatim (whitespace-collapsed, casefolded)
  in the retrieved contexts. Catches quoted text the source never actually said.
* :class:`CitationFaithfulness` - LLM-judged: for each ``[N]`` citation marker,
  does the cited passage (by ordinal) actually support the claim it is attached
  to? Stricter than plain faithfulness - catches misattribution.
"""
from __future__ import annotations

import re
from typing import List, Optional

from ..capture import AgentRun
from ..judge import JudgeClient, _judge_or_default
from ._common import _cr, _f, _judge_user, _nan_or_skip, _verdict_list
from .base import CheckResult, Evaluator


def _collapse_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _find_quoted_spans(text: str, min_words: int = 5) -> List[str]:
    spans: List[str] = []
    for m in re.finditer(r'"([^"]+)"|“([^”]+)”', text or ""):
        s = (m.group(1) or m.group(2) or "").strip()
        if len(s.split()) >= min_words:
            spans.append(s)
    return spans


class QuotedSpansAlignment(Evaluator):
    name = "quoted_spans_alignment"
    kind = "rag"

    def __init__(self, min_span_words: int = 5, casefold: bool = True,
                 threshold: float = 1.0, name: Optional[str] = None):
        self.min_span_words = min_span_words
        self.casefold = casefold
        self.threshold = threshold
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        ctx = run.retrieved_context()
        if not ctx:
            return _nan_or_skip(self.name, "no retrieved context")
        spans = _find_quoted_spans(run.text or "", self.min_span_words)
        if not spans:
            return _cr(self.name, 1.0, self.threshold, details={"spans": 0})
        corpus = " ".join(ctx)
        needle_corpus = _collapse_ws(corpus.casefold() if self.casefold else corpus)
        found = 0
        for s in spans:
            h = _collapse_ws(s.casefold() if self.casefold else s)
            if h and h in needle_corpus:
                found += 1
        score = found / len(spans)
        return _cr(self.name, score, self.threshold,
                   details={"spans": len(spans), "found": found,
                            "spans_found": [s for s in spans if _collapse_ws(s) in needle_corpus]})


class CitationFaithfulness(Evaluator):
    name = "citation_faithfulness"
    kind = "rag"

    _JUDGE = (
        "You are given an answer that cites passages with markers like [1], [2] (1-indexed "
        "into the retrieved passages) and the retrieved passages themselves. For EACH citation "
        "marker in the answer, check the claim immediately before/attached to that marker against "
        "the cited passage. Return JSON {\"verdicts\": [{\"cite\": int, \"faithful\": 0|1, "
        "\"reason\": str}, ...]} with one entry per distinct citation marker.\n"
        "Answer: {answer}\nRetrieved passages (1-indexed):\n{passages}\nJSON:"
    )

    def __init__(self, judge: Optional[JudgeClient] = None, threshold: float = 1.0,
                 name: Optional[str] = None):
        self.judge = _judge_or_default(judge)
        self.threshold = threshold
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        ctx = run.retrieved_context()
        text = run.text or ""
        cites = sorted({int(m) for m in re.findall(r"\[(\d+)\]", text)})
        if not ctx:
            return _nan_or_skip(self.name, "no retrieved context")
        if not cites:
            return _cr(self.name, 1.0, self.threshold, details={"citations": 0})
        out = self.judge.complete_json(_judge_user(_f(
            self._JUDGE, answer=text,
            passages="\n".join(f"[{i}] {p}" for i, p in enumerate(ctx, 1)))))
        flags = _verdict_list((out or {}).get("verdicts"))
        if not flags:
            return CheckResult.error_(self.name, "judge returned no citation verdicts")
        score = sum(flags) / len(flags)
        return _cr(self.name, score, self.threshold,
                   details={"citations": cites, "verdicts": flags})


__all__ = [
    "CitationFaithfulness",
    "QuotedSpansAlignment",
    "_collapse_ws",
    "_find_quoted_spans",
]
