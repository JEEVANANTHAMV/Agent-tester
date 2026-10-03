"""Contextual RAG metrics: how well the *retrieved context* serves the query.

These evaluate the retrieval side (not just the answer). Context is read from
:attr:`AgentRun.retrieved_context`; the reference (ground truth) is passed
explicitly or read from :attr:`AgentRun.reference`.

* :class:`ContextualPrecision` - average precision over retrieved sentences,
  ranked by relevance to the reference (ragas/deepeval).
* :class:`ContextualRecall`    - fraction of reference sentences entailed by the
  union of retrieved contexts (ragas/deepeval).
* :class:`ContextualRelevancy` - average similarity between each retrieved
  context and the query (embedding-based; offline-capable).
* :class:`ContextEntityRecall` - fraction of reference NER entities present in
  the retrieved contexts (LLM-judged).
"""
from __future__ import annotations

import re
from typing import List, Optional

from ..capture import AgentRun
from ..judge import Embeddings, JudgeClient, _judge_or_default
from ._common import _cr, _f, _judge_user, _nan_or_skip, _verdict_list
from .base import CheckResult, Evaluator


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
    return [p.strip() for p in parts if p.strip()]


class ContextualPrecision(Evaluator):
    name = "contextual_precision"
    kind = "rag"

    _JUDGE = (
        "You are given a list of retrieved sentences (in order) and a reference answer. "
        "For each sentence, decide whether it is RELEVANT (helps produce the reference). "
        'Return JSON {"relevant": [0|1, ...]} with one entry per sentence, in order.\n'
        "Reference: {reference}\nSentences (indexed from 1):\n{sentences}\nJSON:"
    )

    def __init__(self, judge: Optional[JudgeClient] = None, reference: Optional[str] = None,
                 threshold: float = 0.5, name: Optional[str] = None):
        self.judge = _judge_or_default(judge)
        self.reference = reference
        self.threshold = threshold
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        ctx = run.retrieved_context()
        if not ctx:
            return _nan_or_skip(self.name, "no retrieved context")
        ref = self.reference if self.reference is not None else run.reference
        if ref is None:
            return _nan_or_skip(self.name, "no reference to measure precision against")
        sentences = _sentences(ref)
        if not sentences:
            return _nan_or_skip(self.name, "reference has no sentences")
        # Judge which retrieved contexts support the reference (coarse), then score
        # the ranked list by whether top-ranked contexts are actually relevant.
        out = self.judge.complete_json(_judge_user(_f(
            self._JUDGE, reference=ref,
            sentences="\n".join(f"{i}. {s}" for i, s in enumerate(ctx, 1)))))
        verdicts = _verdict_list((out or {}).get("relevant"))
        if not verdicts:
            return CheckResult.error_(self.name, "judge returned no relevance verdicts")
        # Average precision: accumulate relevant hits over the ranked list.
        tp_at_k = 0
        rps = []
        for k, v in enumerate(verdicts, start=1):
            if v:
                tp_at_k += 1
                rps.append(tp_at_k / k)
        if not rps:
            return _cr(self.name, 0.0, self.threshold, details={"relevant": 0})
        # normalize by total relevant in the full list (recall-normalised AP)
        total_relevant = sum(verdicts) or 1
        score = min(1.0, (sum(rps) / len(rps)) * min(1.0, total_relevant / max(1, len(sentences))))
        return _cr(self.name, score, self.threshold,
                   details={"ranked": verdicts, "rps": rps, "total_relevant": total_relevant})


class ContextualRecall(Evaluator):
    name = "contextual_recall"
    kind = "rag"

    _JUDGE = (
        "You are given retrieved context (a set of passages) and a reference answer broken "
        "into sentences. For each reference sentence, decide whether it is FULLY SUPPORTED by "
        "the retrieved context. Return JSON {\"supported\": [0|1, ...]} one entry per sentence.\n"
        "Context:\n{context}\nReference sentences (indexed from 1):\n{sentences}\nJSON:"
    )

    def __init__(self, judge: Optional[JudgeClient] = None, reference: Optional[str] = None,
                 threshold: float = 0.5, name: Optional[str] = None):
        self.judge = _judge_or_default(judge)
        self.reference = reference
        self.threshold = threshold
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        ctx = run.retrieved_context()
        if not ctx:
            return _nan_or_skip(self.name, "no retrieved context")
        ref = self.reference if self.reference is not None else run.reference
        if ref is None:
            return _nan_or_skip(self.name, "no reference")
        sentences = _sentences(ref)
        if not sentences:
            return _nan_or_skip(self.name, "reference has no sentences")
        out = self.judge.complete_json(_judge_user(_f(
            self._JUDGE, context="\n".join(ctx),
            sentences="\n".join(f"{i}. {s}" for i, s in enumerate(sentences, 1)))))
        verdicts = _verdict_list((out or {}).get("supported"))
        if not verdicts:
            return CheckResult.error_(self.name, "judge returned no support verdicts")
        score = sum(verdicts) / len(verdicts)
        return _cr(self.name, score, self.threshold,
                   details={"supported": sum(verdicts), "total": len(verdicts), "verdicts": verdicts})


class ContextualRelevancy(Evaluator):
    name = "contextual_relevancy"
    kind = "rag"

    def __init__(self, embeddings: Optional[Embeddings] = None, threshold: float = 0.2,
                 name: Optional[str] = None):
        self.embeddings = embeddings or Embeddings()
        self.threshold = threshold
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        ctx = run.retrieved_context()
        if not ctx or not run.question:
            return _nan_or_skip(self.name, "no retrieved context or no question")
        q = run.question
        sims = [max(0.0, self.embeddings.similarity(q, c)) for c in ctx]
        score = sum(sims) / len(sims)
        return _cr(self.name, score, self.threshold, details={"similarties": sims})


class ContextEntityRecall(Evaluator):
    name = "context_entity_recall"
    kind = "rag"

    _JUDGE = (
        "Extract entities (names, places, numbers, proper nouns) present in the reference and "
        "in the retrieved contexts. Return JSON {\"reference_entities\": [str,...], "
        "\"context_entities\": [str,...], \"recall_verdicts\": [{\"entity\": str, \"in_context\": 0|1}]}."
        "\nReference: {reference}\nContext:\n{context}\nJSON:"
    )

    def __init__(self, judge: Optional[JudgeClient] = None, reference: Optional[str] = None,
                 threshold: float = 0.5, name: Optional[str] = None):
        self.judge = _judge_or_default(judge)
        self.reference = reference
        self.threshold = threshold
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        ctx = run.retrieved_context()
        ref = self.reference if self.reference is not None else run.reference
        if not ctx or ref is None:
            return _nan_or_skip(self.name, "no context or no reference")
        out = self.judge.complete_json(_judge_user(_f(
            self._JUDGE, reference=ref, context="\n".join(ctx))))
        if not out:
            return CheckResult.error_(self.name, "judge returned no entity verdicts")
        verdicts = _verdict_list((out or {}).get("recall_verdicts"), key="in_context")
        if not verdicts:
            return _cr(self.name, 0.0, self.threshold, details={"entities": 0})
        score = sum(verdicts) / len(verdicts)
        return _cr(self.name, score, self.threshold,
                   details={"entities": len(verdicts), "recalled": sum(verdicts)})


__all__ = [
    "ContextEntityRecall",
    "ContextualPrecision",
    "ContextualRecall",
    "ContextualRelevancy",
    "_sentences",
]
