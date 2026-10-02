"""Deterministic reference-based string metrics (ragas _string.py + similarity):
``ExactMatch`` / ``StringPresence`` / ``NonLLMStringSimilarity``. No LLM judge
required; they read ``run.reference`` (or an explicit ``reference``)."""
from __future__ import annotations

from typing import Optional

from ..capture import AgentRun
from .base import CheckResult, Evaluator
from ._common import _cr, _nan_or_skip, _string_similarity


class ExactMatch(Evaluator):
    name = "exact_match"
    kind = "rag"

    def __init__(self, reference: Optional[str] = None, threshold: float = 1.0,
                 name: Optional[str] = None):
        self.reference = reference
        self.threshold = threshold
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        ref = self.reference if self.reference is not None else run.reference
        if ref is None:
            return _nan_or_skip(self.name, "no reference")
        score = 1.0 if (run.text or "").strip() == str(ref).strip() else 0.0
        return _cr(self.name, score, self.threshold, details={"expected": ref})


class StringPresence(Evaluator):
    name = "string_presence"
    kind = "rag"

    def __init__(self, reference: Optional[str] = None, threshold: float = 1.0,
                 case_sensitive: bool = True, name: Optional[str] = None):
        self.reference = reference
        self.threshold = threshold
        self.case_sensitive = case_sensitive
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        ref = self.reference if self.reference is not None else run.reference
        if ref is None:
            return _nan_or_skip(self.name, "no reference substring")
        hay, needle = run.text or "", str(ref)
        if not self.case_sensitive:
            hay, needle = hay.lower(), needle.lower()
        score = 1.0 if needle in hay else 0.0
        return _cr(self.name, score, self.threshold, details={"expected": ref})


class NonLLMStringSimilarity(Evaluator):
    name = "string_similarity"
    kind = "rag"

    def __init__(self, reference: Optional[str] = None, threshold: float = 0.5,
                 distance_measure: str = "levenshtein", name: Optional[str] = None):
        self.reference = reference
        self.threshold = threshold
        self.distance_measure = distance_measure
        if name:
            self.name = name

    def evaluate(self, run: AgentRun) -> CheckResult:
        ref = self.reference if self.reference is not None else run.reference
        if ref is None:
            return _nan_or_skip(self.name, "no reference")
        score = _string_similarity(run.text or "", str(ref))
        return _cr(self.name, score, self.threshold, details={"score": score})


__all__ = ["ExactMatch", "StringPresence", "NonLLMStringSimilarity"]
