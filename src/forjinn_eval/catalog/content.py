"""Content evaluators: non-empty, regex, contains, length, JSON validity."""
from __future__ import annotations

import re
from collections.abc import Iterable
from re import Pattern
from typing import Any, Dict, List, Optional, Union

from ..capture import AgentRun
from .base import CheckResult, Evaluator

__all__ = [
    "OutputContains",
    "OutputDoesNotContain",
    "OutputJsonValid",
    "OutputLengthBounds",
    "OutputMatchesRegex",
    "OutputNotEmpty",
]


class OutputNotEmpty(Evaluator):
    """Final text must be non-empty (after strip)."""

    name = "output_not_empty"
    kind = "content"

    def evaluate(self, run: AgentRun) -> CheckResult:
        if run.text and run.text.strip():
            return CheckResult.pass_(self.name, f"len={len(run.text.strip())}")
        return CheckResult.fail_(self.name, "final text is empty", score=0.0)


class OutputMatchesRegex(Evaluator):
    """Final text matches a regex (``match`` defaults to ``search``)."""

    name = "output_matches_regex"
    kind = "content"

    def __init__(self, pattern: str, match: str = "search", fullmatch: bool = False):
        self.pattern = pattern
        self.match = match  # 'search' | 'full' (kept for back-compat alias)
        self.fullmatch = fullmatch or match == "full"

    def evaluate(self, run: AgentRun) -> CheckResult:
        if not run.text:
            return CheckResult.error_(self.name, "no output text")
        try:
            if self.fullmatch:
                ok = re.fullmatch(self.pattern, run.text.strip()) is not None
            else:
                ok = re.search(self.pattern, run.text, re.DOTALL) is not None
        except re.error as e:
            return CheckResult.error_(self.name, f"bad regex: {e}")
        if ok:
            return CheckResult.pass_(self.name, f"regex {self.pattern!r} matches")
        return CheckResult.fail_(self.name, f"regex {self.pattern!r} did not match", score=0.0)


class OutputContains(Evaluator):
    """Final text contains all (or any) expected substrings, case-insensitive by default."""

    name = "output_contains"
    kind = "content"

    def __init__(self, substrings: Iterable[str], match: str = "all", case_sensitive: bool = False):
        self.substrings = list(substrings)
        self.match = match
        self.case_sensitive = case_sensitive

    def evaluate(self, run: AgentRun) -> CheckResult:
        if not run.text:
            return CheckResult.error_(self.name, "no output text")
        hay = run.text if self.case_sensitive else run.text.lower()

        def present(s: str) -> bool:
            return (s if self.case_sensitive else s.lower()) in hay

        if self.match == "any":
            # PASS iff at least one substring is present; FAIL only if none are.
            found = [s for s in self.substrings if present(s)]
            if found:
                return CheckResult.pass_(self.name, f"at least one of {self.substrings} present: {found}")
            return CheckResult.fail_(self.name, f"none of {self.substrings} present", score=0.0)
        # match == "all": PASS iff every substring is present; else FAIL with a 0..1 partial score.
        missing = [s for s in self.substrings if not present(s)]
        if missing:
            total = len(self.substrings)
            score = (total - len(missing)) / total if total else 0.0
            return CheckResult.fail_(self.name, f"missing substrings: {missing}", score=score)
        return CheckResult.pass_(self.name, f"all {len(self.substrings)} substrings present")


class OutputDoesNotContain(Evaluator):
    """Final text must NOT contain any of the forbidden substrings."""

    name = "output_does_not_contain"
    kind = "content"

    def __init__(self, forbidden: Iterable[Union[str, Pattern]]):
        self.forbidden = list(forbidden)

    def evaluate(self, run: AgentRun) -> CheckResult:
        if not run.text:
            return CheckResult.skip_(self.name, "no output text")
        hay = run.text
        found: List[str] = []
        for f in self.forbidden:
            if isinstance(f, str):
                if f in hay:
                    found.append(f)
            else:  # Pattern
                if f.search(hay):
                    found.append(f.pattern)
        if found:
            return CheckResult.fail_(self.name, f"forbidden substrings found: {found}", score=0.0)
        return CheckResult.pass_(self.name, f"none of {len(self.forbidden)} forbidden found")


class OutputLengthBounds(Evaluator):
    """Final text length (chars) must be within [min, max]."""

    name = "output_length_bounds"
    kind = "content"

    def __init__(self, min_chars: int = 0, max_chars: Optional[int] = None):
        self.min_chars = min_chars
        self.max_chars = max_chars

    def evaluate(self, run: AgentRun) -> CheckResult:
        n = len(run.text or "")
        lo_ok = n >= self.min_chars
        hi_ok = self.max_chars is None or n <= self.max_chars
        if lo_ok and hi_ok:
            return CheckResult.pass_(self.name, f"chars={n}")
        return CheckResult.fail_(self.name, f"chars={n} outside [{self.min_chars}, {self.max_chars or 'inf'}]")


class OutputJsonValid(Evaluator):
    """Final text must parse as JSON (optionally conform to a JSON Schema)."""

    name = "output_json_valid"
    kind = "content"

    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        self.schema = schema

    def evaluate(self, run: AgentRun) -> CheckResult:
        if not run.text:
            return CheckResult.error_(self.name, "no output text")
        import json as _json

        try:
            obj = _json.loads(run.text)
        except _json.JSONDecodeError as e:
            return CheckResult.fail_(self.name, f"not valid JSON: {e}", score=0.0)
        if self.schema is not None:
            try:
                import jsonschema  # optional dep
            except ImportError:
                return CheckResult.error_(self.name, "jsonschema not installed; cannot validate schema")
            try:
                jsonschema.validate(obj, self.schema)
            except Exception as e:
                return CheckResult.fail_(self.name, f"schema violation: {e}", score=0.0)
        return CheckResult.pass_(self.name, "valid JSON")
