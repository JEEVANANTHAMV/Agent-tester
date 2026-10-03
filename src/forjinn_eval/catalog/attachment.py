"""Attachment evaluators: parsed file/image assertions (Flowise-like)."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Optional

from ..capture import AgentRun
from .base import CheckResult, Evaluator

__all__ = ["AttachmentParsed", "AttachmentUploaded"]


class AttachmentParsed(Evaluator):
    """Assert the parsed attachment payload (injected via ``run.raw``) is non-empty
    and contains the expected marker substring(s)."""

    name = "attachment_parsed"
    kind = "attachment"

    def __init__(self, contains: Optional[Iterable[str]] = None, min_items: int = 1):
        self.contains = list(contains or [])
        self.min_items = min_items

    def _parsed(self, run: AgentRun) -> list:
        raw = run.raw or {}
        return raw.get("parsed_attachment") or []

    def evaluate(self, run: AgentRun) -> CheckResult:
        parsed = self._parsed(run)
        if not parsed:
            return CheckResult.fail_(
                self.name,
                "attachment parse returned no items (upload may not have happened)",
                score=0.0,
            )
        if len(parsed) < self.min_items:
            return CheckResult.fail_(
                self.name,
                f"only {len(parsed)} parsed item(s), expected >= {self.min_items}",
                score=len(parsed) / self.min_items,
            )
        flat = str(parsed)
        missing = [s for s in self.contains if s not in flat]
        if missing:
            return CheckResult.fail_(
                self.name,
                f"parsed attachment missing marker(s): {missing}",
                score=1.0 - len(missing) / len(self.contains),
            )
        return CheckResult.pass_(self.name, f"{len(parsed)} parsed item(s), all {len(self.contains)} markers present")


class AttachmentUploaded(Evaluator):
    """Assert the raw dict records that an attachment upload happened."""

    name = "attachment_uploaded"
    kind = "attachment"

    def __init__(self, filenames: Optional[Iterable[str]] = None):
        self.filenames = list(filenames or [])

    def evaluate(self, run: AgentRun) -> CheckResult:
        uploaded = (run.raw or {}).get("uploaded_files")
        if not uploaded:
            return CheckResult.fail_(self.name, "no uploaded_files recorded in run.raw")
        missing = [f for f in self.filenames if f not in uploaded]
        if missing:
            return CheckResult.fail_(self.name, f"expected uploads missing: {missing}")
        return CheckResult.pass_(self.name, f"uploads present: {sorted(uploaded)}")
