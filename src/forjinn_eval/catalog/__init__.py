"""Evaluator catalog.

Deterministic evaluators (structural / performance / tool / content / safety /
attachment) and the LLM-judged catalog (``catalog.llm``). The root
:mod:`forjinn_eval` package re-exports all of these for a flat import surface.
"""
from __future__ import annotations

from .attachment import AttachmentParsed, AttachmentUploaded
from .base import CheckResult, Evaluator
from .content import (
    OutputContains,
    OutputDoesNotContain,
    OutputJsonValid,
    OutputLengthBounds,
    OutputMatchesRegex,
    OutputNotEmpty,
)
from .content import OutputNotEmpty as _OutputNotEmpty
from .performance import LatencyBudget, TokenBudget
from .safety import NoCostLeakage
from .safety import NoCostLeakage as _NoCostLeakage
from .structural import (
    AllNodesFinished,
    ExpectedNodeCount,
    ModelIs,
    RequiredNodesPresent,
    StartNodePassthrough,
)

# Deterministic default battery (kept next to the structural evaluators)
from .structural import AllNodesFinished as _AllNodesFinished
from .tool import (
    AvailableToolsExposed,
    NoToolsExpected,
    OnlyAllowedTools,
    ToolCallCount,
    ToolCallOrder,
    ToolCallSetF1,
)


def default_structural() -> list:
    """A small, deterministic, CI-safe default battery."""
    return [_AllNodesFinished(), _OutputNotEmpty(), _NoCostLeakage()]


# LLM-judged catalog is imported lazily by the root package to avoid a hard
# dependency cycle at import time (it imports only this package, so it is safe
# to import here, but we keep the symbol surface explicit).
from . import llm  # noqa: E402

__all__ = [
    "Evaluator",
    "CheckResult",
    # structural
    "AllNodesFinished",
    "RequiredNodesPresent",
    "ExpectedNodeCount",
    "ModelIs",
    "StartNodePassthrough",
    # performance
    "TokenBudget",
    "LatencyBudget",
    # tool
    "ToolCallOrder",
    "ToolCallSetF1",
    "OnlyAllowedTools",
    "ToolCallCount",
    "NoToolsExpected",
    "AvailableToolsExposed",
    # content
    "OutputNotEmpty",
    "OutputMatchesRegex",
    "OutputContains",
    "OutputDoesNotContain",
    "OutputLengthBounds",
    "OutputJsonValid",
    # safety
    "NoCostLeakage",
    # attachment
    "AttachmentParsed",
    "AttachmentUploaded",
    # convenience
    "default_structural",
    # submodules / namespace
    "llm",  # shim -> forjinn_eval.metrics
]
