"""Backward-compatible shim.

The deterministic evaluator catalog now lives in :mod:`forjinn_eval.catalog`
(split by concern: structural / performance / tool / content / safety /
attachment) and the LLM-judged catalog in :mod:`forjinn_eval.catalog.llm`.
This module re-exports everything from those so existing imports
(``from forjinn_eval.evaluators import ...``) keep working.

Prefer importing from the top-level package for new code::

    from forjinn_eval import AllNodesFinished, TokenBudget, Faithfulness
"""
from __future__ import annotations

from .catalog import (  # noqa: F401
    AllNodesFinished,
    AttachmentParsed,
    AttachmentUploaded,
    AvailableToolsExposed,
    CheckResult,
    Evaluator,
    ExpectedNodeCount,
    LatencyBudget,
    ModelIs,
    NoCostLeakage,
    NoToolsExpected,
    OnlyAllowedTools,
    OutputContains,
    OutputDoesNotContain,
    OutputJsonValid,
    OutputLengthBounds,
    OutputMatchesRegex,
    OutputNotEmpty,
    RequiredNodesPresent,
    StartNodePassthrough,
    TokenBudget,
    ToolCallCount,
    ToolCallOrder,
    ToolCallSetF1,
    default_structural,
)

__all__ = [
    "Evaluator",
    "CheckResult",
    "AllNodesFinished",
    "RequiredNodesPresent",
    "ExpectedNodeCount",
    "ModelIs",
    "StartNodePassthrough",
    "TokenBudget",
    "LatencyBudget",
    "ToolCallOrder",
    "ToolCallSetF1",
    "OnlyAllowedTools",
    "ToolCallCount",
    "NoToolsExpected",
    "AvailableToolsExposed",
    "OutputNotEmpty",
    "OutputMatchesRegex",
    "OutputContains",
    "OutputDoesNotContain",
    "OutputLengthBounds",
    "OutputJsonValid",
    "NoCostLeakage",
    "AttachmentParsed",
    "AttachmentUploaded",
    "default_structural",
]
