"""Backward-compatible shim.

The live LLM-judge catalog now lives in :mod:`forjinn_eval.metrics` (one module
per metric, DeepEval-style). This re-exports everything so older imports
(``from forjinn_eval.catalog.llm import Faithfulness``) keep working.
"""
from __future__ import annotations

from ..metrics import (
    AnswerCorrectness,
    AnswerRelevance,
    AnswerRelevancy,
    AnswerRelevancyDeepeval,
    Bias,
    Conformity,
    Contradiction,
    ExactMatch,
    FactualCorrectness,
    Faithfulness,
    GEval,
    GoalAccuracy,
    Groundedness,
    Hallucination,
    LLMJudge,
    NonLLMStringSimilarity,
    PIILeakage,
    PromptAlignment,
    StringPresence,
    TaskCompletion,
    ToolUse,
    TopicAdherence,
    Toxicity,
    default_judge_metrics,
)
from ..metrics._common import (  # noqa: F401
    _OFFLINE_DEFAULTS,
    NAN,
    _cr,
    _f,
    _fbeta,
    _fmt_list,
    _judge_user,
    _nan_or_skip,
    _norm_list,
    _string_similarity,
    _tool_repr,
    _trace_text,
)

__all__ = [
    "AnswerCorrectness",
    "AnswerRelevance",
    "AnswerRelevancy",
    "AnswerRelevancyDeepeval",
    "Bias",
    "Conformity",
    "Contradiction",
    "ExactMatch",
    "FactualCorrectness",
    "Faithfulness",
    "GEval",
    "GoalAccuracy",
    "Groundedness",
    "Hallucination",
    "LLMJudge",
    "NonLLMStringSimilarity",
    "PIILeakage",
    "PromptAlignment",
    "StringPresence",
    "TaskCompletion",
    "ToolUse",
    "TopicAdherence",
    "Toxicity",
    "default_judge_metrics",
]
