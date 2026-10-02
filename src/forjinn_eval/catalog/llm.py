"""Backward-compatible shim.

The live LLM-judge catalog now lives in :mod:`forjinn_eval.metrics` (one module
per metric, DeepEval-style). This re-exports everything so older imports
(``from forjinn_eval.catalog.llm import Faithfulness``) keep working.
"""
from __future__ import annotations

from ..metrics import (  # noqa: F401
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
    Toxicity,
    TopicAdherence,
    default_judge_metrics,
)
from ..metrics._common import (  # noqa: F401
    NAN,
    _OFFLINE_DEFAULTS,
    _cr,
    _fbeta,
    _f,
    _fmt_list,
    _judge_user,
    _nan_or_skip,
    _norm_list,
    _string_similarity,
    _tool_repr,
    _trace_text,
)

__all__ = [
    "Faithfulness",
    "AnswerRelevancy",
    "AnswerRelevancyDeepeval",
    "AnswerCorrectness",
    "FactualCorrectness",
    "TopicAdherence",
    "ExactMatch",
    "StringPresence",
    "NonLLMStringSimilarity",
    "Hallucination",
    "Bias",
    "Toxicity",
    "PIILeakage",
    "TaskCompletion",
    "GoalAccuracy",
    "PromptAlignment",
    "ToolUse",
    "LLMJudge",
    "Groundedness",
    "Contradiction",
    "Conformity",
    "AnswerRelevance",
    "GEval",
    "default_judge_metrics",
]
