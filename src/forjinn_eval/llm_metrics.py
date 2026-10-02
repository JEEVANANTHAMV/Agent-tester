"""Backward-compatible shim for the LLM-judged catalog.

The live module is :mod:`forjinn_eval.metrics` (one metric per module,
DeepEval-style). This re-exports every LLM judge + metric so existing imports
(``from forjinn_eval.llm_metrics import Faithfulness``) keep working.
"""
from __future__ import annotations

from .metrics import (  # noqa: F401
    GEval,
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
    "PromptAlignment",
    "TaskCompletion",
    "GoalAccuracy",
    "PIILeakage",
    "ToolUse",
    "GEval",
    "LLMJudge",
    "Groundedness",
    "Contradiction",
    "Conformity",
    "AnswerRelevance",
    "default_judge_metrics",
]
