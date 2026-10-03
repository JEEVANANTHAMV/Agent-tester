"""Backward-compatible shim for the LLM-judged catalog.

The live module is :mod:`forjinn_eval.metrics` (one metric per module,
DeepEval-style). This re-exports every LLM judge + metric so existing imports
(``from forjinn_eval.llm_metrics import Faithfulness``) keep working.
"""
from __future__ import annotations

from .metrics import (
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
