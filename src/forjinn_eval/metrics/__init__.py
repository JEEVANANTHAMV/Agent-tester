"""LLM-judged metric catalog (one metric per module, DeepEval-style).

Each metric is a tiny, self-contained class (the "metric is the atom" pattern):
a declared ``name``/``kind`` and a single ``evaluate(run) -> CheckResult``. They
are grouped by origin:

* **RAG-assembled** (ragas + deepeval): :mod:`.faithfulness`,
  :mod:`.answer_relevancy`, :mod:`.answer_correctness`, :mod:`.factual_correctness`,
  :mod:`.topic_adherence`, :mod:`.answer_relevancy_deepeval`
* **Safety / leak** (deepeval): :mod:`.bias`, :mod:`.toxicity`, :mod:`.pii_leakage`,
  :mod:`.hallucination`
* **Agent / task** (deepeval + ragas): :mod:`.task_completion`, :mod:`.goal_accuracy`,
  :mod:`.prompt_alignment`, :mod:`.tool_use`
* **Giskard judges**: :mod:`.llm_judge`, :mod:`.groundedness`, :mod:`.contradiction`,
  :mod:`.conformity`, :mod:`.answer_relevance`
* **Custom rubric**: :mod:`.g_eval` (build-your-own escape hatch)
* **Deterministic reference**: :mod:`.string_metrics` (no judge required)

Shared plumbing lives in :mod:`._common`; the base is :mod:`.base`. The judge
itself is a :class:`forjinn_eval.judge.JudgeClient` (a Forjinn chatflow over the
Forjinn API).
"""
from __future__ import annotations

from typing import List, Optional

from ..judge import Embeddings, JudgeClient
from ._common import (
    _OFFLINE_DEFAULTS,
    _string_similarity,
)
from .answer_correctness import AnswerCorrectness
from .answer_quality import AnswerAccuracy, NoiseSensitivity
from .answer_relevance import AnswerRelevance
from .answer_relevancy import AnswerRelevancy
from .answer_relevancy_deepeval import AnswerRelevancyDeepeval
from .base import CheckResult, Evaluator  # noqa: F401  (re-exported)
from .bias import Bias
from .citations import CitationFaithfulness, QuotedSpansAlignment
from .composition import AllOf, AnyOf, Not
from .conformity import Conformity
from .contextual import (
    ContextEntityRecall,
    ContextualPrecision,
    ContextualRecall,
    ContextualRelevancy,
)
from .contradiction import Contradiction
from .conversational import (
    ConversationalGEval,
    GoalAccuracyMulti,
    MultiTurnToolUse,
    MultiTurnTopicAdherence,
    TurnContextualPrecision,
    TurnContextualRecall,
    TurnContextualRelevancy,
    TurnFaithfulness,
    TurnRelevancy,
)
from .factual_correctness import FactualCorrectness
from .faithfulness import Faithfulness
from .g_eval import GEval
from .goal_accuracy import GoalAccuracy
from .groundedness import Groundedness
from .hallucination import Hallucination
from .llm_judge import LLMJudge
from .loop_detection import AgentLoopDetection
from .misbehaviour import Misuse, NonAdvice, RoleViolation
from .pii_leakage import PIILeakage
from .plan import (
    ConversationCompleteness,
    KnowledgeRetention,
    PlanAdherence,
    PlanQuality,
    RoleAdherence,
    StepEfficiency,
)
from .prompt_alignment import PromptAlignment
from .string_metrics import ExactMatch, NonLLMStringSimilarity, StringPresence
from .summarization import Summarization
from .task_completion import TaskCompletion
from .text_metrics import (
    AnswerSimilarity,
    BleuScore,
    ChrfScore,
    PatternMatch,
    RougeScore,
    SemanticSimilarity,
)
from .tool_correctness import ArgumentCorrectness, ToolCorrectness
from .tool_deterministic import ToolCallAccuracy
from .tool_use import ToolUse
from .topic_adherence import TopicAdherence
from .toxicity import Toxicity


def default_judge_metrics(
    threshold: float = 0.5, judge: Optional[JudgeClient] = None
) -> List[Evaluator]:
    """A representative single-turn LLM-judge battery (skips gracefully without a judge)."""
    return [
        AnswerRelevancy(judge=judge, threshold=threshold),
        Faithfulness(judge=judge, threshold=threshold),
        Hallucination(judge=judge, threshold=threshold),
        PromptAlignment(judge=judge, threshold=threshold),
        Bias(judge=judge, threshold=threshold),
        Toxicity(judge=judge, threshold=threshold),
    ]


def safety_judge_metrics(threshold: float = 0.5, judge: Optional[JudgeClient] = None) -> List[Evaluator]:
    """Safety / misbehaviour battery."""
    return [
        Bias(judge=judge, threshold=threshold),
        Toxicity(judge=judge, threshold=threshold),
        PIILeakage(judge=judge, threshold=threshold),
        Misuse(judge=judge, threshold=threshold),
        NonAdvice(judge=judge, threshold=threshold),
    ]


def rag_judge_metrics(threshold: float = 0.5, judge: Optional[JudgeClient] = None,
                      embeddings: Optional[Embeddings] = None) -> List[Evaluator]:
    """Full RAG battery (answer + retriever quality)."""
    return [
        Faithfulness(judge=judge, threshold=threshold),
        AnswerRelevancy(judge=judge, threshold=threshold),
        AnswerAccuracy(judge=judge, threshold=threshold),
        NoiseSensitivity(judge=judge, threshold=threshold),
        ContextualPrecision(judge=judge, threshold=threshold),
        ContextualRecall(judge=judge, threshold=threshold),
        ContextualRelevancy(embeddings=embeddings, threshold=threshold),
        CitationFaithfulness(judge=judge, threshold=threshold),
    ]


def agent_judge_metrics(threshold: float = 0.5, judge: Optional[JudgeClient] = None) -> List[Evaluator]:
    """Agent-behaviour battery."""
    return [
        TaskCompletion(judge=judge, threshold=threshold),
        GoalAccuracy(judge=judge, threshold=threshold),
        PlanAdherence(judge=judge, threshold=threshold),
        PlanQuality(judge=judge, threshold=threshold),
        StepEfficiency(judge=judge, threshold=threshold),
        ConversationCompleteness(judge=judge, threshold=threshold),
        KnowledgeRetention(judge=judge, threshold=threshold),
        AgentLoopDetection(threshold=threshold),
    ]


__all__ = [
    # RAG-assembled (answer side)
    "Faithfulness",
    "AnswerRelevancy",
    "AnswerRelevancyDeepeval",
    "AnswerCorrectness",
    "FactualCorrectness",
    "AnswerAccuracy",
    "NoiseSensitivity",
    "TopicAdherence",
    # RAG-assembled (retriever side)
    "ContextualPrecision",
    "ContextualRecall",
    "ContextualRelevancy",
    "ContextEntityRecall",
    "CitationFaithfulness",
    "QuotedSpansAlignment",
    # deterministic string / reference / text-quality
    "ExactMatch",
    "StringPresence",
    "NonLLMStringSimilarity",
    "PatternMatch",
    "BleuScore",
    "RougeScore",
    "ChrfScore",
    "SemanticSimilarity",
    "AnswerSimilarity",
    "AgentLoopDetection",
    # tool (LLM + deterministic)
    "ToolUse",
    "ToolCorrectness",
    "ArgumentCorrectness",
    "ToolCallAccuracy",
    # safety / misbehaviour / leak
    "Hallucination",
    "Bias",
    "Toxicity",
    "PIILeakage",
    "Misuse",
    "NonAdvice",
    "RoleViolation",
    "RoleAdherence",
    # agent / task / plan
    "TaskCompletion",
    "GoalAccuracy",
    "PromptAlignment",
    "Summarization",
    "PlanAdherence",
    "PlanQuality",
    "StepEfficiency",
    "ConversationCompleteness",
    "KnowledgeRetention",
    # multi-turn conversational
    "TurnFaithfulness",
    "TurnRelevancy",
    "TurnContextualPrecision",
    "TurnContextualRecall",
    "TurnContextualRelevancy",
    "MultiTurnTopicAdherence",
    "MultiTurnToolUse",
    "ConversationalGEval",
    "GoalAccuracyMulti",
    # giskard judges
    "LLMJudge",
    "Groundedness",
    "Contradiction",
    "Conformity",
    "AnswerRelevance",
    # custom rubric + composition
    "GEval",
    "AllOf",
    "AnyOf",
    "Not",
    # embeddings
    "Embeddings",
    # convenience batteries
    "default_judge_metrics",
    "safety_judge_metrics",
    "rag_judge_metrics",
    "agent_judge_metrics",
]
