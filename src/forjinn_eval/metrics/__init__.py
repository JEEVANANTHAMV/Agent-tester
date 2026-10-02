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

from .answer_correctness import AnswerCorrectness
from .answer_relevance import AnswerRelevance
from .answer_relevancy import AnswerRelevancy
from .answer_relevancy_deepeval import AnswerRelevancyDeepeval
from .bias import Bias
from .conformity import Conformity
from .contradiction import Contradiction
from .factual_correctness import FactualCorrectness
from .faithfulness import Faithfulness
from .g_eval import GEval
from .goal_accuracy import GoalAccuracy
from .groundedness import Groundedness
from .hallucination import Hallucination
from .llm_judge import LLMJudge
from .pii_leakage import PIILeakage
from .prompt_alignment import PromptAlignment
from .string_metrics import ExactMatch, NonLLMStringSimilarity, StringPresence
from .task_completion import TaskCompletion
from .tool_use import ToolUse
from .toxicity import Toxicity
from .topic_adherence import TopicAdherence

from .base import Evaluator, CheckResult  # noqa: F401  (re-exported)
from ._common import (
    _OFFLINE_DEFAULTS,
    _string_similarity,
)
from ..judge import JudgeClient  # noqa: F401  (the judge object type)


def default_judge_metrics(
    threshold: float = 0.5, judge: Optional[JudgeClient] = None
) -> List[Evaluator]:
    """A representative LLM-judge battery (skips gracefully without a judge)."""
    return [
        AnswerRelevancy(judge=judge, threshold=threshold),
        Faithfulness(judge=judge, threshold=threshold),
        Hallucination(judge=judge, threshold=threshold),
        PromptAlignment(judge=judge, threshold=threshold),
        Bias(judge=judge, threshold=threshold),
        Toxicity(judge=judge, threshold=threshold),
    ]


__all__ = [
    # RAG-assembled
    "Faithfulness",
    "AnswerRelevancy",
    "AnswerRelevancyDeepeval",
    "AnswerCorrectness",
    "FactualCorrectness",
    "TopicAdherence",
    # deterministic string / reference
    "ExactMatch",
    "StringPresence",
    "NonLLMStringSimilarity",
    # safety / leak
    "Hallucination",
    "Bias",
    "Toxicity",
    "PIILeakage",
    # agent / task
    "TaskCompletion",
    "GoalAccuracy",
    "PromptAlignment",
    "ToolUse",
    # giskard judges
    "LLMJudge",
    "Groundedness",
    "Contradiction",
    "Conformity",
    "AnswerRelevance",
    # custom rubric
    "GEval",
    # convenience
    "default_judge_metrics",
]
