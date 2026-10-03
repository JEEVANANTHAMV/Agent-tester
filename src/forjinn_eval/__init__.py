"""forjinn_eval - Unit-test-style evaluation toolkit for Forjinn visual-canvas agents.

Build an agent in Forjinn (host, e.g. 172.16.34.7) via its builder, then treat
each run like a unit test:

    from forjinn_eval import (
        ForjinnClient, agent_test, build_cases, SuiteRunner,
        AllNodesFinished, OutputMatchesRegex, TokenBudget, ToolCallOrder,
    )

    client = ForjinnClient("https://172.16.34.7")   # noproxy on by default

    @agent_test("agent-1-count", tags=["agent-1", "text"])
    def _():
        run = client.predict("03d5abc5-...", "Count 1..5, one per line")
        return run, [
            AllNodesFinished(),
            OutputMatchesRegex(r"(?m)^1\n2\n3\n4\n5$"),
            TokenBudget(output_tokens=20),
        ]

    from forjinn_eval import run_registered
    suite = run_registered(parallel=1)
    print(suite.to_markdown())
    suite.save_json("report.json")           # cumulative set of results
    suite.save_junit("report.xml")           # CI-friendly
"""
from __future__ import annotations

from typing import Optional

from . import (
    catalog,  # noqa: F401  (subpackage; deterministic + LLM catalogs)
    evaluators,  # noqa: F401  (backward-compat shim)
    llm_metrics,  # noqa: F401  (backward-compat shim)
    web_scenarios,
)
from .capture import AgentRun, Node, StreamEvent, parse_sse_text
from .catalog import (
    AllNodesFinished,
    AttachmentParsed,
    AttachmentUploaded,
    AvailableToolsExposed,
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
from .client import ForjinnClient
from .judge import (
    Embeddings,
    ForjinnTransport,
    JudgeClient,
    JudgeError,
    MockJudge,
    _cosine,  # noqa: F401  (embedding cosine helper)
    extract_json,
    queuing_judge,
    set_default_judge,
)
from .metrics import (
    AgentLoopDetection,
    AllOf,
    AnswerAccuracy,
    AnswerCorrectness,
    AnswerRelevance,
    AnswerRelevancy,
    AnswerRelevancyDeepeval,
    AnswerSimilarity,
    AnyOf,
    ArgumentCorrectness,
    Bias,
    BleuScore,
    ChrfScore,
    CitationFaithfulness,
    Conformity,
    ContextEntityRecall,
    ContextualPrecision,
    ContextualRecall,
    ContextualRelevancy,
    Contradiction,
    ConversationalGEval,
    ConversationCompleteness,
    ExactMatch,
    FactualCorrectness,
    Faithfulness,
    GEval,
    GoalAccuracy,
    GoalAccuracyMulti,
    Groundedness,
    Hallucination,
    KnowledgeRetention,
    LLMJudge,
    Misuse,
    MultiTurnToolUse,
    MultiTurnTopicAdherence,
    NoiseSensitivity,
    NonAdvice,
    NonLLMStringSimilarity,
    Not,
    PatternMatch,
    PIILeakage,
    PlanAdherence,
    PlanQuality,
    PromptAlignment,
    QuotedSpansAlignment,
    RoleAdherence,
    RoleViolation,
    RougeScore,
    SemanticSimilarity,
    StepEfficiency,
    StringPresence,
    Summarization,
    TaskCompletion,
    ToolCallAccuracy,
    ToolCorrectness,
    ToolUse,
    TopicAdherence,
    Toxicity,
    TurnContextualPrecision,
    TurnContextualRecall,
    TurnContextualRelevancy,
    TurnFaithfulness,
    TurnRelevancy,
    agent_judge_metrics,
    default_judge_metrics,
    rag_judge_metrics,
    safety_judge_metrics,
)
from .results import CheckResult, Metric, Status, rollup
from .suite import (
    AgentTestCase,
    SuiteResult,
    SuiteRunner,
    TestCaseResult,
    agent_test,
    build_cases,
    make_case,
    registered_cases,
)
from .types import Conversation, ForjinnError, Message, ToolCall, UsageMetadata
from .web_scenarios import run_all_scenarios


def _resolve_version() -> str:
    """Resolve ``__version__``: prefer the build-generated ``_version.py``
    (written by setuptools-scm from the latest git tag), then a static
    fallback so the package imports cleanly from an untagged checkout or an
    sdist without git metadata."""
    try:
        from ._version import version as _v  # type: ignore[attr-defined]  # written at build

        return _v
    except Exception:  # pragma: no cover - build file absent in some flows
        return "0.1.0.dev0"


__version__ = _resolve_version()


def run_registered(
    names=None,
    parallel: int = 1,
    suite_name: str = "forjinn-eval",
    min_pass_rate: Optional[float] = None,
) -> SuiteResult:
    """Run all registered :func:`agent_test` cases and return a :class:`SuiteResult`.

    If ``min_pass_rate`` is set and the suite's pass rate is below it, the
    returned :class:`SuiteResult` still reflects the true result but a
    :class:`ForjinnError` is raised - useful as a CI gate.
    """
    cases = build_cases(names)
    sr = SuiteRunner(parallel=parallel, suite_name=suite_name).run(cases)
    if min_pass_rate is not None and (sr.pass_rate is None or sr.pass_rate < min_pass_rate):
        raise ForjinnError(
            f"suite pass rate {sr.pass_rate:.2f} below gate {min_pass_rate:.2f} "
            f"(passed={sr.passed}, failed={sr.failed}, errors={sr.errors})"
        )
    return sr


__all__ = [
    "__version__",
    # client
    "ForjinnClient",
    "ForjinnError",
    # LLM judge
    "JudgeClient",
    "ForjinnTransport",
    "JudgeError",
    "MockJudge",
    "queuing_judge",
    "set_default_judge",
    "extract_json",
    "Embeddings",
    # capture
    "AgentRun",
    "Node",
    "StreamEvent",
    "parse_sse_text",
    "ToolCall",
    "UsageMetadata",
    "Conversation",
    "Message",
    # results
    "CheckResult",
    "Metric",
    "Status",
    "rollup",
    # evaluators
    "Evaluator",
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
    "NoCostLeakage",
    "OutputLengthBounds",
    "OutputJsonValid",
    "AttachmentParsed",
    "AttachmentUploaded",
    # LLM-judged metric catalog
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
    # new RAG (retriever + citations)
    "ContextualPrecision",
    "ContextualRecall",
    "ContextualRelevancy",
    "ContextEntityRecall",
    "CitationFaithfulness",
    "QuotedSpansAlignment",
    "AnswerAccuracy",
    "NoiseSensitivity",
    # new deterministic text + loop + tool
    "PatternMatch",
    "BleuScore",
    "RougeScore",
    "ChrfScore",
    "SemanticSimilarity",
    "AnswerSimilarity",
    "AgentLoopDetection",
    "ToolCallAccuracy",
    "ToolCorrectness",
    "ArgumentCorrectness",
    # new safety / misbehaviour
    "Misuse",
    "NonAdvice",
    "RoleViolation",
    "RoleAdherence",
    # new agent / plan / summarisation
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
    # composition
    "AllOf",
    "AnyOf",
    "Not",
    # batteries
    "default_judge_metrics",
    "safety_judge_metrics",
    "rag_judge_metrics",
    "agent_judge_metrics",
    # web-scenario battery (open forjinn web to all scenarios)
    "web_scenarios",
    "run_all_scenarios",
    # suite
    "AgentTestCase",
    "TestCaseResult",
    "SuiteResult",
    "SuiteRunner",
    "agent_test",
    "build_cases",
    "make_case",
    "registered_cases",
    "run_registered",
]
