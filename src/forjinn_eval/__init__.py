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

from typing import Optional  # noqa: E402

from . import catalog  # noqa: F401  (subpackage; deterministic + LLM catalogs)
from . import evaluators  # noqa: F401  (backward-compat shim)
from . import llm_metrics  # noqa: F401  (backward-compat shim)
from .capture import AgentRun, Node, StreamEvent, parse_sse_text
from .client import ForjinnClient
from .catalog import (
    AllNodesFinished,
    AvailableToolsExposed,
    AttachmentParsed,
    AttachmentUploaded,
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
from .judge import ForjinnTransport, JudgeClient, JudgeError, MockJudge, extract_json
from .metrics import (
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
from .types import ForjinnError, ToolCall, UsageMetadata

__version__ = "0.1.0"


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
    "extract_json",
    # capture
    "AgentRun",
    "Node",
    "StreamEvent",
    "parse_sse_text",
    "ToolCall",
    "UsageMetadata",
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
    "default_judge_metrics",
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
