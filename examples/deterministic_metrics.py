"""deterministic_metrics.py - a tour of the deterministic (no-LLM) catalog.

RUN IT (fully offline - recorded fixtures, no network):

    forjinn-eval run examples/deterministic_metrics.py

Each case demonstrates one family of deterministic evaluator. Deterministic
metrics are the CI baseline: fast, reproducible, and safe to gate on. No judge,
host or token is needed.
"""
from __future__ import annotations

import json
from pathlib import Path

from forjinn_eval import (
    AgentRun,
    AgentLoopDetection,
    AllNodesFinished,
    ExpectedNodeCount,
    ModelIs,
    NoToolsExpected,
    NonLLMStringSimilarity,
    OutputContains,
    OutputDoesNotContain,
    OutputJsonValid,
    OutputLengthBounds,
    OutputMatchesRegex,
    OutputNotEmpty,
    OnlyAllowedTools,
    RequiredNodesPresent,
    StartNodePassthrough,
    TokenBudget,
    ToolCallSetF1,
    ToolCallOrder,
    agent_test,
)

HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parent / "tests" / "fixtures"


def _run(fixture: str) -> AgentRun:
    payload = json.loads((FIXTURES / fixture).read_text(encoding="utf-8"))
    return AgentRun.from_nonstream(fixture, payload)


# ---------------------------------------------------------------------------
# structural
# ---------------------------------------------------------------------------
@agent_test("structural-nodes", tags=["deterministic", "structural"])
def structural_nodes():
    """Every node finished and the expected roles are present."""
    run = _run("agent1_nonstream.json")
    return run, [
        AllNodesFinished(),
        RequiredNodesPresent(["start", "agent"]),
        ExpectedNodeCount(2, at_least=True),
        StartNodePassthrough(),
    ]


# ---------------------------------------------------------------------------
# exact content match
# ---------------------------------------------------------------------------
@agent_test("content-regex", tags=["deterministic", "content"])
def content_regex():
    """The answer matches the expected pattern exactly."""
    run = _run("agent1_count.json")        # answer: 1\n2\n3\n4\n5
    return run, [
        AllNodesFinished(),
        OutputNotEmpty(),
        OutputMatchesRegex(r"(?m)^1\n2\n3\n4\n5$"),
        OutputLengthBounds(min_chars=5, max_chars=20),
    ]


# ---------------------------------------------------------------------------
# content containment / absence
# ---------------------------------------------------------------------------
@agent_test("content-presence", tags=["deterministic", "content"])
def content_presence():
    """Required words are present and forbidden words are absent."""
    run = _run("agent1_nonstream.json")
    return run, [
        AllNodesFinished(),
        OutputContains(["operate", "capacity"], match="any"),
        OutputDoesNotContain(["sorry i cannot", "i don't know"]),
    ]


# ---------------------------------------------------------------------------
# tool-correctness (deterministic)
# ---------------------------------------------------------------------------
@agent_test("tools-guarded", tags=["deterministic", "tools"])
def tools_guarded():
    """This agent is not expected to call tools (stray calls would fail)."""
    run = _run("agent1_nonstream.json")    # no tool calls in this capture
    return run, [
        AllNodesFinished(),
        NoToolsExpected(),                 # guard: no stray tool calls
        OnlyAllowedTools({"search_db", "get_price"}),  # allow-list (vacuously true here)
    ]


# ---------------------------------------------------------------------------
# text-quality vs a reference (no LLM)
# ---------------------------------------------------------------------------
@agent_test("text-quality", tags=["deterministic", "text-quality"])
def text_quality():
    """The answer is close to a reference string (no LLM)."""
    run = _run("agent1_count.json")
    run.raw["reference"] = "1\n2\n3\n4\n5"
    return run, [
        AllNodesFinished(),
        NonLLMStringSimilarity(threshold=0.6),
        AgentLoopDetection(),
    ]


# ---------------------------------------------------------------------------
# JSON output shape
# ---------------------------------------------------------------------------
@agent_test("output-json", tags=["deterministic", "content"])
def output_json():
    """OutputJsonValid is a *contract*: the output MUST be JSON or the check FAILs.

    (Unlike LLM metrics, which SKIP on missing input, a JSON-shape assertion is a
    positive requirement - a non-JSON run is a real failure.)
    """
    run = _run("agent1_count.json")  # "1\n2\n3\n4\n5" is *not* JSON -> demonstrates the FAIL path
    return run, [
        AllNodesFinished(),
        OutputJsonValid(),
    ]
