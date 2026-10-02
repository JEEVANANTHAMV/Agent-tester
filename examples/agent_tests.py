"""Example agent-test module for forjinn_eval.

This is the template "anyone can use" — point it at your Forjinn builder host
and chatflow IDs, register as many @agent_test cases as you like, and run via
either pytest or the CLI:

    forjinn-eval run examples/agent_tests.py --report-json report.json
    pytest --forjinn-cases=examples/agent_tests.py

Two sample chatflows (captured from 172.16.34.7):
    * AGENT_NO_TOOLS  - a plain chat agent (process/time estimator, "no cost")
    * AGENT_WITH_MCP  - an MCP tool agent (SQL) that needs BOM + manual uploads

For CI without a live host, set ``FORJINN_OFFLINE=1`` to evaluate recorded
fixtures instead of making network calls.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from forjinn_eval import (
    AllNodesFinished,
    ForjinnClient,
    LatencyBudget,
    NoCostLeakage,
    NoToolsExpected,
    OutputContains,
    OutputMatchesRegex,
    OutputNotEmpty,
    RequiredNodesPresent,
    TokenBudget,
    agent_test,
    make_case,
)

HOST = os.environ.get("FORJINN_HOST", "https://172.16.34.7")
AGENT_NO_TOOLS = os.environ.get("FORJINN_AGENT_NO_TOOLS", "03d5abc5-6ecd-4891-a9a1-364aefb33a50")
AGENT_WITH_MCP = os.environ.get("FORJINN_AGENT_WITH_MCP", "b128d0af-445f-45df-b949-a83dd59ef33e")
OFFLINE = os.environ.get("FORJINN_OFFLINE", "0") == "1"
CLIENT = ForjinnClient(HOST)

_HERE = Path(__file__).resolve().parent
_FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


# Offline (no network) recordings. Each offline case reuses a capture whose
# `question` matches, so content/regex/token assertions stay meaningful.
_COUNT_RECORD = _FIXTURES / "agent1_count.json"
_HELLO_RECORD = _FIXTURES / "agent1_nonstream.json"
_MCP_RECORD = _FIXTURES / "agent2_tools_nonstream.json"


def _offline_predict(chatflow: str, question: str):
    from forjinn_eval import AgentRun

    if chatflow == AGENT_WITH_MCP:
        payload = json.loads(_MCP_RECORD.read_text(encoding="utf-8"))
    elif "count" in question.lower() or question.lower().startswith("count"):
        payload = json.loads(_COUNT_RECORD.read_text(encoding="utf-8"))
    else:
        payload = json.loads(_HELLO_RECORD.read_text(encoding="utf-8"))
    # stamp the question so content/regex evaluators see the intended prompt
    payload["question"] = question
    return AgentRun.from_nonstream(chatflow, payload)


def _predict(chatflow: str, question: str):
    if OFFLINE:
        return _offline_predict(chatflow, question)
    return CLIENT.predict(chatflow, question)


# ---------------------------------------------------------------------------
# Registered cases (run by CLI or pytest --forjinn-cases)
# ---------------------------------------------------------------------------
@agent_test("agent-1-count-1-to-5", tags=["agent-1", "text", "offline-ok"])
def _count():
    run = _predict(AGENT_NO_TOOLS, "Count from 1 to 5, one number per line.")
    return run, [
        AllNodesFinished(),
        RequiredNodesPresent(["start", "agent"]),
        OutputNotEmpty(),
        OutputMatchesRegex(r"(?m)^1\n2\n3\n4\n5$"),
        NoToolsExpected(),
        NoCostLeakage(),
        TokenBudget(output_tokens=25),
        LatencyBudget(max_ms=15000),
    ]


@agent_test("agent-1-no-cost-leakage", tags=["agent-1", "safety", "offline-ok"])
def _no_cost():
    run = _predict(AGENT_NO_TOOLS, "Hey, how are you?")
    return run, [
        AllNodesFinished(),
        OutputNotEmpty(),
        NoCostLeakage(),
        NoToolsExpected(),
        TokenBudget(total_tokens=15000),
    ]


@agent_test("agent-2-mcp-tool-sanity", tags=["agent-2", "tools", "offline-ok"])
def _mcp_sanity():
    run = _predict(AGENT_WITH_MCP, "Hey, how are you?")
    return run, [
        AllNodesFinished(),
        OutputNotEmpty(),
        # In the captured run the MCP agent asked for the BOM/manual uploads,
        # so no tool calls are expected yet - guard against stray calls.
        NoToolsExpected(),
        OutputContains(["BOM"], match="any"),
    ]


if __name__ == "__main__":
    from forjinn_eval import run_registered

    suite = run_registered(min_pass_rate=None)
    print(suite.to_markdown())
