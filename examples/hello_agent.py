"""hello_agent.py - the minimal, self-contained forjinn-eval test.

RUN IT (fully offline - recorded fixture, no network):

    forjinn-eval run examples/hello_agent.py
    forjinn-eval list examples/hello_agent.py
    python examples/hello_agent.py

WHAT IT SHOWS
-------------
A single agent case = (name + AgentRun + evaluators + tags), then a suite roll-up.
Metrics are layered cheapest-first:

    structural  ->  AllNodesFinished, RequiredNodesPresent
    content     ->  OutputNotEmpty, OutputContains
    performance ->  TokenBudget

TALKING TO A REAL AGENT
-----------------------
Replace the offline ``_run()`` with a live client call; nothing else changes:

    client = ForjinnClient("https://YOUR_FORJINN_HOST")
    run = client.predict("YOUR_CHATFLOW_ID", "Say hello.")
"""
from __future__ import annotations

import json
from pathlib import Path

from forjinn_eval import (
    AgentRun,
    AllNodesFinished,
    OutputContains,
    OutputNotEmpty,
    RequiredNodesPresent,
    SuiteRunner,
    TokenBudget,
    agent_test,
)

_HERE = Path(__file__).resolve().parent
FIXTURES = _HERE.parent / "tests" / "fixtures"


def _run() -> AgentRun:
    """Build an AgentRun from a recorded fixture (offline).

    For a REAL agent, return ``ForjinnClient(HOST).predict(chatflow, question)``
    instead - the rest of this file is unchanged.
    """
    payload = json.loads((FIXTURES / "agent1_nonstream.json").read_text(encoding="utf-8"))
    return AgentRun.from_nonstream("hello-agent", payload)


@agent_test("hello-works", tags=["hello", "text", "offline"])
def hello():
    """The agent finishes and produces a usable reply."""
    run = _run()
    return run, [
        # structural - did the workflow run as designed?
        AllNodesFinished(),                       # every canvas node FINISHED
        RequiredNodesPresent(["start", "agent"]), # expected node roles
        # content - is the reply sane?
        OutputNotEmpty(),
        # "match=all" requires every listed substring; "match=any" requires at
        # least one. We want to make sure the reply mentions *either* "operating"
        # or "capacity" so we use "any" - a common pattern for loose content checks.
        OutputContains(["operating", "capacity"], match="any"),
        # performance - token budget (cumulative across the run)
        TokenBudget(input_tokens=10000, output_tokens=512),
    ]


if __name__ == "__main__":
    from forjinn_eval import run_registered

    suite = run_registered(suite_name="hello-agent")
    print(suite.to_markdown())
