"""Negative cases: one fails a check, one crashes (ERROR)."""
import json
from pathlib import Path

from forjinn_eval import AllNodesFinished, RequiredNodesPresent, agent_test


def _run():
    from forjinn_eval import AgentRun

    f = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
    d = json.loads((f / "agent1_nonstream.json").read_text(encoding="utf-8"))
    return AgentRun.from_nonstream("x", d)


@agent_test("case-fails-check", tags=["negative"])
def _fails():
    run = _run()
    return run, [RequiredNodesPresent(["retrieval"])]


@agent_test("case-crashes", tags=["negative"])
def _crashes():
    # returns (AgentRun, evaluators) but the 'run' slot is a *callable* that raises
    def lazy() -> "AgentRun":
        raise RuntimeError("agent exploded")

    return lazy, [AllNodesFinished()]
