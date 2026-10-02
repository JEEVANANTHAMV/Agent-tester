"""Structural evaluators: node presence / status / model / start-node sanity."""
from __future__ import annotations

from typing import Iterable, List

from ..capture import AgentRun
from ..types import STATUS_FINISHED
from .base import CheckResult, Evaluator

__all__ = [
    "AllNodesFinished",
    "RequiredNodesPresent",
    "ExpectedNodeCount",
    "ModelIs",
    "StartNodePassthrough",
]


class AllNodesFinished(Evaluator):
    """Every canvas node must have reached ``FINISHED`` (none FAILED/INPROGRESS)."""

    name = "all_nodes_finished"
    kind = "structural"

    def evaluate(self, run: AgentRun) -> CheckResult:
        if not run.nodes:
            return CheckResult.error_(self.name, "no nodes captured")
        bad = [(nd.node_id, nd.status) for nd in run.nodes if nd.status != STATUS_FINISHED]
        if bad:
            return CheckResult.fail_(
                self.name,
                f"{len(bad)} node(s) not FINISHED: {bad}"[:400],
                details={"bad_nodes": bad},
            )
        return CheckResult.pass_(self.name, f"all {len(run.nodes)} nodes FINISHED")


class RequiredNodesPresent(Evaluator):
    """The run's node set contains all expected node name-prefixes."""

    name = "required_nodes_present"
    kind = "structural"

    def __init__(self, required: Iterable[str]):
        self.required = list(required)

    def evaluate(self, run: AgentRun) -> CheckResult:
        present = set(run.node_names)
        missing = [r for r in self.required if r not in {n.split("Agentflow")[0] for n in present}]
        if missing:
            return CheckResult.fail_(
                self.name, f"missing node(s): {missing}", details={"required": self.required, "present": sorted(present)}
            )
        return CheckResult.pass_(self.name, f"all required nodes present: {self.required}")


class ExpectedNodeCount(Evaluator):
    """Run contains exactly N nodes (useful for single-agent workflows)."""

    name = "expected_node_count"
    kind = "structural"

    def __init__(self, count: int, at_least: bool = False):
        self.count = count
        self.at_least = at_least

    def evaluate(self, run: AgentRun) -> CheckResult:
        n = len(run.nodes)
        ok = n >= self.count if self.at_least else n == self.count
        if ok:
            return CheckResult.pass_(self.name, f"node_count={n}", score=1.0)
        return CheckResult.fail_(self.name, f"node_count={n}, expected {'>=' if self.at_least else '=='} {self.count}")


class ModelIs(Evaluator):
    """Sanity gate: the captured agent node must be using the expected model name."""

    name = "model_is"
    kind = "structural"

    def __init__(self, model_name: str):
        self.model_name = model_name

    def evaluate(self, run: AgentRun) -> CheckResult:
        node = run.final_agent_node
        if not node:
            return CheckResult.error_(self.name, "no agent node found")
        actual = node.model_name
        if actual and actual.lower() == self.model_name.lower():
            return CheckResult.pass_(self.name, f"model={actual}")
        return CheckResult.fail_(self.name, f"model={actual!r}, expected {self.model_name!r}")


class StartNodePassthrough(Evaluator):
    """The start node must echo ``question`` unchanged (flow-wiring smoke test)."""

    name = "start_node_passthrough"
    kind = "structural"

    def evaluate(self, run: AgentRun) -> CheckResult:
        starts = run.nodes_by_type("start")
        if not starts:
            return CheckResult.skip_(self.name, "no start node present (maybe not a start-agent flow)")
        q = run.question
        for nd in starts:
            if nd.input.get("question") == q and nd.output.get("question") == q:
                return CheckResult.pass_(self.name, "start node passes question through unchanged")
        return CheckResult.fail_(
            self.name,
            f"start node did not echo question; input={starts[0].input.get('question')!r}",
            details={"question": q, "start_input": starts[0].input.get("question")},
        )
