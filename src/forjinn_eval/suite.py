"""Test-case definition + runner + cumulative aggregation.

A *test case* bundles: a question (or call), an ``AgentRun`` (the captured
execution), the evaluators to apply, and the resulting :class:`CheckResult`s.
A *suite runner* executes many cases (optionally in parallel) and produces a
:class:`SuiteResult` - the "cumulative set of results" the library is meant to
surface: total pass/fail counts, per-evaluator rollup, token + latency sum/avg
across runs, and optional grouping by tag.

Design notes (from deepeval / ragas / giskard / agenteval):
* metric = the atom (each evaluator returns a CheckResult per run)
* 4-state ladder PASS/FAIL/ERROR/SKIP (giskard)
* aggregate = per-evaluator rollup over the whole suite (ragas mean; here we do
  both an ALL-PASS mode and an AVG score mode)
* pass-rate gate for CI (deepeval/pytest + agenteval exit-code)
"""
from __future__ import annotations

import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .capture import AgentRun
from .evaluators import Evaluator
from .results import CheckResult, Status, rollup


def _llm_judge_from_env():
    """Build the LLM-judge overlay from environment, or return ``None``.

    * ``FORJINN_LLM_JUDGE=1`` (or ``true``), or a configured
      ``FORJINN_JUDGE_CHATFLOW``  -> attach the standard battery.
    * ``FORJINN_OFFLINE=1`` -> use a permissive :class:`MockJudge` so the whole
      LLM catalog runs deterministically with **no network** (a CI demo).
    * otherwise -> a real :class:`JudgeClient` to ``FORJINN_JUDGE_CHATFLOW``
      (a Forjinn chatflow built as an LLM judge) on ``FORJINN_HOST``.
    """
    enabled = os.environ.get("FORJINN_LLM_JUDGE", "").strip().lower() in {"1", "true", "yes", "on"}
    has_judge = bool(os.environ.get("FORJINN_JUDGE_CHATFLOW"))
    if not (enabled or has_judge):
        return None
    from .judge import JudgeClient, queuing_judge

    if os.environ.get("FORJINN_OFFLINE", "").strip().lower() in {"1", "true", "yes", "on"}:
        # Permissive offline stand-in: the canonical queuing judge answers each
        # per-request rubric with a favourable, shape-correct value so the whole
        # LLM catalog runs with no network.
        return queuing_judge()
    return JudgeClient()


# ---------------------------------------------------------------------------
# Test-case + case result
# ---------------------------------------------------------------------------
@dataclass
class AgentTestCase:
    """One test case: name, how to obtain the run, and which evaluators to run.

    ``run_callable`` is the unit-test analog of running your agent.
    Accepts either:

    * an existing ``AgentRun`` (offline, e.g. captured from a JSON file), or
    * a zero-arg callable ``() -> AgentRun`` (online, e.g. a lambda that
      ``client.predict(...)``s).

    Multiple evaluators run per case. ``tags`` allow grouped rollup
    (e.g. ``agent-1-no-tools``, ``file-upload``, ``mcp-sql``).
    """

    name: str
    run: Any  # AgentRun | Callable[[], AgentRun]
    evaluators: Sequence[Evaluator] = field(default_factory=list)
    tags: Sequence[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def resolve_run(self) -> AgentRun:
        if isinstance(self.run, AgentRun):
            return self.run
        if callable(self.run):
            return self.run()
        raise TypeError(f"test.case.run must be AgentRun or callable, got {type(self.run)!r}")


@dataclass
class TestCaseResult:
    """The outcome of one :class:`AgentTestCase`: run + check results + rollup."""

    name: str
    run: Optional[AgentRun]
    check_results: List[CheckResult]
    tags: Sequence[str]
    duration_ms: float
    error: Optional[str] = None  # capture-level (not evaluator) failure

    @property
    def status(self) -> Status:
        if self.error:
            return Status.ERROR
        return rollup([c.status for c in self.check_results])

    @property
    def passed(self) -> bool:
        return self.status == Status.PASS

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "check_results": [c.to_dict() for c in self.check_results],
            "tags": list(self.tags),
            "usage": self.run.usage.to_dict() if self.run else None,
            "text_preview": (self.run.text[:240] if self.run else None),
        }


# ---------------------------------------------------------------------------
# Suite + aggregate
# ---------------------------------------------------------------------------
@dataclass
class EvaluatorRollup:
    """Aggregated verdict for one evaluator across all cases that used it."""

    name: str
    kind: str = "eval"
    used_in: int = 0  # number of cases that applied it (excluding SKIP)
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    scores: List[float] = field(default_factory=list)

    @property
    def pass_rate(self) -> Optional[float]:
        n = self.passed + self.failed + self.errors
        return (self.passed / n) if n else None

    @property
    def avg_score(self) -> Optional[float]:
        return (sum(self.scores) / len(self.scores)) if self.scores else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "used_in": self.used_in,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "skipped": self.skipped,
            "pass_rate": self.pass_rate,
            "avg_score": self.avg_score,
            "scores": self.scores,
        }


@dataclass
class SuiteResult:
    """Cumulative set of results from a suite of test cases."""

    results: List[TestCaseResult] = field(default_factory=list)
    started_at: float = 0.0
    duration_ms: float = 0.0
    suite_name: str = ""

    def add(self, r: TestCaseResult) -> None:
        self.results.append(r)

    # ---- rollups ----------------------------------------------------
    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == Status.PASS)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == Status.FAIL)

    @property
    def errors(self) -> int:
        return sum(1 for r in self.results if r.status == Status.ERROR)

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.status == Status.SKIP)

    @property
    def pass_rate(self) -> Optional[float]:
        n = self.passed + self.failed + self.errors
        return (self.passed / n) if n else None

    # ---- evaluator-level cumulative ---------------------------------
    def evaluator_rollups(self) -> Dict[str, EvaluatorRollup]:
        out: Dict[str, EvaluatorRollup] = {}
        for r in self.results:
            for c in r.check_results:
                roll = out.setdefault(c.name, EvaluatorRollup(name=c.name, kind="eval"))
                if c.status == Status.SKIP:
                    roll.skipped += 1
                else:
                    roll.used_in += 1
                    if c.status == Status.PASS:
                        roll.passed += 1
                    elif c.status == Status.FAIL:
                        roll.failed += 1
                    elif c.status == Status.ERROR:
                        roll.errors += 1
                    if c.score is not None:
                        roll.scores.append(c.score)
        return out

    # ---- cumulative numeric aggregates across RUNS ------------------
    def usage_rollup(self) -> Dict[str, float]:
        total_in = total_out = total_all = 0
        deltas: List[float] = []
        for r in self.results:
            if not r.run:
                continue
            total_in += r.run.usage.input_tokens
            total_out += r.run.usage.output_tokens
            total_all += r.run.usage.total_tokens
            node = r.run.final_agent_node
            if node and node.time.delta_ms is not None:
                deltas.append(node.time.delta_ms)
        return {
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_tokens": total_all,
            "runs_with_nodes": self.total,
            "avg_latency_ms": (sum(deltas) / len(deltas)) if deltas else 0.0,
            "max_latency_ms": (max(deltas)) if deltas else 0.0,
            "min_latency_ms": (min(deltas)) if deltas else 0.0,
        }

    def group_by(self, tag_key: str) -> Dict[str, Dict[str, Any]]:
        """Group cases by a tag value (e.g. ``agent-1``) and compute a mini rollup.

        Every case that carries ``tag_key`` in its ``tags`` is grouped under it;
        cases without it are bucketed under ``"(untagged)"`` so nothing is lost.
        """
        groups: Dict[str, List[TestCaseResult]] = {}
        for r in self.results:
            if tag_key in r.tags:
                bucket = tag_key
            else:
                bucket = "(untagged)"
            groups.setdefault(bucket, []).append(r)
        return {
            k: {
                "total": len(v),
                "passed": sum(1 for x in v if x.status == Status.PASS),
                "failed": sum(1 for x in v if x.status == Status.FAIL),
                "errors": sum(1 for x in v if x.status == Status.ERROR),
            }
            for k, v in groups.items()
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suite": self.suite_name,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "skipped": self.skipped,
            "pass_rate": self.pass_rate,
            "duration_ms": self.duration_ms,
            "evaluator_rollups": {
                k: v.to_dict() for k, v in sorted(self.evaluator_rollups().items())
            },
            "usage_rollup": self.usage_rollup(),
            "results": [r.to_dict() for r in self.results],
        }

    # ---- exports ----------------------------------------------------
    def to_junit_xml(self) -> str:
        from xml.sax.saxutils import quoteattr

        out = ['<?xml version="1.0" encoding="utf-8"?>', "<testsuites>"]
        suite = self.suite_name or "forjinn-eval"
        fail_n = self.failed + self.errors
        out.append(
            f'<testsuite name={quoteattr(suite)} tests="{self.total}" '
            f'failures="{fail_n}" errors="0" time="{self.duration_ms / 1000:.3f}">'
        )
        for r in self.results:
            tc_attrs = (
                f'<testcase name={quoteattr(r.name)} '
                f'time="{(r.duration_ms or 0) / 1000:.3f}">'
            )
            if r.status in (Status.FAIL, Status.ERROR):
                detail = r.error or "; ".join(
                    f"{c.name}: {c.reason}" for c in r.check_results if c.status in (Status.FAIL, Status.ERROR)
                )
                tc_attrs += f"<failure message={quoteattr(detail[:400])}></failure>"
            elif r.status == Status.SKIP:
                tc_attrs += "<skipped/>"
            tc_attrs += "</testcase>"
            out.append(tc_attrs)
        out.append("</testsuite>")
        out.append("</testsuites>")
        return "\n".join(out)

    def to_markdown(self) -> str:
        pr = (self.pass_rate or 0) * 100
        lines = [
            f"# Forjinn Eval - {self.suite_name or 'suite'}",
            "",
            f"**Total:** {self.total} | **Passed:** {self.passed} | **Failed:** {self.failed} "
            f"| **Errors:** {self.errors} | **Skipped:** {self.skipped}",
            f"**Pass rate:** {pr:.1f}%",
            "",
            "## Evaluator rollups",
            "",
            "| evaluator | kind | used | pass | fail | err | pass% | avg score |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for k, v in self.evaluator_rollups().items():
            prate = f"{v.pass_rate * 100:.0f}%" if v.pass_rate is not None else "-"
            avg = f"{v.avg_score:.2f}" if v.avg_score is not None else "-"
            lines.append(f"| {k} | {v.kind} | {v.used_in} | {v.passed} | {v.failed} | {v.errors} | {prate} | {avg} |")

        u = self.usage_rollup()
        lines += [
            "",
            "## Usage",
            "",
            f"- total input tokens: {u['total_input_tokens']}",
            f"- total output tokens: {u['total_output_tokens']}",
            f"- total tokens: {u['total_tokens']}",
            f"- avg latency: {u['avg_latency_ms']:.0f}ms (max {u['max_latency_ms']:.0f}ms)",
            "",
        ]
        return "\n".join(lines)

    # ---- convenience savers -----------------------------------------
    def save_json(self, path) -> None:
        import json as _json

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def save_junit(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_junit_xml(), encoding="utf-8")

    def save_markdown(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_markdown(), encoding="utf-8")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
class SuiteRunner:
    """Runs a list of :class:`AgentTestCase` and aggregates into :class:`SuiteResult`."""

    def __init__(self, parallel: int = 1, suite_name: str = "", llm_judge=None):
        self.parallel = max(1, parallel)
        self.suite_name = suite_name
        # Optional LLM-judge overlay (offline -> MockJudge, online -> JudgeClient).
        self.llm_judge = llm_judge if llm_judge is not None else _llm_judge_from_env()

    def run(self, cases: Sequence[AgentTestCase]) -> SuiteResult:
        t0 = time.time()
        sr = SuiteResult(suite_name=self.suite_name)
        for case in cases:
            sr.add(self._run_one(case))
        sr.duration_ms = (time.time() - t0) * 1000.0
        return sr

    def _extend(self, case: AgentTestCase) -> AgentTestCase:
        """Return ``case`` with the LLM-judge battery appended (if enabled)."""
        if self.llm_judge is None:
            return case
        from .llm_metrics import GEval, default_judge_metrics

        base = list(case.evaluators)
        overlay = default_judge_metrics(judge=self.llm_judge)
        if case.metadata.get("description"):
            overlay.append(GEval(judge=self.llm_judge, criteria=case.metadata["description"]))
        elif case.tags:
            overlay.append(GEval(judge=self.llm_judge, criteria=f"Conform to: {', '.join(case.tags)}"))
        return AgentTestCase(
            name=case.name,
            run=case.run,
            evaluators=base + overlay,
            tags=list(case.tags) + ["llm-judge"],
            metadata=dict(case.metadata),
        )

    def _run_one(self, case: AgentTestCase) -> TestCaseResult:
        t0 = time.time()
        case = self._extend(case)
        try:
            run = case.resolve_run()
        except Exception as e:  # capture failure = ERROR, not FAIL
            return TestCaseResult(
                name=case.name,
                run=None,
                check_results=[],
                tags=case.tags,
                duration_ms=(time.time() - t0) * 1000.0,
                error=f"resolve_run failed: {e!r}",
            )
        check_results: List[CheckResult] = []
        for ev in case.evaluators:
            try:
                check_results.append(ev.evaluate(run))
            except Exception as e:  # evaluator crash = ERROR on that check
                check_results.append(
                    CheckResult.error_(ev.name, f"evaluator raised: {e!r}")
                )
        return TestCaseResult(
            name=case.name,
            run=run,
            check_results=check_results,
            tags=case.tags,
            duration_ms=(time.time() - t0) * 1000.0,
        )


# ---------------------------------------------------------------------------
# Declarative decorators (the "unit-test" feel)
# ---------------------------------------------------------------------------
_REGISTRY: Dict[str, AgentTestCase] = {}


def agent_test(name, tags=(), metadata=None):
    """Register an :class:`AgentTestCase`.

    Use as a decorator::

        @agent_test("agent-1-no-tools", tags=["agent-1", "text"])
        def case():
            run = client.predict(chatflow, "Count 1..5")
            return run, [AllNodesFinished(), OutputNotEmpty()]

    The decorated function must return either an ``AgentRun`` or a
    ``(AgentRun, [evaluators])`` tuple.
    """
    tags_list = list(tags or ())
    meta: Dict[str, Any] = dict(metadata or {})

    def deco(fn: Callable[..., Any]):
        if "description" not in meta and fn.__doc__:
            meta["description"] = fn.__doc__.strip()
        _REGISTRY[name] = _CaseFactory(
            name=name,
            factory=fn,
            tags=list(tags_list),
            metadata=dict(meta),
        )
        return fn

    return deco


@dataclass
class _CaseFactory:
    name: str
    factory: Callable[[], Any]
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


def registered_cases() -> List[str]:
    return list(_REGISTRY.keys())


def build_cases(names: Optional[Sequence[str]] = None) -> List[AgentTestCase]:
    """Materialise registered cases into :class:`AgentTestCase` objects.

    The factory's return convention is either an ``AgentRun`` or a
    ``(AgentRun, [evaluators])`` tuple; evaluators are resolved here.
    """
    wanted = names or list(_REGISTRY.keys())
    cases: List[AgentTestCase] = []
    for nm in wanted:
        if nm not in _REGISTRY:
            raise KeyError(f"unregistered agent_test case: {nm}")
        f = _REGISTRY[nm]
        res = f.factory()
        if isinstance(res, tuple) and len(res) == 2:
            run, evaluators = res
        elif isinstance(res, AgentRun):
            run, evaluators = res, []
        else:
            run = res
            evaluators = []
        cases.append(
            AgentTestCase(
                name=f.name,
                run=run,
                evaluators=list(evaluators),
                tags=list(f.tags),
                metadata=dict(f.metadata),
            )
        )
    return cases


# Re-export for convenience
def make_case(
    name,
    run,
    evaluators=None,
    tags=None,
    metadata=None,
):
    """Programmatic constructor for a single test case.

    ``run`` may be an ``AgentRun`` (offline / pre-captured) or a zero-arg
    ``Callable[[], AgentRun]`` (online - called lazily by the runner). A bare
    callable is *not* invoked at construction time, so a raising agent is
    reported as an ERROR result rather than crashing the suite build.
    """
    return AgentTestCase(
        name=name,
        run=run,
        evaluators=list(evaluators or []),
        tags=list(tags or []),
        metadata=dict(metadata or {}),
    )
