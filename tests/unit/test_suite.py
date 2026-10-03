"""Suite / aggregation / export scenarios."""
from __future__ import annotations

import json

import pytest

from forjinn_eval import (
    AllNodesFinished,
    ForjinnError,
    OutputMatchesRegex,
    OutputNotEmpty,
    SuiteRunner,
    TokenBudget,
    agent_test,
    build_cases,
    make_case,
    registered_cases,
    run_registered,
)
from forjinn_eval.results import Status
from forjinn_eval.suite import _REGISTRY


def _cases(agent1_count, agent1_node_failed):
    return [
        make_case("good", agent1_count,
                  [AllNodesFinished(), OutputMatchesRegex(r"(?m)^5$"), TokenBudget(output_tokens=100)],
                  tags=["agent-1", "text"]),
        make_case("bad", agent1_node_failed,
                  [AllNodesFinished(), OutputNotEmpty()],
                  tags=["agent-1"]),
    ]


class TestRunner:
    def test_run_counts(self, agent1_count, agent1_node_failed):
        sr = SuiteRunner(suite_name="t").run(_cases(agent1_count, agent1_node_failed))
        assert sr.total == 2
        assert sr.passed == 1
        assert sr.failed == 1

    def test_pass_rate(self, agent1_count, agent1_node_failed):
        sr = SuiteRunner(suite_name="t").run(_cases(agent1_count, agent1_node_failed))
        assert sr.pass_rate == 0.5

    def test_case_status_rollup_is_error_beats_fail(self, agent1_count):
        # make one case ERROR (run callable raises) and one fail
        cases = [
            make_case("boom", lambda: (_ for _ in ()).throw(RuntimeError("nope")), [AllNodesFinished()]),
            make_case("fail", agent1_count, [OutputMatchesRegex(r"(?m)^999$")]),
        ]
        sr = SuiteRunner(suite_name="t").run(cases)
        assert sr.results[0].status == Status.ERROR
        assert sr.results[1].status == Status.FAIL
        assert sr.errors == 1 and sr.failed == 1

    def test_evaluator_crash_is_error_not_fail(self, agent1_count):
        from forjinn_eval import Evaluator

        class Boom(Evaluator):
            name = "boom"
            kind = "x"

            def evaluate(self, run):
                raise RuntimeError("boom")

        sr = SuiteRunner(suite_name="t").run([make_case("c", agent1_count, [AllNodesFinished(), Boom()])])
        names = {c.name: c.status.value for c in sr.results[0].check_results}
        assert names["boom"] == Status.ERROR.value
        assert names["all_nodes_finished"] == Status.PASS.value
        assert sr.results[0].status == Status.ERROR  # error beats pass

    def test_usage_rollup(self, agent1_count, agent1_node_failed):
        sr = SuiteRunner(suite_name="t").run(_cases(agent1_count, agent1_node_failed))
        u = sr.usage_rollup()
        assert u["total_tokens"] > 0
        assert u["runs_with_nodes"] == 2

    def test_group_by_tag(self, agent1_count, agent1_node_failed):
        sr = SuiteRunner(suite_name="t").run(_cases(agent1_count, agent1_node_failed))
        # both cases carry the "agent-1" tag -> grouped under it (no double-count)
        g = sr.group_by("agent-1")
        assert g["agent-1"]["total"] == 2
        # a tag no case has -> everything lands in the (untagged) bucket
        g2 = sr.group_by("does-not-exist")
        assert g2["(untagged)"]["total"] == 2


class TestExports:
    def test_to_dict(self, agent1_count, agent1_node_failed):
        sr = SuiteRunner(suite_name="t").run(_cases(agent1_count, agent1_node_failed))
        d = sr.to_dict()
        assert d["total"] == 2
        assert "evaluator_rollups" in d and "usage_rollup" in d
        assert len(d["results"]) == 2

    def test_junit_xml(self, agent1_count, agent1_node_failed):
        sr = SuiteRunner(suite_name="t").run(_cases(agent1_count, agent1_node_failed))
        xml = sr.to_junit_xml()
        assert xml.startswith("<?xml")
        assert 'tests="2"' in xml
        assert "failure" in xml  # the failing case is reported

    def test_markdown(self, agent1_count, agent1_node_failed):
        sr = SuiteRunner(suite_name="t").run(_cases(agent1_count, agent1_node_failed))
        md = sr.to_markdown()
        assert "# Forjinn Eval" in md
        assert "Pass rate" in md
        assert "evaluator" in md.lower()

    def test_save_creates_parent_dirs(self, agent1_count, tmp_path):
        sr = SuiteRunner(suite_name="t").run([make_case("c", agent1_count, [AllNodesFinished()])])
        p_json = tmp_path / "a" / "b" / "r.json"
        p_xml = tmp_path / "a" / "r.xml"
        p_md = tmp_path / "a" / "r.md"
        sr.save_json(p_json)
        sr.save_junit(p_xml)
        sr.save_markdown(p_md)
        assert p_json.exists() and p_xml.exists() and p_md.exists()
        assert json.loads(p_json.read_text("utf-8"))["total"] == 1


class TestRegistration:
    def test_agent_test_register_and_build(self, agent1_count):
        _REGISTRY.clear()

        @agent_test("unit-case", tags=["x"], metadata={"description": "counts"})
        def _case():
            return agent1_count, [AllNodesFinished()]

        assert "unit-case" in registered_cases()
        cases = build_cases(["unit-case"])
        assert len(cases) == 1
        assert cases[0].evaluators
        _REGISTRY.clear()

    def test_build_unknown_raises(self):
        _REGISTRY.clear()
        with pytest.raises(KeyError):
            build_cases(["does-not-exist"])
        _REGISTRY.clear()

    def test_run_registered_gate(self, agent1_count):
        _REGISTRY.clear()

        @agent_test("gate-case")
        def _case():
            return agent1_count, [AllNodesFinished()]

        try:
            sr = run_registered(names=["gate-case"], min_pass_rate=0.0)
            assert sr.passed == 1
            with pytest.raises(ForjinnError):
                run_registered(names=["gate-case"], min_pass_rate=1.1)
        finally:
            _REGISTRY.clear()
