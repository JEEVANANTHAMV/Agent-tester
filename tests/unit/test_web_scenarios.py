"""Offline tests for the 'open forjinn web to all scenarios' runner."""
from __future__ import annotations

import pytest

from forjinn_eval import web_scenarios as W
from forjinn_eval import run_all_scenarios
from forjinn_eval.web_scenarios import (
    ForjinnWebRunner,
    Persona,
    PERSONAS,
    Scenario,
)
from forjinn_eval.results import Status


class TestScenarioCatalog:
    def test_all_scenarios(self):
        scs = W.all_scenarios()
        assert len(scs) >= 4
        assert any(s.id == "count-1-to-5" for s in scs)

    def test_persona_registry(self):
        assert "helpful" in PERSONAS
        assert isinstance(PERSONAS["confused"], Persona)


class TestBatteriesCompose:
    def test_structural_batteries(self):
        run = W._fixture_run(W.Scenario("count-1-to-5", "q", "question"))
        evs = W.structural_batteries(run, W.Scenario("count-1-to-5", "q", "question"))
        assert any(type(e).__name__ == "OutputNotEmpty" for e in evs)
        assert any(type(e).__name__ == "AgentLoopDetection" for e in evs)

    def test_deterministic_batteries_with_reference(self):
        run = W._fixture_run(W.Scenario("math-two-plus-two", "What is 2+2?",
                                        "question", reference="4"))
        evs = W.deterministic_batteries(run, W.Scenario("math-two-plus-two",
                                                        "What is 2+2?", "question",
                                                        reference="4"))
        names = {type(e).__name__ for e in evs}
        assert "BleuScore" in names and "RougeScore" in names and "SemanticSimilarity" in names

    def test_conversational_batteries_only_for_multi(self):
        from forjinn_eval import AgentRun
        single = AgentRun(chatflow_id="x", question="q", text="a")
        assert W.conversational_batteries(None, None, single,
                                          Scenario("s", "q", "question")) == []


class TestRunnerOffline:
    def test_run_all_scenarios_offline_greens(self):
        suite = run_all_scenarios(offline=True)
        assert suite.total >= 4
        assert suite.failed == 0 and suite.errors == 0
        assert (suite.pass_rate or 0) == 1.0

    def test_include_persona(self):
        suite = run_all_scenarios(offline=True, include_persona=True)
        assert any("persona" in r.name for r in suite.results)

    def test_full_metric_coverage(self):
        suite = run_all_scenarios(offline=True)
        rollups = suite.evaluator_rollups()
        # a representative slice across every family must be present
        for kind in ("faithfulness", "answer_relevancy", "bleu_score",
                     "semantic_similarity", "bias", "toxicity", "plan_adherence",
                     "agent_loop_detection", "turn_faithfulness"):
            assert kind in rollups, f"expected metric {kind} in rollups"

    def test_reports_serialise(self):
        suite = run_all_scenarios(offline=True)
        xml = suite.to_junit_xml()
        md = suite.to_markdown()
        assert xml.startswith("<?xml")
        assert "# Forjinn Eval" in md


class TestCliWebScenarios:
    def test_cli_runs(self, capsys):
        from forjinn_eval.cli import main
        rc = main(["web-scenarios", "--offline"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "Forjinn Eval" in out

    def test_cli_persona_gate(self, capsys, tmp_path):
        from forjinn_eval.cli import main
        j = tmp_path / "w.json"
        md = tmp_path / "w.md"
        rc = main(["web-scenarios", "--offline", "--include-persona",
                   "--report-json", str(j), "--report-md", str(md),
                   "--min-pass-rate", "0.9"])
        assert rc == 0
        assert j.is_file() and md.is_file()
