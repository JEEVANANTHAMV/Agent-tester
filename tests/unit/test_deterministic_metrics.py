"""Offline tests for the new deterministic text / tool / loop / composition metrics."""
from __future__ import annotations

from forjinn_eval import (
    AgentLoopDetection,
    AgentRun,
    AllOf,
    AnswerSimilarity,
    AnyOf,
    BleuScore,
    ChrfScore,
    Not,
    OutputMatchesRegex,
    OutputNotEmpty,
    PatternMatch,
    RougeScore,
    SemanticSimilarity,
    ToolCall,
    ToolCallAccuracy,
)
from forjinn_eval.results import Status
from tests.unit._helpers import multi_turn_run, single_turn_run


class TestPatternMatch:
    def test_present(self):
        assert PatternMatch(r"\d+").evaluate(single_turn_run("q", "hello 42")).status == Status.PASS

    def test_absent_fails(self):
        assert PatternMatch(r"\d+").evaluate(single_turn_run("q", "no digits")).status == Status.FAIL

    def test_invert(self):
        assert PatternMatch(r"\d+", invert_match=True).evaluate(
            single_turn_run("q", "no digits")).status == Status.PASS


class TestTextQuality:
    def test_bleu_identical(self):
        ref = "the cat sat on the mat"
        assert BleuScore(reference=ref).evaluate(single_turn_run("q", ref)).score == 1.0

    def test_bleu_different(self):
        assert BleuScore(reference="the cat sat on the mat").evaluate(
            single_turn_run("q", "quantum physics lecture today")).score == 0.0

    def test_rouge_bounds(self):
        assert RougeScore(reference="the cat sat on the mat").evaluate(
            single_turn_run("q", "the cat sat on the mat")).score == 1.0
        s = RougeScore(reference="the cat sat on the mat").evaluate(
            single_turn_run("q", "a completely different sentence here")).score
        assert 0.0 <= s <= 1.0

    def test_chrf_bounds(self):
        assert ChrfScore(reference="hello world").evaluate(
            single_turn_run("q", "hello world")).score == 1.0

    def test_semantic_similarity_identical(self):
        s = SemanticSimilarity(reference="the cat sat on the mat").evaluate(
            single_turn_run("q", "the cat sat on the mat")).score
        assert s > 0.9

    def test_semantic_similarity_alias(self):
        assert AnswerSimilarity is not None
        s = AnswerSimilarity(reference="hello world").evaluate(
            single_turn_run("q", "hello world")).score
        assert 0.0 <= s <= 1.0

    def test_reference_metrics_skip_without_reference(self):
        run = single_turn_run("q", "a")
        for cls in (BleuScore, RougeScore, ChrfScore, SemanticSimilarity):
            assert cls().evaluate(run).status == Status.SKIP


class TestAgentLoopDetection:
    def test_clean_run(self):
        run = AgentRun(chatflow_id="x", question="q", text="answer")
        assert AgentLoopDetection().evaluate(run).score == 1.0

    def test_repeated_tool_calls_detected(self):
        run = AgentRun(chatflow_id="x", question="q", text="answer")
        for _ in range(4):
            run.called_tools.append(ToolCall(name="search", arguments={"q": "x"}))
        res = AgentLoopDetection(repetition_threshold=3).evaluate(run)
        assert res.score < 1.0

    def test_failed_node_penalised(self):
        from forjinn_eval import AgentRun as AR
        payload = {
            "question": "q", "text": "x",
            "agentFlowExecutedData": [
                {"nodeId": "a", "nodeLabel": "agent 0", "status": "FAILED",
                 "data": {"name": "agentAgentflow", "output": {"content": "x"}}},
            ],
        }
        run = AR.from_nonstream("x", payload)
        assert AgentLoopDetection().evaluate(run).score < 1.0


class TestToolCallAccuracy:
    def test_exact_order(self):
        run = multi_turn_run([("", "", [], ["list_tables", "describe_table"])])
        res = ToolCallAccuracy(reference_tool_calls=["list_tables", "describe_table"],
                               threshold=0.5).evaluate(run)
        assert res.status == Status.PASS and res.score == 1.0

    def test_wrong_tools(self):
        run = multi_turn_run([("", "", [], ["delete_databases"])])
        res = ToolCallAccuracy(reference_tool_calls=["list_tables"], threshold=0.5).evaluate(run)
        assert res.status == Status.FAIL

    def test_skip_without_reference(self):
        run = multi_turn_run([("", "", [], ["list_tables"])])
        assert ToolCallAccuracy().evaluate(run).status == Status.SKIP


class TestComposition:
    def test_all_of_pass(self):
        run = single_turn_run("q", "hello 1")
        c = AllOf([OutputNotEmpty(), OutputMatchesRegex(r"\d")])
        assert c.evaluate(run).status == Status.PASS

    def test_all_of_fail_on_one(self):
        run = single_turn_run("q", "no digits")
        c = AllOf([OutputNotEmpty(), OutputMatchesRegex(r"\d")])
        assert c.evaluate(run).status == Status.FAIL

    def test_any_of(self):
        run = single_turn_run("q", "hello")
        c = AnyOf([PatternMatch(r"\d"), PatternMatch(r"hello")])
        assert c.evaluate(run).status == Status.PASS

    def test_not_inverts(self):
        run = single_turn_run("q", "hello")
        assert Not(OutputNotEmpty()).evaluate(run).status == Status.FAIL
        empty = single_turn_run("q", "")
        assert Not(OutputNotEmpty()).evaluate(empty).status == Status.PASS
