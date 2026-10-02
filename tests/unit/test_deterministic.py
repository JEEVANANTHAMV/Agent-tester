"""Deterministic evaluator scenarios, by kind.

Each evaluator is exercised across its verdict ladder (PASS / FAIL / ERROR /
SKIP) using the recorded fixtures plus small synthetic runs.
"""
from __future__ import annotations

from forjinn_eval import (
    AgentRun,
    AllNodesFinished,
    AvailableToolsExposed,
    NoToolsExpected,
    OnlyAllowedTools,
    RequiredNodesPresent,
    ExpectedNodeCount,
    ModelIs,
    StartNodePassthrough,
    TokenBudget,
    LatencyBudget,
    ToolCallOrder,
    ToolCallSetF1,
    ToolCallCount,
    OutputNotEmpty,
    OutputMatchesRegex,
    OutputContains,
    OutputDoesNotContain,
    OutputLengthBounds,
    OutputJsonValid,
    NoCostLeakage,
    AttachmentParsed,
    AttachmentUploaded,
    ToolCall,
)
from forjinn_eval.results import Status


def _S(c: object) -> str:
    return c.status.value


# ---------------------------------------------------------------------------
# Structural
# ---------------------------------------------------------------------------
class TestStructural:
    def test_all_nodes_finished_pass(self, agent1_count):
        assert _S(AllNodesFinished().evaluate(agent1_count)) == Status.PASS.value

    def test_all_nodes_finished_fail(self, agent1_node_failed):
        assert _S(AllNodesFinished().evaluate(agent1_node_failed)) == Status.FAIL.value

    def test_all_nodes_finished_error_when_no_nodes(self):
        run = AgentRun(chatflow_id="x", question="q")
        assert _S(AllNodesFinished().evaluate(run)) == Status.ERROR.value

    def test_required_nodes_present(self, agent1_count):
        assert _S(RequiredNodesPresent(["start", "agent"]).evaluate(agent1_count)) == Status.PASS.value
        assert _S(RequiredNodesPresent(["retrieval"]).evaluate(agent1_count)) == Status.FAIL.value

    def test_expected_node_count(self, agent1_count):
        n = len(agent1_count.nodes)
        assert _S(ExpectedNodeCount(n).evaluate(agent1_count)) == Status.PASS.value
        assert _S(ExpectedNodeCount(n + 1).evaluate(agent1_count)) == Status.FAIL.value
        assert _S(ExpectedNodeCount(1, at_least=True).evaluate(agent1_count)) == Status.PASS.value

    def test_start_node_passthrough(self, agent1_count):
        assert _S(StartNodePassthrough().evaluate(agent1_count)) in {Status.PASS.value, Status.SKIP.value}

    def test_model_is(self, agent1_count):
        model = agent1_count.final_agent_node.model_name
        assert _S(ModelIs(model).evaluate(agent1_count)) == Status.PASS.value
        assert _S(ModelIs("a-different-model-name").evaluate(agent1_count)) == Status.FAIL.value


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------
class TestPerformance:
    def test_token_budget_pass_and_fail(self, agent1_count):
        u = agent1_count.usage
        assert _S(TokenBudget(total_tokens=u.total_tokens + 1000).evaluate(agent1_count)) == Status.PASS.value
        assert _S(TokenBudget(total_tokens=1).evaluate(agent1_count)) == Status.FAIL.value

    def test_token_budget_all_limits_none(self, agent1_count):
        assert _S(TokenBudget().evaluate(agent1_count)) == Status.PASS.value

    def test_latency_budget_pass_and_fail(self, agent1_count):
        delta = agent1_count.final_agent_node.time.delta_ms or 0
        assert _S(LatencyBudget(max_ms=delta + 1000).evaluate(agent1_count)) == Status.PASS.value
        assert _S(LatencyBudget(max_ms=max(1, delta - 1)).evaluate(agent1_count)) == Status.FAIL.value


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------
class TestTool:
    def test_no_tools_expected(self, agent1_count, agent1_node_failed):
        assert _S(NoToolsExpected().evaluate(agent1_count)) == Status.PASS.value
        # give it a tool call to force a FAIL
        agent1_node_failed.called_tools = [ToolCall.from_dict({"name": "sql", "arguments": {}})]
        assert _S(NoToolsExpected().evaluate(agent1_node_failed)) == Status.FAIL.value

    def test_tool_call_order(self, agent1_node_failed):
        run = agent1_node_failed
        run.called_tools = [ToolCall.from_dict({"name": "a", "arguments": {"x": 1}}),
                            ToolCall.from_dict({"name": "b", "arguments": {}})]
        assert _S(ToolCallOrder(["a", "b"]).evaluate(run)) == Status.PASS.value
        assert _S(ToolCallOrder(["b", "a"]).evaluate(run)) == Status.FAIL.value

    def test_tool_call_order_with_args(self, agent1_node_failed):
        run = agent1_node_failed
        run.called_tools = [ToolCall.from_dict({"name": "calc", "arguments": {"expr": "1+2"}})]
        assert _S(ToolCallOrder([("calc", {"expr": "1+2"})]).evaluate(run)) == Status.PASS.value
        assert _S(ToolCallOrder([("calc", {"expr": "9+9"})]).evaluate(run)) == Status.FAIL.value

    def test_tool_call_set_f1(self, agent1_node_failed):
        run = agent1_node_failed
        run.called_tools = [ToolCall.from_dict({"name": "a", "arguments": {}}),
                            ToolCall.from_dict({"name": "b", "arguments": {}})]
        assert _S(ToolCallSetF1(["a", "b"]).evaluate(run)) == Status.PASS.value
        assert ToolCallSetF1(["a", "c"]).evaluate(run).score == 0.5  # 1 of 2 expected

    def test_only_allowed_tools(self, agent1_node_failed):
        run = agent1_node_failed
        run.called_tools = [ToolCall.from_dict({"name": "allowed_t", "arguments": {}})]
        assert _S(OnlyAllowedTools(["allowed_t"]).evaluate(run)) == Status.PASS.value
        run.called_tools = [ToolCall.from_dict({"name": "evil_t", "arguments": {}})]
        assert _S(OnlyAllowedTools(["allowed_t"]).evaluate(run)) == Status.FAIL.value

    def test_tool_call_count(self, agent1_node_failed):
        run = agent1_node_failed
        run.called_tools = [ToolCall.from_dict({"name": "t", "arguments": {}}),
                            ToolCall.from_dict({"name": "t", "arguments": {}})]
        assert _S(ToolCallCount(min_calls=1, max_calls=2).evaluate(run)) == Status.PASS.value
        assert _S(ToolCallCount(min_calls=3).evaluate(run)) == Status.FAIL.value


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------
class TestContent:
    def test_output_not_empty(self, agent1_count):
        assert _S(OutputNotEmpty().evaluate(agent1_count)) == Status.PASS.value
        assert _S(OutputNotEmpty().evaluate(AgentRun(chatflow_id="x", text="   "))) == Status.FAIL.value

    def test_output_matches_regex(self, agent1_count):
        assert _S(OutputMatchesRegex(r"(?m)^1\n2\n3\n4\n5$").evaluate(agent1_count)) == Status.PASS.value
        assert _S(OutputMatchesRegex(r"(?m)^42$").evaluate(agent1_count)) == Status.FAIL.value

    def test_output_matches_regex_fullmatch(self, agent1_count):
        assert _S(OutputMatchesRegex(r"1\n2\n3\n4\n5", fullmatch=True).evaluate(agent1_count)) == Status.PASS.value

    def test_output_matches_regex_error_on_bad_pattern(self, agent1_count):
        assert _S(OutputMatchesRegex("[unclosed").evaluate(agent1_count)) == Status.ERROR.value

    def test_output_contains(self, agent1_count):
        assert _S(OutputContains(["1", "5"]).evaluate(agent1_count)) == Status.PASS.value
        assert _S(OutputContains(["1", "9999"], match="all").evaluate(agent1_count)) == Status.FAIL.value
        assert _S(OutputContains(["8", "9999"], match="any").evaluate(agent1_count)) == Status.FAIL.value

    def test_output_does_not_contain(self, agent1_count):
        assert _S(OutputDoesNotContain(["ZZZZ"]).evaluate(agent1_count)) == Status.PASS.value
        assert _S(OutputDoesNotContain(["1"]).evaluate(agent1_count)) == Status.FAIL.value

    def test_output_length_bounds(self, agent1_count):
        n = len(agent1_count.text)
        assert _S(OutputLengthBounds(min_chars=0, max_chars=n + 10).evaluate(agent1_count)) == Status.PASS.value
        assert _S(OutputLengthBounds(min_chars=n + 100).evaluate(agent1_count)) == Status.FAIL.value

    def test_output_json_valid(self):
        good = AgentRun(chatflow_id="x", text='{"a": 1}')
        bad = AgentRun(chatflow_id="x", text="{not json")
        assert _S(OutputJsonValid().evaluate(good)) == Status.PASS.value
        assert _S(OutputJsonValid().evaluate(bad)) == Status.FAIL.value
        assert _S(OutputJsonValid().evaluate(AgentRun(chatflow_id="x", text=""))) == Status.ERROR.value


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------
class TestSafety:
    def test_no_cost_leakage_pass(self, agent1_count):
        # the count answer has no currency figures
        assert _S(NoCostLeakage().evaluate(agent1_count)) == Status.PASS.value

    def test_no_cost_leakage_fail(self):
        run = AgentRun(chatflow_id="x", text="The cost is $50 and INR 20.")
        res = NoCostLeakage().evaluate(run)
        assert _S(res) == Status.FAIL.value
        assert res.details.get("match")

    def test_no_cost_leakage_skip_on_empty(self):
        assert _S(NoCostLeakage().evaluate(AgentRun(chatflow_id="x", text=""))) == Status.SKIP.value


# ---------------------------------------------------------------------------
# Attachment
# ---------------------------------------------------------------------------
class TestAttachment:
    def test_attachment_parsed(self):
        run = AgentRun(chatflow_id="x", text="", raw={"parsed_attachment": ["sheet A", "sheet B"]})
        assert _S(AttachmentParsed(contains=["sheet A"], min_items=2).evaluate(run)) == Status.PASS.value
        empty = AgentRun(chatflow_id="x", text="", raw={})
        assert _S(AttachmentParsed().evaluate(empty)) == Status.FAIL.value

    def test_attachment_uploaded(self):
        run = AgentRun(chatflow_id="x", text="", raw={"uploaded_files": ["BOM.xlsx", "manual.pdf"]})
        assert _S(AttachmentUploaded(["BOM.xlsx"]).evaluate(run)) == Status.PASS.value
        assert _S(AttachmentUploaded(["BOM.xlsx", "missing.pdf"]).evaluate(run)) == Status.FAIL.value
        assert _S(AttachmentUploaded().evaluate(AgentRun(chatflow_id="x", text="", raw={}))) == Status.FAIL.value
