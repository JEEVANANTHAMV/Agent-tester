"""Capture/data-model scenarios: non-stream + streaming -> one AgentRun."""
from __future__ import annotations

from forjinn_eval import AgentRun
from forjinn_eval.capture import parse_sse_text
from forjinn_eval.types import STATUS_FINISHED


class TestFromNonstream:
    def test_top_level_fields(self, agent1_nonstream):
        run = agent1_nonstream
        assert run.question
        assert run.chat_id
        assert run.session_id
        assert run.text and run.text.strip()

    def test_nodes_parsed(self, agent1_nonstream):
        run = agent1_nonstream
        assert run.node_names == ["startAgentflow", "agentAgentflow"]
        assert run.all_nodes_finished is True
        assert run.usage.total_tokens > 0
        assert run.usage.input_tokens > 0

    def test_agent_node_hoisted(self, agent1_nonstream):
        node = agent1_nonstream.final_agent_node
        assert node is not None
        assert node.model_name
        assert node.messages
        assert node.messages[0]["role"] == "system"

    def test_node_type_and_ids(self, agent1_nonstream):
        by_id = {n.node_id: n for n in agent1_nonstream.nodes}
        assert by_id["startAgentflow_0"].is_start
        assert by_id["agentAgentflow_0"].is_agent
        assert by_id["agentAgentflow_0"].previous_node_ids == ["startAgentflow_0"]

    def test_tool_calls_empty_for_no_tools(self, agent1_nonstream):
        assert agent1_nonstream.all_tool_calls == []

    def test_round_trip_dict(self, agent1_nonstream):
        d = agent1_nonstream.to_dict()
        assert d["chatflow_id"] == "agent-1"
        assert d["usage"]["total_tokens"] > 0
        assert len(d["nodes"]) == 2

    def test_usage_rollup(self, agent1_nonstream):
        assert agent1_nonstream.usage.input_tokens == agent1_nonstream.final_agent_node.usage.input_tokens

    def test_node_failed_status(self, agent1_node_failed):
        run = agent1_node_failed
        assert not run.all_nodes_finished
        bad = [nd.status for nd in run.nodes if nd.status != STATUS_FINISHED]
        assert bad


class TestFromStream:
    def test_parse_sse_file(self, sse_streaming_text):
        events = parse_sse_text(sse_streaming_text)
        kinds = {e.event for e in events}
        assert {"token", "agentFlowExecutedData", "metadata", "end"} <= kinds

    def test_assembled_run_matches_nonstream(self, sse_streaming_text, agent1_count):
        events = parse_sse_text(sse_streaming_text)
        run = AgentRun.from_stream("agent-1", events)
        assert run.streaming is True
        # same final answer as the non-streaming count capture
        assert run.text.strip() == agent1_count.text.strip()
        assert run.all_nodes_finished
        assert run.usage.total_tokens > 0

    def test_streams_are_indexed(self, sse_streaming_text):
        events = parse_sse_text(sse_streaming_text)
        assert [e.index for e in events] == list(range(len(events)))

    def test_empty_tokens_join_to_text(self, sse_streaming_text):
        events = parse_sse_text(sse_streaming_text)
        run = AgentRun.from_stream("a", events)
        assert run.tokens  # raw token deltas captured
        assert "".join(run.tokens).strip() == run.text.strip()


class TestContextAccessors:
    def test_retrieved_context_empty_by_default(self, agent1_nonstream):
        assert agent1_nonstream.retrieved_context() == []
        assert agent1_nonstream.reference is None

    def test_explicit_retrieved_contexts(self, agent1_nonstream):
        run = agent1_nonstream
        run.raw["retrieved_contexts"] = ["doc A", "doc B"]
        assert run.retrieved_context() == ["doc A", "doc B"]

    def test_node_derived_context(self, sse_streaming_text):
        # build a run and inject a node-level context to prove the lookup path
        events = parse_sse_text(sse_streaming_text)
        run = AgentRun.from_stream("a", events)
        for nd in run.nodes:
            if nd.is_agent:
                nd.output.setdefault("metadata", {})["context"] = "kb chunk"
        assert "kb chunk" in run.retrieved_context()

    def test_reference_via_raw(self, agent1_nonstream):
        run = agent1_nonstream
        run.raw["reference"] = "The expected answer."
        assert run.reference == "The expected answer."

    def test_available_tool_names_list(self, agent2_mcp):
        assert isinstance(agent2_mcp.available_tool_names, list)
