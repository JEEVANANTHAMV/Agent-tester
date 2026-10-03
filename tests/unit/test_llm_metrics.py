"""LLM-judged catalog scenarios (offline).

Two layers:
  1. **MockJudge** - each LLM metric's scoring math is verified by feeding it
     canned judge JSON (the exact shapes the real Forjinn judge returns).
  2. **ForjinnTransport plumbing** - a fake ForjinnClient proves the real chain
     JudgeClient -> ForjinnTransport -> ForjinnClient.predict(streaming) ->
     extract_json -> per-metric scoring, no network.

The *real* LLM path (actual judge over the live host) is in test_live_llm.py.
"""
from __future__ import annotations

import json

from forjinn_eval import (
    AgentRun,
    AllNodesFinished,
    ForjinnTransport,
    JudgeClient,
    JudgeError,
    MockJudge,
)
from forjinn_eval.catalog.llm import (
    AnswerCorrectness,
    AnswerRelevance,
    AnswerRelevancy,
    AnswerRelevancyDeepeval,
    Bias,
    Conformity,
    Contradiction,
    ExactMatch,
    FactualCorrectness,
    Faithfulness,
    GEval,
    GoalAccuracy,
    Groundedness,
    Hallucination,
    LLMJudge,
    NonLLMStringSimilarity,
    PIILeakage,
    PromptAlignment,
    StringPresence,
    TaskCompletion,
    ToolUse,
    TopicAdherence,
    Toxicity,
    default_judge_metrics,
)
from forjinn_eval.results import Status


def _S(c) -> str:
    return c.status.value


def _ctx(run: AgentRun) -> AgentRun:
    run.raw["retrieved_contexts"] = ["Paris is the capital of France."]
    return run


class TestRagAssembled:
    def test_faithfulness_pass(self, agent1_count):
        j = MockJudge()
        j.add({"statements": ["Paris is the capital of France."], })
        j.add({"statements": [{"statement": "s", "reason": "ok", "verdict": 1}]})
        res = Faithfulness(judge=j, threshold=0.9).evaluate(_ctx(agent1_count))
        assert _S(res) == Status.PASS.value and res.score == 1.0

    def test_faithfulness_skip_without_context(self, agent1_count):
        assert _S(Faithfulness(judge=MockJudge()).evaluate(agent1_count)) == Status.SKIP.value

    def test_answer_relevancy_majority(self, agent1_count):
        j = MockJudge()
        for s in (9, 6, 9):
            j.add({"score": s, "reason": "ok"})
        res = AnswerRelevancy(judge=j, threshold=0.5).evaluate(agent1_count)
        assert res.score == 0.8

    def test_answer_relevancy_deepeval(self, agent1_count):
        j = MockJudge()
        j.add({"statements": ["a", "b"]})
        j.add({"verdicts": [{"verdict": "yes"}, {"verdict": "no"}]})
        assert AnswerRelevancyDeepeval(judge=j).evaluate(agent1_count).score == 0.5

    def test_answer_correctness_without_reference_skips(self, agent1_count):
        assert _S(AnswerCorrectness(judge=MockJudge()).evaluate(agent1_count)) == Status.SKIP.value

    def test_factual_correctness_without_reference_skips(self, agent1_count):
        assert _S(FactualCorrectness(judge=MockJudge()).evaluate(agent1_count)) == Status.SKIP.value

    def test_topic_adherence_skip_without_reference_topics(self, agent1_count):
        assert _S(TopicAdherence(judge=MockJudge()).evaluate(agent1_count)) == Status.SKIP.value


class TestDeepevalFamily:
    def test_hallucination_score(self, agent1_count):
        j = MockJudge()
        j.add({"verdicts": [{"verdict": "yes"}, {"verdict": "no"}]})
        assert Hallucination(judge=j, threshold=0.5).evaluate(_ctx(agent1_count)).score == 0.5

    def test_bias_no_opinions_passes(self, agent1_count):
        j = MockJudge(); j.add({"opinions": []})
        assert Bias(judge=j, threshold=1.0).evaluate(agent1_count).score == 1.0

    def test_toxicity_no_opinions_passes(self, agent1_count):
        j = MockJudge(); j.add({"opinions": []})
        assert Toxicity(judge=j, threshold=1.0).evaluate(agent1_count).score == 1.0

    def test_pii_leakage_clean(self, agent1_count):
        j = MockJudge()
        j.add({"extracted_pii": ["stmt"]})
        j.add({"verdicts": [{"verdict": "no"}]})
        assert PIILeakage(judge=j, threshold=1.0).evaluate(agent1_count).score == 1.0

    def test_prompt_alignment(self, agent1_count):
        j = MockJudge()
        j.add({"verdicts": [{"verdict": "yes"}, {"verdict": "no"}]})
        res = PromptAlignment(judge=j, instructions=["i1", "i2"]).evaluate(agent1_count)
        assert res.score == 0.5

    def test_prompt_alignment_skip_on_huge_system_prompt(self, agent1_count):
        # the recorded agent's 25k-char system prompt is not a usable instruction set
        assert _S(PromptAlignment(judge=MockJudge()).evaluate(agent1_count)) == Status.SKIP.value

    def test_task_completion(self, agent1_count):
        j = MockJudge()
        j.add({"task": "Count 1..5", "outcome": "1..5 produced"})
        j.add({"verdict": 0.9, "reason": "done"})
        assert TaskCompletion(judge=j, threshold=0.8).evaluate(agent1_count).score == 0.9

    def test_goal_accuracy(self, agent1_count):
        j = MockJudge()
        j.add({"user_goal": "Count 1..5", "end_state": "Counted 1 to 5."})
        j.add({"reason": "matches", "verdict": "1"})
        assert GoalAccuracy(judge=j).evaluate(agent1_count).score == 1.0

    def test_tool_use_min_of_halves(self, agent1_count):
        from forjinn_eval import ToolCall

        agent1_count.called_tools = [
            ToolCall.from_dict({"name": "calc", "arguments": {"expr": "1+2"}}),
            ToolCall.from_dict({"name": "lookup", "arguments": {"key": "x"}}),
        ]
        j = MockJudge()
        j.add({"score": 0.6, "reason": "sel"})
        j.add({"score": 0.5, "reason": "arg calc"})
        j.add({"score": 0.7, "reason": "arg lookup"})
        res = ToolUse(judge=j, threshold=0.5).evaluate(agent1_count)
        assert res.score == 0.6  # min(0.6, 0.6)
        assert res.details["argument_correctness"] == 0.6

    def test_tool_use_skips_when_no_tools(self, agent1_count):
        assert _S(ToolUse(judge=MockJudge()).evaluate(agent1_count)) == Status.SKIP.value


class TestGEval:
    def test_g_eval_normalization(self, agent1_count):
        j = MockJudge()
        j.add({"steps": ["check format", "check content"]})
        j.add({"score": 8, "reason": "strong"})
        res = GEval(judge=j, criteria="Be concise.").evaluate(agent1_count)
        assert res.score == 0.8

    def test_g_eval_strict(self, agent1_count):
        j = MockJudge()
        j.add({"steps": ["s"]})
        j.add({"score": 9, "reason": "ok"})
        assert GEval(judge=j, criteria="c", strict_mode=True).evaluate(agent1_count).score == 1.0

    def test_g_eval_requires_criteria_or_steps(self):
        try:
            GEval(judge=MockJudge())
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


class TestGiskardJudges:
    def test_llmjudge_pass(self, agent1_count):
        j = MockJudge(); j.add({"reason": "ok", "passed": True})
        res = LLMJudge(judge=j, instruction="must count").evaluate(agent1_count)
        assert _S(res) == Status.PASS.value and res.details["reason"] == "ok"

    def test_llmjudge_fail(self, agent1_count):
        j = MockJudge(); j.add({"reason": "bad", "passed": False})
        assert _S(LLMJudge(judge=j, instruction="must count").evaluate(agent1_count)) == Status.FAIL.value

    def test_llmjudge_error_without_instruction(self, agent1_count):
        assert _S(LLMJudge(judge=MockJudge()).evaluate(agent1_count)) == Status.ERROR.value

    def test_groundedness(self, agent1_count):
        j = MockJudge(); j.add({"reason": "supported", "passed": True})
        assert _S(Groundedness(judge=j).evaluate(_ctx(agent1_count))) == Status.PASS.value
        j2 = MockJudge(); j2.add({"reason": "ungrounded", "passed": False})
        assert _S(Groundedness(judge=j2).evaluate(_ctx(agent1_count))) == Status.FAIL.value

    def test_groundedness_skips_without_context(self, agent1_count):
        assert _S(Groundedness(judge=MockJudge()).evaluate(agent1_count)) == Status.SKIP.value

    def test_contradiction(self, agent1_count):
        j = MockJudge(); j.add({"reason": "no conflict", "passed": True})
        assert _S(Contradiction(judge=j).evaluate(_ctx(agent1_count))) == Status.PASS.value
        j2 = MockJudge(); j2.add({"reason": "conflict", "passed": False})
        assert _S(Contradiction(judge=j2).evaluate(_ctx(agent1_count))) == Status.FAIL.value

    def test_conformity(self, agent1_count):
        j = MockJudge(); j.add({"reason": "conforms", "passed": True})
        assert _S(Conformity(judge=j, rule="only numbers").evaluate(agent1_count)) == Status.PASS.value

    def test_answer_relevance(self, agent1_count):
        j = MockJudge(); j.add({"reason": "relevant", "passed": True})
        assert _S(AnswerRelevance(judge=j).evaluate(agent1_count)) == Status.PASS.value


class TestDeterministicString:
    def test_exact_match(self, agent1_count):
        assert ExactMatch(reference=agent1_count.text).evaluate(agent1_count).score == 1.0
        assert ExactMatch(reference="nope").evaluate(agent1_count).score == 0.0

    def test_string_presence(self, agent1_count):
        assert StringPresence(reference="1\n2").evaluate(agent1_count).score == 1.0
        assert StringPresence(reference="ZZZZ").evaluate(agent1_count).score == 0.0

    def test_string_similarity_bounds(self, agent1_count):
        assert NonLLMStringSimilarity(reference=agent1_count.text).evaluate(agent1_count).score == 1.0
        assert 0.0 <= NonLLMStringSimilarity(reference="q").evaluate(agent1_count).score <= 1.0

    def test_reference_metrics_skip_without_reference(self, agent1_count):
        assert _S(ExactMatch().evaluate(agent1_count)) == Status.SKIP.value
        assert _S(StringPresence().evaluate(agent1_count)) == Status.SKIP.value
        assert _S(NonLLMStringSimilarity().evaluate(agent1_count)) == Status.SKIP.value


class TestJudgePlumbingOffline:
    """Real JudgeClient -> ForjinnTransport -> (fake) ForjinnClient -> SSE -> extract_json."""

    def test_judge_client_requires_chatflow(self):
        import os

        saved = os.environ.pop("FORJINN_JUDGE_CHATFLOW", None)
        try:
            try:
                JudgeClient()
                raise AssertionError("expected JudgeError")
            except JudgeError:
                pass
        finally:
            if saved is not None:
                os.environ["FORJINN_JUDGE_CHATFLOW"] = saved

    def test_mockjudge_queue_then_responder(self):
        j = MockJudge(responder=lambda q: '{"score": 9}')
        j.add({"score": 8})
        assert j.complete("first") == '{"score": 8}'
        assert j.complete("second") == '{"score": 9}'
        assert j.calls == ["first", "second"]

    def test_forjinn_transport_streams_end_to_end(self, agent1_count):
        streamed = json.dumps({"statements": [{"statement": "Paris is the capital of France.",
                                               "reason": "yes", "verdict": 1}]})
        seen = []

        class _FakeForjinnClient:
            def predict(self, cf, q, streaming=False, **kw):
                seen.append(streaming)
                r = AgentRun(chatflow_id="judge", question=q, streaming=streaming)
                r.text = streamed
                return r

        transport = ForjinnTransport("judge-cf", streaming=True, client=_FakeForjinnClient())
        judge = JudgeClient(transport=transport, streaming=True)
        assert judge.streaming is True

        _ctx(agent1_count)
        res = Faithfulness(judge=judge, threshold=0.9).evaluate(agent1_count)
        assert _S(res) == Status.PASS.value and res.score == 1.0
        assert seen and all(seen)  # both judge calls used the SSE streaming path
        assert len(transport.calls) >= 2

    def test_offline_overlay_attaches_battery(self, agent1_count):
        import os

        os.environ["FORJINN_OFFLINE"] = "1"
        os.environ["FORJINN_LLM_JUDGE"] = "1"
        os.environ.pop("FORJINN_JUDGE_CHATFLOW", None)
        try:
            from forjinn_eval import SuiteRunner, make_case

            sr = SuiteRunner(suite_name="t").run([make_case("c", agent1_count, [AllNodesFinished()])])
            evs = {c.name for c in sr.results[0].check_results}
            for nm in ("answer_relevancy", "faithfulness", "hallucination", "bias", "toxicity"):
                assert nm in evs
            assert "all_nodes_finished" in evs
            assert sr.passed == 1
        finally:
            os.environ.pop("FORJINN_OFFLINE", None)
            os.environ.pop("FORJINN_LLM_JUDGE", None)


def test_default_judge_metrics_bundle():
    ms = default_judge_metrics(judge=MockJudge())
    assert len(ms) >= 5
    assert all(m.judge is not None for m in ms)
