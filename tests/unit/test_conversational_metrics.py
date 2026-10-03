"""Offline tests for the multi-turn / conversational metric family."""
from __future__ import annotations

from forjinn_eval import (
    AgentRun,
    ConversationalGEval,
    GoalAccuracyMulti,
    MockJudge,
    MultiTurnToolUse,
    MultiTurnTopicAdherence,
    TurnContextualPrecision,
    TurnContextualRecall,
    TurnContextualRelevancy,
    TurnFaithfulness,
    TurnRelevancy,
)
from forjinn_eval.results import Status
from tests.unit._helpers import canned_judge, multi_turn_run


def _conv():
    return multi_turn_run(
        [
            ("What is 2+2?", "2+2 is 4.", ["Math: 2+2=4."]),
            ("And 3+3?", "3+3 is 6.", ["Math: 3+3=6."]),
            ("Multiply those.", "4 * 6 = 24."),
        ],
        reference="24",
    )


def _tool_conv():
    return multi_turn_run(
        [
            ("How many tables?", "5 tables in db1.",
             ["db1 has 5 tables."], ["list_tables"]),
            ("Query them.", "Done, here are the rows."),
        ],
    )


class TestTurnFaithfulness:
    def test_all_faithful(self):
        j = canned_judge({"faithful": 1}, {"faithful": 1}, {"faithful": 1})
        res = TurnFaithfulness(judge=j, threshold=0.5).evaluate(_conv())
        assert res.status == Status.PASS and res.score == 1.0

    def test_one_ungrounded_turn(self):
        j = canned_judge({"faithful": 1}, {"faithful": 0}, {"faithful": 1})
        res = TurnFaithfulness(judge=j, threshold=0.5).evaluate(_conv())
        assert res.score == round(2 / 3, 4)

    def test_skips_empty(self):
        empty = AgentRun(chatflow_id="x", question="", text="")
        assert TurnFaithfulness(judge=MockJudge()).evaluate(empty).status == Status.SKIP


class TestTurnRelevancy:
    def test_all_relevant(self):
        j = canned_judge({"score": 10}, {"score": 10}, {"score": 10})
        res = TurnRelevancy(judge=j).evaluate(_conv())
        assert res.score == 1.0

    def test_skips_empty(self):
        empty = AgentRun(chatflow_id="x", question="", text="")
        assert TurnRelevancy(judge=MockJudge()).evaluate(empty).status == Status.SKIP


class TestTurnContextual:
    def test_precision(self):
        j = canned_judge({"useful": [1, 1]}, {"useful": [1]}, {"useful": []})
        res = TurnContextualPrecision(judge=j).evaluate(_conv())
        assert 0.0 <= res.score <= 1.0

    def test_recall_with_reference(self):
        j = canned_judge({"covered": [1, 1]}, {"covered": [1]}, {"covered": []})
        res = TurnContextualRecall(judge=j, reference="24", threshold=0.0).evaluate(_conv())
        assert 0.0 <= res.score <= 1.0

    def test_relevancy_bound(self):
        # embedding-only metric (no judge call); a mock judge satisfies the base init
        res = TurnContextualRelevancy(judge=MockJudge(), threshold=0.0).evaluate(_conv())
        assert 0.0 <= res.score <= 1.0


class TestMultiTurnTopicAdherence:
    def test_on_topic(self):
        j = canned_judge({"on_topic": 1})
        res = MultiTurnTopicAdherence(["math"], judge=j).evaluate(_conv())
        assert res.status == Status.PASS and res.score == 1.0

    def test_requires_topics(self):
        try:
            MultiTurnTopicAdherence([])
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


class TestMultiTurnToolUse:
    def test_correct_tools(self):
        j = canned_judge({"tool_selection": 1, "argument_correctness": 1, "reason": "ok"})
        res = MultiTurnToolUse(judge=j).evaluate(_tool_conv())
        assert res.status == Status.PASS and res.score == 1.0

    def test_skips_without_tools(self):
        assert MultiTurnToolUse(judge=MockJudge()).evaluate(_conv()).status == Status.SKIP


class TestGoalAccuracyMulti:
    def test_achieved(self):
        # 5-level scale: "Perfectly achieved" is top (1.0), "Fully achieved" is 0.75
        j = canned_judge({"goal": "Perfectly achieved", "reason": "ok"})
        res = GoalAccuracyMulti(judge=j).evaluate(_conv())
        assert res.status == Status.PASS and res.score == 1.0

    def test_not_achieved(self):
        j = canned_judge({"goal": "Not achieved", "reason": "no"})
        res = GoalAccuracyMulti(judge=j, threshold=0.5).evaluate(_conv())
        assert res.status == Status.FAIL and res.score == 0.0

    def test_does_not_shadow_single_turn(self):
        # single-turn GoalAccuracy still imports and works as before
        from forjinn_eval import GoalAccuracy
        j = canned_judge({"user_goal": "g", "end_state": "s"},
                         {"reason": "ok", "verdict": "1"})
        run = AgentRun(chatflow_id="x", question="q", text="out")
        assert GoalAccuracy(judge=j).evaluate(run).score == 1.0


class TestConversationalGEval:
    def test_score(self):
        j = canned_judge({"steps": ["answer correctly"]}, {"score": 9, "reason": "good"})
        res = ConversationalGEval(judge=j, criteria="answer correctly").evaluate(_conv())
        assert res.status == Status.PASS and res.score == 0.9

    def test_requires_criteria(self):
        try:
            ConversationalGEval(judge=MockJudge())
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
