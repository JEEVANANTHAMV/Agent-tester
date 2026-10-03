"""Offline tests for the newly-added RAG/contextual/citation metrics."""
from __future__ import annotations

from forjinn_eval import (
    AnswerAccuracy,
    CitationFaithfulness,
    ContextEntityRecall,
    ContextualPrecision,
    ContextualRecall,
    ContextualRelevancy,
    MockJudge,
    NoiseSensitivity,
    QuotedSpansAlignment,
)
from forjinn_eval.results import Status
from tests.unit._helpers import canned_judge, single_turn_run


def _ctx_run():
    return single_turn_run(
        question="What is the capital of France?",
        text="The capital of France is Paris [1]. It has museums.",
        reference="The capital of France is Paris. It has museums.",
        contexts=["Paris is the capital of France. It has museums.",
                  "The Eiffel Tower is in Paris."],
    )


class TestContextualPrecision:
    def test_pass_when_all_relevant(self):
        j = canned_judge({"relevant": [1, 1]})
        res = ContextualPrecision(judge=j, threshold=0.5).evaluate(_ctx_run())
        assert res.status == Status.PASS and res.score == 1.0

    def test_partial_precision(self):
        j = canned_judge({"relevant": [1, 0, 1]})
        res = ContextualPrecision(judge=j, threshold=0.5).evaluate(_ctx_run())
        assert 0.0 <= res.score <= 1.0

    def test_skip_without_context(self):
        run = single_turn_run("q", "a", reference="a")
        assert ContextualPrecision(judge=MockJudge()).evaluate(run).status == Status.SKIP


class TestContextualRecall:
    def test_full_recall(self):
        j = canned_judge({"supported": [1, 1]})
        res = ContextualRecall(judge=j, threshold=0.5).evaluate(_ctx_run())
        assert res.status == Status.PASS and res.score == 1.0

    def test_half_recall(self):
        j = canned_judge({"supported": [1, 0]})
        res = ContextualRecall(judge=j, threshold=0.5).evaluate(_ctx_run())
        assert res.score == 0.5

    def test_skip_without_context(self):
        run = single_turn_run("q", "a", reference="a")
        assert ContextualRecall(judge=MockJudge()).evaluate(run).status == Status.SKIP
        assert ContextualRecall(judge=MockJudge()).evaluate(
            single_turn_run("q", "a", contexts=["x"], reference=None)).status == Status.SKIP


class TestContextualRelevancy:
    def test_embedding_similarity_bounds(self):
        res = ContextualRelevancy(threshold=0.0).evaluate(_ctx_run())
        assert 0.0 <= res.score <= 1.0

    def test_skip_without_context(self):
        run = single_turn_run("q", "a")
        assert ContextualRelevancy().evaluate(run).status == Status.SKIP


class TestContextEntityRecall:
    def test_full_entity_recall(self):
        j = canned_judge({"recall_verdicts": [{"entity": "Paris", "in_context": 1},
                                              {"entity": "museums", "in_context": 1}]})
        res = ContextEntityRecall(judge=j).evaluate(_ctx_run())
        assert res.score == 1.0

    def test_skip_without_context_or_ref(self):
        assert ContextEntityRecall(judge=MockJudge()).evaluate(
            single_turn_run("q", "a")).status == Status.SKIP


class TestCitationFaithfulness:
    def test_all_faithful(self):
        j = canned_judge({"verdicts": [{"cite": 1, "faithful": 1, "reason": "ok"}]})
        res = CitationFaithfulness(judge=j).evaluate(_ctx_run())
        assert res.status == Status.PASS and res.score == 1.0

    def test_misattributed_citation_fails(self):
        j = canned_judge({"verdicts": [{"cite": 1, "faithful": 0, "reason": "wrong"}]})
        res = CitationFaithfulness(judge=j, threshold=1.0).evaluate(_ctx_run())
        assert res.status == Status.FAIL and res.score == 0.0

    def test_no_citations_passes(self):
        run = single_turn_run("q", "no citations here", contexts=["ctx"])
        res = CitationFaithfulness(judge=MockJudge()).evaluate(run)
        assert res.status == Status.PASS and res.score == 1.0


class TestNoiseSensitivity:
    def test_no_noise(self):
        j = canned_judge({"statements": ["a", "b"]}, {"verdicts": [1, 1]})
        res = NoiseSensitivity(judge=j).evaluate(_ctx_run())
        assert res.score == 1.0

    def test_some_unsupported_claims(self):
        j = canned_judge({"statements": ["a", "b"]}, {"verdicts": [1, 0]})
        res = NoiseSensitivity(judge=j).evaluate(_ctx_run())
        assert res.score == 0.5

    def test_skip_without_reference(self):
        run = single_turn_run("q", "a", contexts=["x"])
        assert NoiseSensitivity(judge=MockJudge()).evaluate(run).status == Status.SKIP


class TestAnswerAccuracy:
    def test_top_rating(self):
        j = canned_judge({"rating": 4}, {"rating": 4})
        res = AnswerAccuracy(judge=j).evaluate(_ctx_run())
        assert res.status == Status.PASS and res.score == 1.0

    def test_low_rating(self):
        j = canned_judge({"rating": 1}, {"rating": 1})
        res = AnswerAccuracy(judge=j, threshold=0.5).evaluate(_ctx_run())
        assert res.status == Status.FAIL and res.score == 0.0

    def test_skip_without_reference(self):
        run = single_turn_run("q", "a")
        assert AnswerAccuracy(judge=MockJudge()).evaluate(run).status == Status.SKIP


class TestQuotedSpansAlignment:
    def test_quoted_span_found(self):
        run = single_turn_run("q", "As stated: \"Paris is the capital of France today.\"",
                              contexts=["Paris is the capital of France today."])
        res = QuotedSpansAlignment(min_span_words=4).evaluate(run)
        assert res.status == Status.PASS and res.score == 1.0

    def test_quoted_span_not_found(self):
        run = single_turn_run("q", "He said \"the sun is a green cube, obviously.\"",
                              contexts=["Paris is the capital of France."])
        res = QuotedSpansAlignment(min_span_words=4).evaluate(run)
        assert res.status == Status.FAIL and res.score == 0.0

    def test_no_spans_passes(self):
        run = single_turn_run("q", "no quotes at all here", contexts=["ctx"])
        assert QuotedSpansAlignment().evaluate(run).score == 1.0
