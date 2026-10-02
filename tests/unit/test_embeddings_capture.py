"""Offline tests for the embedding backend and the multi-turn capture layer."""
from __future__ import annotations

from forjinn_eval import (
    AgentRun,
    Conversation,
    Message,
    Embeddings,
    _cosine,
)
from forjinn_eval.judge import Embeddings as Emb2


class TestEmbeddings:
    def test_identical_text_is_one(self):
        e = Embeddings()
        s = e.similarity("the cat sat on the mat", "the cat sat on the mat")
        assert s > 0.9999

    def test_dissimilar_texts_low(self):
        e = Embeddings()
        s = e.similarity("the cat sat on the mat", "quantum entanglement of photons")
        assert 0.0 <= s < 0.4

    def test_custom_embedder_is_used(self):
        def embed(text):
            return [1.0] if "cat" in text else [0.0, 1.0]
        e = Embeddings(embedder=embed)
        assert e.similarity("a cat", "another cat") == 1.0
        assert e.similarity("a cat", "the dog") == 0.0

    def test_cosine_bounds(self):
        assert -1.0 <= _cosine([1.0], [1.0]) <= 1.0
        assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_from_env_and_singleton_types_match(self):
        assert Emb2 is Embeddings


class TestConversation:
    def test_from_messages_pairs_turns(self):
        c = Conversation.from_messages([
            {"role": "human", "content": "hi"},
            {"role": "ai", "content": "hello"},
            {"role": "human", "content": "bye"},
            {"role": "ai", "content": "goodbye"},
        ])
        assert len(c.human_turns) == 2
        assert len(c.ai_turns) == 2
        assert c.turns[0].is_human and c.turns[1].is_ai

    def test_from_runs_stitches(self):
        r1 = AgentRun(chatflow_id="cf", question="q1", text="a1")
        r2 = AgentRun(chatflow_id="cf", question="q2", text="a2")
        c = Conversation.from_runs(r1, r2)
        assert [m.content for m in c.human_turns] == ["q1", "q2"]
        assert [m.content for m in c.ai_turns] == ["a1", "a2"]

    def test_retrieval_contexts_cumulative(self):
        c = Conversation()
        c.add_human("q1")
        c.messages.append(Message(role="ai", content="a1", retrieval_contexts=["c1"]))
        c.add_human("q2")
        c.messages.append(Message(role="ai", content="a2", retrieval_contexts=["c2"]))
        assert c.retrieval_contexts(upto=2) == ["c1"]
        assert c.retrieval_contexts() == ["c1", "c2"]


class TestAgentRunFromConversation:
    def test_text_and_question_from_transcript(self):
        c = Conversation()
        c.add_human("What is 2+2?")
        c.add_ai("2+2 is 4.")
        run = AgentRun.from_conversation(c)
        assert run.question == "What is 2+2?"
        assert run.text == "2+2 is 4."

    def test_reference_via_metadata(self):
        c = Conversation.from_messages([
            {"role": "human", "content": "q"}, {"role": "ai", "content": "a"},
        ])
        c.metadata["reference"] = "ref"
        run = AgentRun.from_conversation(c)
        assert run.reference == "ref"

    def test_turns_accessor(self):
        c = Conversation.from_messages([
            {"role": "human", "content": "q"}, {"role": "ai", "content": "a"},
        ])
        run = AgentRun.from_conversation(c)
        assert len(run.turns()) == 2

    def test_single_turn_view_fallback(self):
        run = AgentRun(chatflow_id="x", question="q", text="a")
        conv = run.conversation
        assert conv.turns[0].is_human and conv.turns[1].is_ai
