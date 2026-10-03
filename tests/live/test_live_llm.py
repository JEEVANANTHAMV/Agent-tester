"""REAL-LLM tests: exercise the Forjinn SSE judge against a LIVE host.

These are the "not only mock, actual LLM" tests. They skip (no fail) unless the
environment opts in and the host is reachable:

    FORJINN_LLM_TESTS=1                 # opt in to running the live-LLM suite
    FORJINN_HOST=<https://host>         # Forjinn builder host (default 172.16.34.7)
    FORJINN_TOKEN=<jwt>                 # optional
    FORJINN_JUDGE_CHATFLOW=<chatflowId> # a Forjinn agent built as an LLM judge
    FORJINN_JUDGE_STREAMING=1           # force the SSE path (recommended)

If ``FORJINN_JUDGE_CHATFLOW`` is unset we fall back to any working chatflow so
the *transport* (ForjinnClient -> SSE -> extract_json) is still exercised
end-to-end; the judge-specific semantic checks (faithfulness, GEval, ...) are
skipped in that case but the plumbing + JSON-extraction checks still run.

Run with:  pytest -m live  (or: pytest tests/test_live_llm.py)
"""
from __future__ import annotations

import os

import pytest

from forjinn_eval import (
    AgentRun,
    Faithfulness,
    ForjinnClient,
    GEval,
    JudgeClient,
    JudgeError,
    LLMJudge,
    extract_json,
)
from forjinn_eval.judge import ForjinnTransport

pytestmark = pytest.mark.live

HOST = os.environ.get("FORJINN_HOST", "https://172.16.34.7")
JUDGE_CF = os.environ.get("FORJINN_JUDGE_CHATFLOW")
# Any working chatflow on the host (the known agent-1 no-tools) - used to prove
# the transport when no dedicated judge chatflow is configured.
FALLBACK_CF = os.environ.get("FORJINN_AGENT_FALLBACK", "03d5abc5-6ecd-4891-a9a1-364aefb33a50")


def _opt_in() -> bool:
    return os.environ.get("FORJINN_LLM_TESTS", "").strip().lower() in {"1", "true", "yes", "on"}


requires_llm = pytest.mark.skipif(
    not _opt_in(), reason="set FORJINN_LLM_TESTS=1 to run live-LLM tests"
)


def _reachable() -> bool:
    try:
        ForjinnClient(HOST, timeout=20).predict(FALLBACK_CF, "ping")
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def host_ok():
    """True when the Forjinn host answers a real prediction (no proxy)."""
    return _reachable()


def _judge(streaming=True):
    cf = JUDGE_CF or FALLBACK_CF
    return JudgeClient(judge_chatflow=cf, base_url=HOST, token=os.environ.get("FORJINN_TOKEN"), streaming=streaming)


# ---------------------------------------------------------------------------
# Transport: Forjinn SSE judge works end-to-end (streaming + non-streaming)
# ---------------------------------------------------------------------------
@requires_llm
class TestForjinnJudgeTransport:
    def test_nonstream_returns_text(self, host_ok):
        assert host_ok
        j = _judge(streaming=False)
        txt = j.complete("Reply with exactly the single word PONG.", max_tokens=50)
        assert txt and "PONG" in txt.upper()

    def test_stream_sse_returns_same_text(self, host_ok):
        assert host_ok
        j = _judge(streaming=True)
        txt = j.complete("Reply with exactly the single word PONG.", max_tokens=50)
        assert txt and "PONG" in txt.upper()

    def test_streaming_text_matches_nonstream_text(self, host_ok):
        """SSE-assembled text is identical to the non-streaming answer for the
        same deterministic prompt (the 'same AgentRun, same evaluators' guarantee)."""
        assert host_ok
        q = "What is 2 + 3? Reply with just the number, no punctuation."
        a = _judge(streaming=False).complete(q, max_tokens=50)
        b = _judge(streaming=True).complete(q, max_tokens=50)
        assert a.strip() == b.strip()
        assert a.strip() == "5"

    def test_unknown_chatflow_is_a_forjinn_error(self, host_ok):
        """A bogus chatflow is a Forjinn API error; the transport surfaces it
        (wrapped after retries) as a JudgeError carrying the ForjinnError cause."""
        assert host_ok
        j = JudgeClient(judge_chatflow="00000000-0000-0000-0000-000000000000", base_url=HOST, streaming=False)
        with pytest.raises(JudgeError) as ei:
            j.complete("x", max_tokens=50)
        # Forjinn reports the missing chatflow; the judge surfaces it via the error
        assert "not found in the database" in str(ei.value)

    def test_forjinn_transport_records_calls_and_streams(self, host_ok):
        assert host_ok

        class _Fake:
            def __init__(self):
                self.seen = []

            def predict(self, cf, q, streaming=False, **kw):
                self.seen.append(streaming)
                r = AgentRun(chatflow_id=cf, question=q, streaming=streaming)
                r.text = '{"ok": 1}'
                return r

        fake = _Fake()
        t = ForjinnTransport("fake", streaming=True, client=fake)
        assert t.complete("hello", max_tokens=5) == '{"ok": 1}'
        # ForjinnTransport records the question; the fake saw the SSE (streaming) flag True
        assert t.calls == ["hello"]
        assert fake.seen == [True]
        assert extract_json('{"ok": 1}') == {"ok": 1}


# ---------------------------------------------------------------------------
# JSON extraction from a real LLM's (free-text) reply
# ---------------------------------------------------------------------------
@requires_llm
class TestRealJsonExtraction:
    def test_extract_json_from_real_reply(self, host_ok):
        assert host_ok
        j = _judge(streaming=True)
        raw = j.complete(
            'Answer with ONLY valid JSON, no prose: {"verdict": "yes", "score": 9}.',
            max_tokens=60,
        )
        obj = extract_json(raw)
        assert isinstance(obj, dict)
        assert obj.get("verdict") == "yes"
        assert obj.get("score") == 9

    def test_complete_json_helper(self, host_ok):
        assert host_ok
        j = _judge(streaming=True)
        obj = j.complete_json('Answer with ONLY valid JSON: {"verdict":"yes"}.', max_tokens=50)
        assert obj and obj.get("verdict") == "yes"


# ---------------------------------------------------------------------------
# Full LLM-metric pipeline driven by the real Forjinn judge
# ---------------------------------------------------------------------------
@requires_llm
class TestRealLLMMetrics:
    @staticmethod
    def _run_with_context():
        run = ForjinnClient(HOST).predict(
            FALLBACK_CF, "Count from 1 to 3, one number per line."
        )
        run.raw["retrieved_contexts"] = [
            "Paris is the capital of France. The Eiffel Tower is located in Paris."
        ]
        return run

    def test_geval_says_real_judge_scores(self, host_ok):
        assert host_ok
        j = _judge(streaming=True)
        run = self._run_with_context()
        res = GEval(judge=j, criteria="The answer should be a short, correct sequence of numbers.",
                    threshold=0.1).evaluate(run)
        # a live judge produces a concrete 0-1 score (PASS or FAIL, not ERROR)
        assert res.status.value in ("PASS", "FAIL")
        assert res.score is not None and 0.0 <= res.score <= 1.0

    def test_llmjudge_real_reason(self, host_ok):
        assert host_ok
        j = _judge(streaming=True)
        run = self._run_with_context()
        res = LLMJudge(judge=j, instruction="Does the answer consist only of digits and newlines?").evaluate(run)
        # real judge returns a reason + verdict (pass or fail both acceptable here)
        assert res.status.value in ("PASS", "FAIL")
        assert res.details.get("reason")

    def test_faithfulness_end_to_end_when_judge_configured(self, host_ok):
        """Only valid with a real judge chatflow (a plain text model can be
        instructed to emit the NLI verdict schema). Skips otherwise."""
        if not JUDGE_CF:
            pytest.skip("no FORJINN_JUDGE_CHATFLOW configured")
        assert host_ok
        j = _judge(streaming=True)
        run = self._run_with_context()
        run.text = "The Eiffel Tower is in Paris and Paris is the capital of France."
        res = Faithfulness(judge=j, threshold=0.3).evaluate(run)
        assert res.status.value in ("PASS", "FAIL", "SKIP")
