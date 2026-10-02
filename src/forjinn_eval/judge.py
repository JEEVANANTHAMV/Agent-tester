"""Pluggable *LLM judge* backend for the LLM-judged metric catalog.

The deterministic catalog (``catalog/``) never touches an LLM. The LLM-judged
metrics (faithfulness, answer-relevancy, hallucination, GEval, ...) *do* need an
LLM to render a verdict. In ``forjinn-eval`` the judge is **not** an OpenAI
endpoint - it is a **Forjinn agent**: build a small text-chat canvas agent with a
"judge"-style system prompt (any model, any host), get its chatflow id, and point
the metrics at it. Every judge call then goes through the exact same Forjinn
prediction API the agents under test use - **including the SSE streaming path**.

Why this is the right fit:
  * Reuses the one transport (``ForjinnClient``) + one API contract already
    battle-tested against the builder at 172.16.34.7 (noproxy, self-signed TLS
    bypass, ``x-request-from: internal``, optional JWT cookie).
  * The judge can be any of your Forjinn chatflows - including a model served by
    the same self-hosted vLLM Forjinn already runs - so you evaluate *and* judge
    with the same stack, no second SDK or API key.
  * Both transports are supported: non-streaming JSON and **SSE streaming**
    (the "Forjinn SSE compatible" path).

Configure:

    from forjinn_eval import JudgeClient
    judge = JudgeClient(
        judge_chatflow="11111111-....",      # the Forjinn chatflow that is your judge
        base_url="https://172.16.34.7",
    )
    Faithfulness(judge=judge)

Environment (auto-wire): ``FORJINN_JUDGE_CHATFLOW`` (required to enable the judge),
``FORJINN_HOST`` / ``FORJINN_JUDGE_BASE_URL``, ``FORJINN_TOKEN``, and
``FORJINN_JUDGE_STREAMING=1`` to force the SSE path (default: stream when the
host allows).

For offline / unit tests, inject a :class:`MockJudge` - identical interface, no
network.

The judge is only *asked* (system prompt + instructions live in the Forjinn
agent's canvas). The per-metric **rubric/question** is sent as the prediction
``question``; the judge is expected to answer with JSON, which we parse with
:meth:`complete_json` -> :func:`extract_json`.
"""
from __future__ import annotations

import json
import os
import re
import ssl  # noqa: F401  (kept for parity/optional TLS contexts)
import urllib.request  # noqa: F401
import urllib.error  # noqa: F401
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


class JudgeError(RuntimeError):
    """Raised when the Forjinn judge cannot be reached or returns unusable output."""


# ---------------------------------------------------------------------------
# JSON-in-text extraction + verdict coercion (shared by all LLM metrics)
# ---------------------------------------------------------------------------
def extract_json(text: str) -> Optional[Any]:
    """Parse JSON out of an LLM's free text.

    Tries, in order: whole-string ``json.loads``; the first ``{``..last ``}``
    substring; a trailing-comma-stripped version; and fenced ```json blocks.
    Returns ``None`` if nothing parses.
    """
    if not text:
        return None
    text = text.strip()
    candidates = [text]
    # fenced code blocks
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
        if m:
            candidates.insert(0, m.group(1))
    if "{" in text and text.rfind("}") > text.find("{"):
        blob = text[text.find("{"): text.rfind("}") + 1]
        candidates.append(blob)
        candidates.append(re.sub(r",\s*([}\]])", r"\1", blob))
    if "[" in text and text.rfind("]") > text.find("["):
        arr = text[text.find("["): text.rfind("]") + 1]
        candidates.append(arr)
    for c in candidates:
        try:
            return json.loads(c)
        except (ValueError, TypeError):
            continue
    return None


def as_int01(x: Any) -> int:
    """Coerce a judge verdict into 0/1 (ragas NLI ``verdict: int`` style).

    Tolerates the common nested shape ``{"verdict": 1}`` / ``{"faithful": 1}`` /
    ``{"relevant": 1}`` by probing a few well-known keys.
    """
    if isinstance(x, dict):
        for k in ("verdict", "faithful", "relevant", "supported", "useful", "correct",
                  "in_context", "covered", "on_topic"):
            if k in x:
                return as_int01(x[k])
        if "reason" in x:
            return 1  # a permissive/structured entry with no explicit negation
        return 0
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, float)):
        return int(round(float(x)))
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"1", "true", "yes", "supported", "faithful", "pass", "passing", "t"}:
            return 1
        if s in {"0", "false", "no", "unsupported", "unfaithful", "fail", "f"}:
            return 0
    return 0


def as_bool(x: Any) -> bool:
    """Coerce a judge verdict to bool (giskard ``passed`` style)."""
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "pass", "passing", "t", "supported"}
    return False


# ---------------------------------------------------------------------------
# Judge backend protocol (structural duck-typing)
# ---------------------------------------------------------------------------
class BaseJudge:
    """Interface the LLM metrics call.

    * :meth:`complete`      -> the judge's raw text answer.
    * :meth:`complete_json` -> that answer parsed as JSON (``None`` if unparseable).
    """

    def complete(self, question: str, *, max_tokens: int = 1024, retries: int = 2) -> str:  # pragma: no cover
        raise NotImplementedError

    def complete_json(self, question: str, *, max_tokens: int = 1024, retries: int = 2) -> Optional[Any]:  # pragma: no cover
        return extract_json(self.complete(question, max_tokens=max_tokens, retries=retries))


# ---------------------------------------------------------------------------
# ForjinnTransport - the seam between JudgeClient and the network
# ---------------------------------------------------------------------------
class ForjinnTransport(BaseJudge):
    """Default transport: a Forjinn chatflow, driven over the Forjinn API.

    ``complete`` uses the SSE streaming path when ``streaming`` is set (the
    "Forjinn SSE compatible" path); the streamed tokens are assembled into the
    same ``AgentRun.text`` the agent produced, so the judge's answer is clean.

    Tests can subclass this and override :meth:`complete` (without a network) to
    exercise the full real-JudgeClient plumbing - SSE text capture, JSON
    extraction, and per-metric scoring - offline.
    """

    def __init__(self, judge_chatflow: str, streaming: bool = False, client=None):
        from .client import ForjinnClient

        self.judge_chatflow = judge_chatflow
        self.streaming = bool(streaming)
        self._client = client or ForjinnClient(token=os.environ.get("FORJINN_TOKEN"))
        self.calls: List[str] = []  # last prompt sent (for debugging/tests)

    def complete(self, question: str, *, max_tokens: int = 1024, retries: int = 2) -> str:
        self.calls.append(question)
        last: Optional[Exception] = None
        attempt = 0
        while attempt <= max(1, retries):
            attempt += 1
            try:
                run = self._client.predict(self.judge_chatflow, question, streaming=self.streaming)
                text = (run.text or "").strip()
                if text:
                    return text
                last = JudgeError("judge returned empty answer")
                continue
            except Exception as e:  # network / HTTP / parse
                last = e
                continue
        raise JudgeError(f"Forjinn judge '{self.judge_chatflow}' failed: {last!r}")

    @property
    def client(self):  # for test/inspection
        return self._client


class JudgeClient(BaseJudge):
    """An LLM judge that is a **Forjinn chatflow**, called over Forjinn's own API.

    ``judge_chatflow`` is the id of a Forjinn agent built as an LLM judge: a
    plain text-chat canvas agent (no tools) whose system prompt instructs the
    model to answer the user's rubric as JSON. It can run on any Forjinn builder
    host and with any model that host serves.

    A custom :class:`ForjinnTransport` may be injected for testing or to reuse an
    existing :class:`ForjinnClient` session.
    """

    def __init__(
        self,
        judge_chatflow: Optional[str] = None,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[float] = None,
        streaming: Optional[bool] = None,
        verify_ssl: bool = False,
        noproxy: bool = True,
        temperature: float = 0.0,  # accepted for API parity; Forjinn sets temp on the agent
        transport: Optional[ForjinnTransport] = None,
    ):
        host = base_url or os.environ.get("FORJINN_JUDGE_BASE_URL") or os.environ.get("FORJINN_HOST") or "https://172.16.34.7"
        self.judge_chatflow = judge_chatflow or os.environ.get("FORJINN_JUDGE_CHATFLOW")
        self.api_key = token or os.environ.get("FORJINN_TOKEN")
        if transport is not None:
            self._transport = transport
            if not self.judge_chatflow:
                self.judge_chatflow = transport.judge_chatflow
        else:
            from .client import ForjinnClient  # local import avoids a cycle during package init

            if not self.judge_chatflow:
                raise JudgeError(
                    "JudgeClient requires a Forjinn judge chatflow: pass judge_chatflow="
                    "or set FORJINN_JUDGE_CHATFLOW to a chatflow id built as an LLM judge."
                )
            self._transport = ForjinnTransport(
                judge_chatflow=self.judge_chatflow,
                streaming=bool(streaming),
                client=ForjinnClient(
                    host,
                    token=self.api_key,
                    timeout=timeout or 120.0,
                    verify_ssl=verify_ssl,
                    noproxy=noproxy,
                ),
            )
        if streaming is None:
            streaming = os.environ.get("FORJINN_JUDGE_STREAMING", "").lower() in {"1", "true", "yes", "on"}
        self._transport.streaming = bool(streaming)
        self.temperature = temperature

    @property
    def streaming(self) -> bool:
        return self._transport.streaming

    def complete(self, question: str, *, max_tokens: int = 1024, retries: int = 2) -> str:
        """Ask the Forjinn judge ``question``; return its answer text."""
        return self._transport.complete(question, max_tokens=max_tokens, retries=retries)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<JudgeClient chatflow={self.judge_chatflow!r} streaming={self._transport.streaming}>"


# ---------------------------------------------------------------------------
# Embeddings (for embedding-based metrics: semantic similarity, relevancy, recall)
# ---------------------------------------------------------------------------
def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    c = dot / (na * nb)
    return max(-1.0, min(1.0, c))


class Embeddings:
    """Text-embedding backend with graceful degradation.

    Resolution order: (1) an injected ``embedder(text) -> List[float]`` callable
    (bring your own - e.g. a Forjinn embedding chatflow or an OpenAI-style
    client); (2) ``sentence-transformers`` if installed; (3) a deterministic
    hash-based bag-of-words model (always available, offline, no dependencies).

    The hash fallback keeps the metric *meaningful* (it scores by overlapping
    word content, which is the crux of semantic-similarity / recall metrics)
    rather than a random vector, so the whole catalog runs offline for Forjinn
    web without any network.
    """

    def __init__(self, embedder: Optional[Callable[[str], List[float]]] = None,
                 model: Optional[str] = None):
        self._embedder = embedder
        self._st = None
        self._st_model = model
        if embedder is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._st = SentenceTransformer(model or "all-MiniLM-L6-v2")
            except Exception:
                self._st = None

    @classmethod
    def from_env(cls, embedder: Optional[Callable[[str], List[float]]] = None) -> "Embeddings":
        """Build from env; ``EMBEDDING_MODEL`` selects a sentence-transformers model."""
        return cls(embedder=embedder, model=os.environ.get("EMBEDDING_MODEL") or None)

    def embed(self, text: str) -> List[float]:
        text = text or ""
        if self._embedder is not None:
            return list(self._embedder(text) or [])
        if self._st is not None:
            return list(self._st.encode(text, normalize_embeddings=True))
        return self._hash_embed(text)

    def _hash_embed(self, text: str) -> List[float]:
        """Deterministic, dependency-free bag-of-words embedding.

        Each token maps to a fixed bucket (via md5) with a unit weight; the
        vector is L2-normalised. Cosine of two such vectors is a token-overlap
        similarity (Jaccard-ish) - meaningful for semantic-similarity / recall
        metrics, fully offline, and stable across processes.
        """
        import hashlib

        dim = 512
        vec = [0.0] * dim
        words = re.findall(r"[a-z0-9']+", (text or "").lower())
        for w in words:
            i = int.from_bytes(hashlib.md5(w.encode("utf-8")).digest()[:4], "little") % dim
            vec[i] += 1.0
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def similarity(self, a: str, b: str) -> float:
        return _cosine(self.embed(a), self.embed(b))

    def cosine(self, a: List[float], b: List[float]) -> float:
        return _cosine(a, b)


def from_env() -> Optional["BaseJudge"]:
    """Build a :class:`JudgeClient` from environment, or ``None`` if not configured.

    Enabled when ``FORJINN_JUDGE_CHATFLOW`` is set (optionally forced to stream via
    ``FORJINN_JUDGE_STREAMING``).
    """
    if not os.environ.get("FORJINN_JUDGE_CHATFLOW"):
        return None
    return JudgeClient()


# ---------------------------------------------------------------------------
# Offline judge for unit tests / FORJINN_OFFLINE demos
# ---------------------------------------------------------------------------
class MockJudge(BaseJudge):
    """Offline stand-in for :class:`JudgeClient` (no network).

    Accepts either:

    * a single ``responder`` callable ``fn(question) -> str-or-dict`` used for
      *every* call, or
    * a **queue** of responders/values via :meth:`add` - each is consumed by the
      next call (FIFO), letting tests script a per-request response (the way a
      real judge answers one rubric at a time). A queued entry may be a
      ``str``/``dict`` (used directly) or a ``callable(question) -> ...``.

    Records every ``question`` in :attr:`calls`. If the queue is exhausted and
    no ``responder`` is set, returns ``{}``.
    """

    def __init__(self, responder: Optional[Callable[[str], Any]] = None):
        self._queue: List[Any] = []
        self._responder = responder
        self.calls: List[str] = []

    def add(self, result: Any) -> "MockJudge":
        self._queue.append(result)
        return self

    def complete(self, question: str, *, max_tokens: int = 1024, retries: int = 2) -> str:
        self.calls.append(question)
        if self._queue:
            nxt = self._queue.pop(0)
            if callable(nxt):
                out = nxt(question)
            else:
                out = nxt
            return out if isinstance(out, str) else json.dumps(out)
        if self._responder:
            out = self._responder(question)
            return out if isinstance(out, str) else json.dumps(out)
        return "{}"


def queuing_judge(script: Optional[Sequence[Any]] = None,
                  fallback: Optional[Callable[[str], Any]] = None) -> MockJudge:
    """Build a per-call offline judge.

    ``script`` is a sequence of pre-scripted responses (str/dict/callable), one
    consumed per judge call in order. ``fallback`` answers once the script is
    exhausted - by default the permissive shape-aware responder from
    :mod:`.metrics._common` - so long multi-step metrics (decompose-then-judge)
    keep running.
    """
    if fallback is None:
        from .metrics._common import _offline_responder

        fallback = _offline_responder
    j = MockJudge(responder=fallback)
    for item in script or []:
        j.add(item)
    return j


# Process-wide default judge (lazy).
_DEFAULT_JUDGE: Optional[BaseJudge] = None


def get_default_judge() -> BaseJudge:
    """Return the module-level default judge, or raise if none is configured."""
    global _DEFAULT_JUDGE
    if _DEFAULT_JUDGE is None:
        j = from_env()
        if j is None:
            raise JudgeError(
                "No default judge configured. Set FORJINN_JUDGE_CHATFLOW to a Forjinn "
                "judge chatflow, or pass judge= to the metric, or call set_default_judge()."
            )
        _DEFAULT_JUDGE = j
    return _DEFAULT_JUDGE


def set_default_judge(client: BaseJudge) -> None:
    global _DEFAULT_JUDGE
    _DEFAULT_JUDGE = client


def _judge_or_default(judge: Optional[BaseJudge]) -> BaseJudge:
    return judge if judge is not None else get_default_judge()
