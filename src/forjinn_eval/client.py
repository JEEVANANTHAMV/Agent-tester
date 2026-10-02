"""HTTP client for the Forjinn builder API.

The client is deliberately thin - it turns a ``predict`` / ``stream`` /
``upload_and_parse`` call into a :class:`forjinn_eval.capture.AgentRun` (or the
parsed-attachment payload) and lets the rest of the library evaluate it.

Auth model observed live at 172.16.34.7:
* ``POST /api/v1/prediction/<chatflowId>`` works with a JSON body
  ``{"question": "...", "streaming": bool}`` and a header
  ``x-request-from: internal``. A JWT ``Cookie: token=...`` is optional for
  plain prediction but is always sent if provided (``token`` or ``cookie``).
* Attachments: ``POST /api/v1/attachments/<chatflowId>/<chatId>`` - multipart
  form field ``files`` for upload; JSON ``{"chatId": ..., "question": ...}``
  for parse.

Corporate proxy: if the caller is behind a proxy that blocks the direct IP,
pass ``noproxy=True`` to disable the environment proxy for these calls
(this is what our curl smoke tests used: ``curl --noproxy '*'``).
"""
from __future__ import annotations

import json
import warnings
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

import requests

from .capture import AgentRun, Conversation, Message, iter_sse_events, parse_sse_text
from .types import ForjinnError

# Forjinn self-hosted builders use self-signed certs; verify_ssl=False is the
# norm. Silence the (correct but noisy) urllib3 warning for that case.
try:
    import urllib3

    urllib3.disable_warnings()
except Exception:  # pragma: no cover - best effort
    pass
warnings.filterwarnings("ignore", message="Unverified HTTPS request")


class ForjinnClient:
    """Minimal client for the Forjinn builder REST API."""

    def __init__(
        self,
        base_url: str = "https://172.16.34.7",
        token: Optional[str] = None,
        cookie: Optional[str] = None,
        timeout: float = 120.0,
        verify_ssl: bool = False,
        extra_headers: Optional[Dict[str, str]] = None,
        noproxy: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.cookie = cookie or (f"token={token}" if token else None)
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        self.noproxy = noproxy
        self._http = requests.Session()
        hdrs = {"Content-Type": "application/json", "x-request-from": "internal"}
        if extra_headers:
            hdrs.update(extra_headers)
        self._http.headers.update(hdrs)
        if self.cookie:
            self._http.headers["Cookie"] = self.cookie
        if noproxy:
            # Bypass any system/corporate proxy (mirrors `curl --noproxy '*'`).
            self._http.trust_env = False

    # ---- core ----------------------------------------------------------
    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def predict(
        self,
        chatflow_id: str,
        question: str,
        *,
        streaming: bool = False,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> AgentRun:
        """Run the agent (non-streaming or streaming) and return an AgentRun."""
        body: Dict[str, Any] = {"question": question, "streaming": streaming}
        if extra_body:
            body.update(extra_body)
        url = self._url(f"/api/v1/prediction/{chatflow_id}")
        resp = self._http.post(url, data=json.dumps(body), timeout=self.timeout, verify=self.verify_ssl)
        if resp.status_code >= 400:
            raise ForjinnError(f"prediction failed [{resp.status_code}]: {_short(resp.text)}")
        if streaming:
            # read raw bytes; SSE parser utf-8 decodes (avoids latin-1 mojibake)
            events = list(iter_sse_events(resp.iter_lines()))
            run = AgentRun.from_stream(chatflow_id, events)
        else:
            run = AgentRun.from_nonstream(chatflow_id, resp.json())
        return run

    def predict_multi_turn(
        self,
        chatflow_id: str,
        turns: Sequence[Union[str, Dict[str, Any]]],
        *,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> AgentRun:
        """Run a multi-turn conversation against a Forjinn agent.

        ``turns`` is an ordered list where each entry is either a bare human
        prompt string, or a dict with ``"role"`` and ``"content"``. Human turns
        are sent as separate ``/api/v1/prediction`` calls (the agent's memory
        carries the history when ``agentEnableMemory`` is on); the final
        assistant answers are stitched into a single :class:`AgentRun` via
        :meth:`AgentRun.from_conversation` so both single-turn and multi-turn
        metrics can read it.
        """
        # Collect the API prompts (str / human-dict) in order; ai-dicts are
        # transcript-only (seeded history), no API call.
        prompts: List[str] = []
        for t in turns:
            if isinstance(t, dict):
                role = str(t.get("role", "human")).lower()
                if role in {"ai", "assistant"}:
                    continue
            prompts.append(t if isinstance(t, str) else str(t.get("content", "")))

        runs: List[AgentRun] = []
        for content in prompts:
            body: Dict[str, Any] = {"question": content, "streaming": False}
            if extra_body:
                body.update(extra_body)
            url = self._url(f"/api/v1/prediction/{chatflow_id}")
            resp = self._http.post(url, data=json.dumps(body), timeout=self.timeout, verify=self.verify_ssl)
            if resp.status_code >= 400:
                raise ForjinnError(f"multi-turn prediction failed [{resp.status_code}]: {_short(resp.text)}")
            runs.append(AgentRun.from_nonstream(chatflow_id, resp.json()))

        conv = Conversation()
        r_ptr = 0
        for t in turns:
            if isinstance(t, dict) and str(t.get("role", "human")).lower() in {"ai", "assistant"}:
                conv.add_ai(str(t.get("content", "")))
            else:
                content = t if isinstance(t, str) else (str(t.get("content", "")) if isinstance(t, dict) else str(t))
                if content:
                    conv.add_human(content)
                if r_ptr < len(runs):
                    run = runs[r_ptr]; r_ptr += 1
                    conv.add_ai(run.text, run.all_tool_calls)
        conv.metadata["chatflow_id"] = chatflow_id
        conv.metadata["reference"] = extra_body.get("reference") if isinstance(extra_body, dict) else None
        return AgentRun.from_conversation(conv)

    def stream(self, chatflow_id: str, question: str, **kw: Any) -> Iterator[Any]:
        """Yield :class:`StreamEvent`s as they arrive (live)."""
        url = self._url(f"/api/v1/prediction/{chatflow_id}")
        body = {"question": question, "streaming": True, **(kw.get("extra_body") or {})}
        resp = self._http.post(
            url,
            data=json.dumps(body),
            timeout=self.timeout,
            stream=True,
            verify=self.verify_ssl,
        )
        if resp.status_code >= 400:
            raise ForjinnError(f"stream failed [{resp.status_code}]: {_short(resp.text)}")
        # decode_unicode=True forces iso-8859-1 for text/plain; Forjinn SSE is
        # actually UTF-8, so read raw bytes and let the SSE parser utf-8 decode.
        yield from iter_sse_events(resp.iter_lines())

    # ---- attachments --------------------------------------------------
    def upload_attachment(
        self,
        chatflow_id: str,
        chat_id: str,
        files: Union[tuple, List[tuple]],
        *,
        question: Optional[str] = None,
        streaming: bool = False,
    ) -> Dict[str,Any]:
        """Upload one or more files under the given chatflow/chatId.

        ``files`` is a single requests multipart tuple
        ``("files", ("name.pdf", fh, "application/pdf"))`` or a list of them.
        Returns the JSON response.
        """
        url = self._url(f"/api/v1/attachments/{chatflow_id}/{chat_id}")
        data: Dict[str, Any] = {"chatId": chat_id, "streaming": streaming}
        if question:
            data["question"] = question
        # requests sets the multipart boundary when `files` is provided.
        resp = self._http.post(
            url,
            files=files if isinstance(files, (list, tuple)) else [files],
            data=data,
            timeout=self.timeout,
            verify=self.verify_ssl,
            headers={"x-request-from": "internal", **({"Cookie": self.cookie} if self.cookie else {})},
        )
        if resp.status_code >= 400:
            raise ForjinnError(f"attachment upload failed [{resp.status_code}]: {_short(resp.text)}")
        try:
            return resp.json()
        except ValueError:
            return {"raw": resp.text}

    def parse_attachment(
        self,
        chatflow_id: str,
        chat_id: str,
        question: str,
        streaming: bool = False,
    ) -> Any:
        """Ask the builder to (re)parse an already-uploaded attachment.

        In the live capture, a fresh chatId returned ``[]`` until the file was
        actually uploaded; after upload the same call yields parsed content.
        """
        url = self._url(f"/api/v1/attachments/{chatflow_id}/{chat_id}")
        resp = self._http.post(
            url,
            data=json.dumps({"chatId": chat_id, "question": question, "streaming": streaming}),
            timeout=self.timeout,
            verify=self.verify_ssl,
        )
        if resp.status_code >= 400:
            raise ForjinnError(f"attachment parse failed [{resp.status_code}]: {_short(resp.text)}")
        try:
            return resp.json()
        except ValueError:
            return {"raw": resp.text}


def _short(s: str, n: int = 400) -> str:
    s = s or ""
    return s[:n] + ("..." if len(s) > n else "")
