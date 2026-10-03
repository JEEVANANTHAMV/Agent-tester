"""Offline tests for the OpenAIJudge backend (no network; injectable client)."""
from __future__ import annotations

import json

import pytest

from forjinn_eval import OpenAIJudge
from forjinn_eval.judge import JudgeError


class FakeResp:
    def __init__(self, payload, status=200, text="ok"):
        self._p = payload
        self.status_code = status
        self.text = text

    def json(self):
        return self._p


def make_client(content):
    class C:
        def __init__(self):
            self.requests = []
            self._content = content

        def post(self, url, headers=None, data=None):
            self.requests.append({"url": url, "headers": headers, "body": json.loads(data)})
            return FakeResp({"choices": [{"message": {"content": self._content}}]})

    return C()


def test_requires_api_key():
    with pytest.raises(JudgeError):
        OpenAIJudge()  # no key, no env


def test_from_env_reads_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("OPENAI_JUDGE_MODEL", "gpt-4o-mini")
    j = OpenAIJudge.from_env()
    assert j.api_key == "sk-test"
    assert j.base_url == "https://example.test/v1"
    assert j.model == "gpt-4o-mini"


def test_explicit_kwargs_override_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    j = OpenAIJudge(api_key="sk-explicit", model="my-model")
    assert j.api_key == "sk-explicit"
    assert j.model == "my-model"


def test_complete_parses_chat_completions_shape():
    client = make_client('{"score": 8, "reason": "good"}')
    j = OpenAIJudge(api_key="sk-test", base_url="https://example.test/v1", client=client)
    out = j.complete_json("rubric")
    assert out == {"score": 8, "reason": "good"}
    # request shape: Bearer auth, JSON mode, target URL
    r = client.requests[0]
    assert r["url"].endswith("/chat/completions")
    assert r["headers"]["Authorization"] == "Bearer sk-test"
    assert r["body"]["response_format"] == {"type": "json_object"}
    assert r["body"]["model"] == "gpt-4o-mini"
    assert r["body"]["messages"][-1]["role"] == "user"


def test_prose_wrapped_json_is_extracted():
    client = make_client('Sure! Here is the verdict: {"score": 7, "reason": "fine"}')
    j = OpenAIJudge(api_key="sk-test", client=client)
    assert j.complete_json("rubric") == {"score": 7, "reason": "fine"}


def test_empty_response_raises():
    client = make_client("")
    j = OpenAIJudge(api_key="sk-test", client=client, retries=0)
    with pytest.raises(JudgeError):
        j.complete("rubric")


def test_http_error_raises():
    class BadClient:
        def __init__(self):
            self.requests = []

        def post(self, url, headers=None, data=None):
            self.requests.append(url)
            return FakeResp({"error": "boom"}, status=429, text="rate limited")

    j = OpenAIJudge(api_key="sk-test", client=BadClient(), retries=0)
    with pytest.raises(JudgeError):
        j.complete("rubric")
