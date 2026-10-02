"""Backward-compat: the original single-file suite now lives, split by concern,
into the structured modules next to this file:

    test_capture.py      - data model (non-stream + streaming)
    test_deterministic.py - deterministic catalog (structural/perf/tool/content/safety/attachment)
    test_llm_metrics.py  - LLM-judge catalog (MockJudge + Forjinn-transport plumbing)
    test_suite.py        - runner / aggregation / exports / registration
    test_cli_plugin.py   - CLI + pytest plugin (offline)
    test_live_llm.py     - REAL LLM against the live host (opt-in via FORJINN_LLM_TESTS)

This file keeps only the few unit checks that were not restructured elsewhere
(client config sanity + a couple of result/rollup primitives).
"""
from __future__ import annotations

from forjinn_eval import ForjinnClient
from forjinn_eval.results import CheckResult, Status, rollup


def test_client_config_and_url():
    c = ForjinnClient("https://172.16.34.7/", token="abc")
    assert c.base_url == "https://172.16.34.7"
    assert c.cookie == "token=abc"
    assert c.noproxy is True
    assert c._url("/api/v1/prediction/xyz").startswith("https://172.16.34.7/api/v1/")


def test_status_priority_rollup():
    assert rollup([Status.PASS, Status.SKIP]) == Status.PASS
    assert rollup([Status.FAIL, Status.PASS]) == Status.FAIL
    assert rollup([Status.ERROR, Status.FAIL, Status.PASS]) == Status.ERROR
    assert rollup([]) == Status.SKIP


def test_checkresult_helpers():
    p = CheckResult.pass_("x", "ok", score=0.7)
    assert p.status == Status.PASS and p.ok and p.passed is True
    f = CheckResult.fail_("x", "bad", score=0.3)
    assert f.status == Status.FAIL and not f.ok and f.passed is False
    e = CheckResult.error_("x", "boom")
    assert e.status == Status.ERROR and e.passed is None
    s = CheckResult.skip_("x", "n/a")
    assert s.status == Status.SKIP and s.passed is None
    d = p.to_dict()
    assert d["status"] == "PASS" and d["score"] == 0.7
