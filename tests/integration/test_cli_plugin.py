"""CLI + pytest-plugin scenarios (offline)."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from forjinn_eval import cli  # noqa: E402

HERE = Path(__file__).resolve().parent.parent.parent  # repo root
EXAMPLES = HERE / "examples"


def _env_offline_on():
    os.environ["FORJINN_OFFLINE"] = "1"
    os.environ["FORJINN_LLM_JUDGE"] = "1"
    os.environ.pop("FORJINN_JUDGE_CHATFLOW", None)


def _env_offline_off():
    os.environ.pop("FORJINN_OFFLINE", None)
    os.environ.pop("FORJINN_LLM_JUDGE", None)


class TestSmoke:
    def test_smoke_passes_and_writes_report(self, tmp_path, monkeypatch):
        _env_offline_on()
        try:
            import io, contextlib

            buf = io.StringIO()
            report = tmp_path / "smoke.json"
            with contextlib.redirect_stdout(buf):
                rc = cli.main(["smoke", "--report", str(report)])
            assert rc == 0
            data = json.loads(report.read_text("utf-8"))
            assert data["total"] >= 1
            assert data["failed"] == 0
            assert data["errors"] == 0
        finally:
            _env_offline_off()

    def test_smoke_without_llm_overlay_still_passes(self, tmp_path, monkeypatch):
        os.environ["FORJINN_OFFLINE"] = "1"
        os.environ.pop("FORJINN_LLM_JUDGE", None)
        os.environ.pop("FORJINN_JUDGE_CHATFLOW", None)
        try:
            report = tmp_path / "smoke2.json"
            rc = cli.main(["smoke", "--report", str(report)])
            assert rc == 0
            data = json.loads(report.read_text("utf-8"))
            assert data["failed"] == 0 and data["errors"] == 0
        finally:
            os.environ.pop("FORJINN_OFFLINE", None)


class TestListAndRun:
    def test_list_prints_registered_cases(self, tmp_path, capsys):
        _env_offline_on()
        try:
            rc = cli.main(["list", str(EXAMPLES / "agent_tests.py")])
            out = capsys.readouterr().out
            assert rc == 0
            assert "agent-1-count-1-to-5" in out
        finally:
            _env_offline_off()

    def test_run_examples_offline_with_llm(self, tmp_path, capsys):
        _env_offline_on()
        try:
            rj = tmp_path / "r.json"
            rx = tmp_path / "r.xml"
            rc = cli.main(["run", str(EXAMPLES / "agent_tests.py"),
                            "--report-json", str(rj), "--report-xml", str(rx)])
            assert rc == 0
            data = json.loads(rj.read_text("utf-8"))
            assert data["failed"] == 0 and data["errors"] == 0
            assert rx.read_text("utf-8").startswith("<?xml")
        finally:
            _env_offline_off()

    def test_run_filter_substring(self, tmp_path, capsys):
        _env_offline_on()
        try:
            out = capsys.readouterr()
            rc = cli.main(["run", str(EXAMPLES / "agent_tests.py"), "-k", "count"])
            # at least the count case ran; exit 0 because it passes
            assert rc == 0
        finally:
            _env_offline_off()

    def test_run_negative_path_exits_nonzero(self, tmp_path, capsys):
        _env_offline_on()
        try:
            rc = cli.main(["run", str(EXAMPLES / "agent_tests_negative.py")])
            assert rc == 2  # intentional FAIL/ERROR cases
        finally:
            _env_offline_off()

    def test_run_min_pass_rate_gate(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setenv("FORJINN_OFFLINE", "1")
        monkeypatch.delenv("FORJINN_LLM_JUDGE", raising=False)
        monkeypatch.delenv("FORJINN_JUDGE_CHATFLOW", raising=False)
        rc = cli.main(["run", str(EXAMPLES / "agent_tests.py"),
                       "--min-pass-rate", "1.0", "--fail-on-gate"])
        assert rc == 0  # offline examples pass 100%


class TestPytestPlugin:
    def test_forjinn_cases_option_registered(self):
        from forjinn_eval import plugin

        assert hasattr(plugin, "pytest_addoption")
        assert hasattr(plugin, "pytest_collection_finish")

    def test_forjinn_cases_runs_registered_cases_offline(self, tmp_path):
        """The pytest plugin runs the registered @agent_test cases offline.

        Run as a subprocess so the plugin's options are applied to a fresh
        pytest instance (options must be set before parse)."""
        import subprocess
        import sys

        _env_offline_on()
        try:
            env = dict(os.environ)
            env["PYTHONIOENCODING"] = "utf-8"
            proc = subprocess.run(
                [sys.executable, "-m", "pytest",
                 "--forjinn-cases", str(EXAMPLES / "agent_tests.py"),
                 "-p", "no:cacheprovider", "-q", "-x", "--co", "-q"],
                cwd=str(HERE), env=env, capture_output=True, text=True, timeout=120,
            )
            # collection lists our forjinn_case items alongside normal tests
            assert proc.returncode == 0
            assert "forjinn_case" in proc.stdout
        finally:
            _env_offline_off()
