"""Shared pytest fixtures for the forjinn_eval suite.

Lives at ``tests/`` so it applies to every subpackage (``unit`` / ``integration``
/ ``live``). Fixtures point at the recorded captures in ``tests/fixtures/``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from forjinn_eval import AgentRun  # noqa: E402

# fixtures live one level down from this file (tests/fixtures)
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture
def agent1_nonstream() -> AgentRun:
    """The plain (no-tools) agent, non-streaming capture (a greeting reply)."""
    return AgentRun.from_nonstream("agent-1", _load("agent1_nonstream.json"))


@pytest.fixture
def agent1_count() -> AgentRun:
    """The plain agent's "count 1..5" capture (deterministic numeric answer)."""
    return AgentRun.from_nonstream("agent-1", _load("agent1_count.json"))


@pytest.fixture
def agent1_node_failed() -> AgentRun:
    """A capture where an agent node FAILED (for negative structural tests)."""
    return AgentRun.from_nonstream("agent-1", _load("agent1_node_failed.json"))


@pytest.fixture
def agent2_mcp() -> AgentRun:
    """The MCP/tool agent capture (available tools, ask-for-upload reply)."""
    return AgentRun.from_nonstream("agent-2", _load("agent2_tools_nonstream.json"))


@pytest.fixture
def sse_streaming_text() -> str:
    """The recorded SSE byte-stream body from a live counting run."""
    return (FIXTURES / "agent1_streaming.sse").read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def _clean_agent_test_registry():
    """Keep the global @agent_test registry clean between tests.

    Tests that run an example module register its cases; without this the
    registry leaks into unrelated tests that call ``registered_cases()``.
    """
    from forjinn_eval.suite import _REGISTRY

    snapshot = dict(_REGISTRY)
    _REGISTRY.clear()
    yield
    _REGISTRY.clear()
    _REGISTRY.update(snapshot)
