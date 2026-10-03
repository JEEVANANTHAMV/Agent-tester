"""llm_judged.py - the LLM-judged catalog, three ways to attach a judge.

The LLM-judged catalog gives you the *semantic* bar (Faithfulness, Answer
Relevancy, Hallucination, GEval, …). The judge is **another Forjinn agent** -
build a small text-chat chatflow with a "judge"-style system prompt that ends:

    "Respond with JSON only, exactly the requested keys."

Get its chatflow id. Then, one of three ways:

A) **Explicit** - `judge=` on the metrics you want (the example below).
B) **Overlay** - set env and the runner auto-attaches the standard battery to
   *every* case, no code change:

       export FORJINN_LLM_JUDGE=1
       export FORJINN_JUDGE_CHATFLOW="11111111-2222-3333-4444-555555555555"
       forjinn-eval run examples/llm_judged.py

C) **Offline** - run with no network and a permissive mock judge (what the CI does):

       FORJINN_OFFLINE=1 forjinn-eval run examples/llm_judged.py

Without a judge (and not offline), the LLM metrics SKIP (never FAIL), so this
module still runs and the deterministic checks still gate.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from forjinn_eval import (
    AgentRun,
    AllNodesFinished,
    AnswerRelevancy,
    Embeddings,
    Faithfulness,
    Hallucination,
    GEval,
    JudgeClient,
    OutputNotEmpty,
    TokenBudget,
    agent_test,
)

HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parent / "tests" / "fixtures"

HOST = os.environ.get("FORJINN_HOST", "https://172.16.34.7")
OFFLINE = os.environ.get("FORJINN_OFFLINE", "0") == "1"
JUDGE_CHATFLOW = os.environ.get("FORJINN_JUDGE_CHATFLOW") or os.environ.get(
    "FORJINN_JUDGE_CHATFLOW_DEFAULT", "11111111-2222-3333-4444-555555555555"
)

# Offline uses the permissive default judge; otherwise build a JudgeClient from env.
# (The suite overlay in `SuiteRunner` also picks these up automatically when
# FORJINN_LLM_JUDGE=1, but here we attach the judge explicitly for clarity.)
_judge = None
if JUDGE_CHATFLOW and not OFFLINE:
    _judge = JudgeClient(
        judge_chatflow=JUDGE_CHATFLOW,
        base_url=os.environ.get("FORJINN_JUDGE_BASE_URL", HOST),
        streaming=os.environ.get("FORJINN_JUDGE_STREAMING", "0") == "1",
    )
else:
    # No live judge: metrics SKIP (no offline set_default_judge here on purpose,
    # to show that LLM metrics degrade gracefully rather than failing).
    _judge = None


def _run() -> AgentRun:
    # A RAG-ish run: answer + a bit of retrieved context (so Faithfulness /
    # Hallucination have something to ground against).
    payload = json.loads((FIXTURES / "agent1_nonstream.json").read_text(encoding="utf-8"))
    run = AgentRun.from_nonstream("llm-judged", payload)
    run.raw["retrieved_contexts"] = [
        "The agent is a senior manufacturing process and time estimation assistant. "
        "It never quotes cost figures; it only estimates process times."
    ]
    run.raw["reference"] = "The agent is ready to assist with process and time estimation."
    return run


@agent_test("llm-quality", tags=["llm", "rag"])
def llm_quality():
    """Semantic quality bar: faithful, relevant, non-hallucinated."""
    run = _run()
    evs = [AllNodesFinished(), OutputNotEmpty(), TokenBudget(total_tokens=20000)]
    if _judge is not None:
        evs += [
            Faithfulness(judge=_judge, threshold=0.5),     # statements inferable from context
            AnswerRelevancy(judge=_judge, threshold=0.4),  # answers the question
            Hallucination(judge=_judge, threshold=0.5),    # doesn't contradict context
            GEval(judge=_judge,
                  criteria="Be concise, stay in the assistant's role, and use only the "
                           "given context. Respond 0-10."),
        ]
    return run, evs
