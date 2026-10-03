"""openai_judge.py - use an OpenAI (or any OpenAI-compatible) model as the judge.

The LLM-judged metrics (Faithfulness, AnswerRelevancy, Hallucination, GEval, …)
call ``judge.complete_json(rubric)``. By default the judge is a Forjinn chatflow
(``JudgeClient``). To judge with OpenAI instead, pass an ``OpenAIJudge`` - the
same metric code runs unchanged.

RUN IT (live - needs OPENAI_API_KEY):

    export OPENAI_API_KEY="sk-…"
    python examples/openai_judge.py

RUN IT (offline - no key needed, uses a recorded agent run + a mock judge):

    FORJINN_OFFLINE=1 python examples/openai_judge.py

WHAT'S SHOWN
------------
1. Build an ``OpenAIJudge`` (auto-reads ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` /
   ``OPENAI_JUDGE_MODEL``).
2. Build an ``AgentRun`` (a recorded fixture here; swap in
   ``ForjinnClient(...).predict(...)`` for a live agent).
3. Pass the judge to the metrics you want, run the suite, and read the **scores**:
   ``run.results[i].check_results[j].score``.

For any OpenAI-compatible server (OpenRouter / Together / Groq / vLLM / Ollama /
LM-Studio), just point ``base_url`` at it:

    judge = OpenAIJudge(api_key="…", base_url="https://openrouter.ai/api/v1",
                        model="anthropic/claude-3.5-sonnet")
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from forjinn_eval import (
    AgentRun,
    AllNodesFinished,
    AnswerRelevancy,
    Faithfulness,
    Hallucination,
    GEval,
    MockJudge,
    OpenAIJudge,
    OutputNotEmpty,
    SuiteRunner,
    TokenBudget,
    make_case,
)

HERE = Path(__file__).resolve().parent
FIXTURES = HERE.parent / "tests" / "fixtures"
OFFLINE = os.environ.get("FORJINN_OFFLINE", "0") == "1"


def _agentic_judge(q: str) -> dict:
    """Mimic a real judge by answering the *actual* rubric it is given.

    A live OpenAI judge reads the question (the rendered rubric), sees how many
    list items it must produce and what verdict shape to use, and answers all of
    them favorably. This function does the same, deterministically, so the
    whole LLM-catalog path (including decompose-then-judge) runs end-to-end
    without an API key.
    """
    import re

    def n_items() -> int:
        m = re.search(r'a list of (\d+) objects', q)
        if m:
            return max(1, min(int(m.group(1)), 8))
        nums = re.findall(r"^\s*\d+\.", q, re.MULTILINE)
        return max(1, min(len(nums), 8)) if nums else 3

    def verdict_shape():
        if r'"yes"' in q or '"yes"|"no"' in q or r'"yes"' in q:
            return {"verdict": "yes", "reason": "consistent with context"}
        return {"verdict": 1, "reason": "supported"}

    n = n_items()
    import re as _re

    # the rubric's primary first-named key tells us what shape it wants
    m = _re.search(r'a single key "(\w+)"|return JSON\s*{?"?(\w+)"?\s*:', q, _re.I)
    primary = (m.group(1) or m.group(2) or "").lower() if m else ""

    # Faithfulness does a decompose-then-judge: the FIRST "statements" call wants
    # a list of bare strings (no verdict), the SECOND (NLI) wants a list of
    # {"statement", "verdict"} objects. Disambiguate by "verdict" in the rubric.
    if primary == "statements":
        if "verdict" in q:
            stmts = [{"statement": f"s{i}", "reason": "inferred", "verdict": 1} for i in range(1, n + 1)]
        else:
            stmts = [f"statement {i}" for i in range(1, n + 1)]
    else:
        stmts = []

    out = {
        "score": 9,
        "raw_score": 9,
        "score_0to10": 9,
        "reason": "ok (offline agentic judge)",
        "passed": True,
        "statements": stmts,
        "steps": ["be on-topic", "stay in role", "ground in context"],  # GEval step-generation
        "verdicts": [verdict_shape() for _ in range(n)],
        "useful": [1] * n,
        "relevant": [1] * n,
        "supported": [1] * n,
        "covered": [1] * n,
        "correct": [1] * n,
        "recalled": [1] * n,
        "in_context": 1,
        "faithful": True,
        "recall_verdicts": [{"entity": f"e{i}", "in_context": 1} for i in range(1, n + 1)],
        "claim_verdicts": [{"claim": f"c{i}", "verdict": 1, "reason": "ok"} for i in range(1, n + 1)],
        "classifications": [True] * n,
        "evaluated": True,
        "quality_verdict": "Good",
    }
    return out


def build_judge():
    """An OpenAI-compatible judge, or a permissive mock judge when offline."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if OFFLINE or not api_key:
        print(">> no OPENAI_API_KEY (or FORJINN_OFFLINE=1): using a permissive mock judge\n"
              "   (offline = demonstration; a real OpenAI judge scores these metrics live)\n")
        return MockJudge(responder=lambda q: json.dumps(_agentic_judge(q)))
    return OpenAIJudge.from_env()   # reads OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_JUDGE_MODEL


def build_run():
    """A single-turn AgentRun with a bit of retrieved context for RAG metrics."""
    payload = json.loads((FIXTURES / "agent1_nonstream.json").read_text(encoding="utf-8"))
    run = AgentRun.from_nonstream("openai-judged", payload)
    run.raw["retrieved_contexts"] = [
        "The assistant is a senior manufacturing process and time estimation specialist. "
        "It is ready to assist and it never quotes cost figures, only process times."
    ]
    run.raw["reference"] = "I am operating at full capacity and ready to assist."
    return run


def main() -> int:
    judge = build_judge()
    run = build_run()

    evaluators = [
        AllNodesFinished(),                              # deterministic (no judge)
        OutputNotEmpty(),                                # deterministic (no judge)
        TokenBudget(total_tokens=20000),                 # deterministic (no judge)
        Faithfulness(judge=judge, threshold=0.5),        # LLM: faithful to context
        AnswerRelevancy(judge=judge, threshold=0.4),     # LLM: answers the question
        Hallucination(judge=judge, threshold=0.5),       # LLM: no contradiction of context
        GEval(judge=judge,                              # LLM: your own rubric
              criteria="Rate 0-10 how well the answer stays in its assistant role.",
              threshold=0.7),
    ]

    case = make_case(
        "openai-judged-case",
        run,
        evaluators,
        tags=["openai"],
        metadata={"description": "judged by an OpenAI-compatible model"},
    )
    suite = SuiteRunner(suite_name="openai-judge").run([case])

    print("=== per-metric verdicts + scores ===")
    for res in suite.results:
        for cr in res.check_results:
            s = f"{cr.score:.3f}" if isinstance(cr.score, (int, float)) and cr.score == cr.score else "-"
            print(f"  {cr.status.value:5}  {cr.name:20}  score={s:6}  {cr.reason[:70]}")
    print(f"\n=== suite: pass_rate={suite.pass_rate:.1%}  "
          f"passed={suite.passed} failed={suite.failed} errors={suite.errors} skipped={suite.skipped} ===")
    print(suite.to_markdown())
    return 0


# ---- the canonical one-liner to pass an OpenAI judge ---------------------------
#   from forjinn_eval import OpenAIJudge, Faithfulness
#   judge = OpenAIJudge(api_key="sk-…", model="gpt-4o-mini")
#   score = Faithfulness(judge=judge, threshold=0.5).evaluate(run).score  # 0.0..1.0

if __name__ == "__main__":
    sys.exit(main())
