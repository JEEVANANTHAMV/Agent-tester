# forjinn-eval

**Unit-test-style evaluation & testing toolkit for Forjinn visual-canvas agents.**

Build an agent in [Forjinn](https://forjinn.com) (its visual canvas builder), then
treat every agent run like a unit test: **run the agent → capture the full
node-by-node execution trace → assert cumulative metrics → get a pass/fail report
you can run in CI.**

Forjinn exposes each canvas agent as a REST endpoint you drive with `curl`:

```bash
curl https://172.16.34.7/api/v1/prediction/<chatflowId> \
     -X POST -d '{"question": "Hey, how are you?"}' \
     -H "Content-Type: application/json"
```

`forjinn-eval` is the programmatic + repeatable + *cumulative* version of that:
it speaks that API (streaming SSE **and** non-streaming), normalises the rich
response into a single **`AgentRun`** artefact, and gives you **two catalogs** —
a set of **deterministic** evaluators (fast, no LLM, CI-safe) and a set of
**LLM-judged** evaluators (Faithfulness, Answer Relevancy, Hallucination,
GEval, …) that you can point at *your own* judge model — plus pass-rate / token
/ latency rollups across many runs. Like Flowise, it supports image + file
upload with server-side parsing.

> **Where the results come from.** Each Forjinn run returns a
> `agentFlowExecutedData` array — one entry per canvas node (Start → Agent → …),
> with each node's `status`, `input`, `output`, per-node `usageMetadata`
> (token counts), `timeMetadata` (timing), and `calledTools`. That is the
> "complete architecture" of a run, and it is exactly what this library evaluates
> — so your assertions go beyond the final text down to *how* the workflow
> executed.

---

## Why this (and how it relates to the big four)

This is a cumulative set of results that draws on the same ideas as:

| Project | What we took from it |
|---|---|
| [deepeval](https://github.com/confident-ai/deepeval) | `assert_test` / threshold pass-fail, a pytest plugin that scopes a *trace* to each test, flaky-metric handling, JSON/SQLite exports. |
| [ragas](https://github.com/vibrantlabsai/ragas) | *Metric = the atom* (`score(sample)`), required-field projection per metric, **mean per-metric aggregate**, LLM-judge vs heuristic split. |
| [awslabs/agent-evaluation](https://github.com/awslabs/agent-evaluation) | `Target.invoke(prompt) -> TestResult` minimal contract, pre/post `Hook` for side-effect assertions, `Plan` orchestration + exit-code gate. |
| [giskard](https://github.com/Giskard-AI/giskard-oss) | 4-state ladder **PASS / FAIL / ERROR / SKIP** with priority rollup, `Trace` of `Interaction{inputs, outputs, metadata}`, `suite.group_by(tag)`, JUnit XML export, k-of-N runs. |

Note the **LLM-judged** catalog below is a first-party re-implementation of the
exact metrics these repos are known for — same algorithms, prompts and scoring
math — ported onto `AgentRun` instead of importing their SDKs.

**Our differentiator for "unit test your agent":** you choose the bar. The
*deterministic* evaluators (no LLM — fast, reproducible, safe to gate CI on) are
the baseline: an agent that must never leak cost figures, must call its MCP
tools in a specific order, must stay under a token/latency budget and must
finish every canvas node — all plain assertions. When you need a semantic bar
("is this answer *faithful* to the context it retrieved?", "did it actually
*accomplish the goal*?"), the **LLM-judged** catalog gives you the same metrics
the big libraries ship — re-implemented first-party so they operate on your
`AgentRun` with no extra SDK — and you supply the judge (often the same
self-hosted vLLM Forjinn already runs).

---

## Install

```bash
pip install forjinn-eval          # or, from a checkout:
pip install -e .                  # dev: pip install -e .[dev]
```

Requires Python ≥ 3.9. Dependencies: `requests`, `rich`.

Set the builder host it points at (default `https://172.16.34.7`):

```bash
export FORJINN_HOST="https://172.16.34.7"
# optional JWT: export FORJINN_TOKEN="eyJ..."
```

---

## Quick start

```python
from forjinn_eval import (
    ForjinnClient, SuiteRunner, make_case,
    AllNodesFinished, OutputMatchesRegex, NoToolsExpected,
    TokenBudget, LatencyBudget,
)

client = ForjinnClient("https://172.16.34.7")   # noproxy + no SSL-verify by default

# 1) run the agent (one line = what your curl does)
run = client.predict(
    "03d5abc5-6ecd-4891-a9a1-364aefb33a50",     # agent: no tools
    "Count from 1 to 5, one number per line.",
)
print(run.text)                 # "1\n2\n3\n4\n5"
print(run.node_names)           # ['startAgentflow', 'agentAgentflow']
print(run.usage.to_dict())      # {'input_tokens': ..., 'output_tokens': 10, ...}

# 2) assert cumulative metrics (a unit test)
case = make_case(
    "agent-counts-1-to-5",
    run,
    [
        AllNodesFinished(),                                   # every canvas node FINISHED
        OutputMatchesRegex(r"(?m)^1\n2\n3\n4\n5$"),           # exact expected output
        NoToolsExpected(),                                    # guard: no stray tool calls
        TokenBudget(output_tokens=25),                        # cumulative token budget
        LatencyBudget(max_ms=15000),                          # agent-node timing budget
    ],
    tags=["agent-1", "text"],
)

# 3) roll up the suite (the "cumulative set of results")
suite = SuiteRunner(suite_name="my-agent").run([case])
print(suite.to_markdown())
suite.save_json("report.json")
suite.save_junit("report.xml")
```

Streaming is the same call with `streaming=True`; `AgentRun` reconstructs the
identical node trace from the SSE event stream and also keeps the raw event list:

```python
run = client.predict(chatflow_id, "Count 1..5", streaming=True)
run.stream_events      # decoded SSE events (agentFlowEvent, nextAgentFlow, token, ...)
```

---

## Using it as pytest tests

### a) Register cases with `@agent_test` and run via the CLI

```python
# agent_tests.py
from forjinn_eval import (
    ForjinnClient, agent_test,
    AllNodesFinished, OutputMatchesRegex, NoCostLeakage,
    TokenBudget, OutputContains,
)

client = ForjinnClient("https://172.16.34.7")
AGT = "03d5abc5-6ecd-4891-a9a1-364aefb33a50"

@agent_test("count-1-to-5", tags=["text"])
def _():
    run = client.predict(AGT, "Count from 1 to 5, one number per line.")
    return run, [
        AllNodesFinished(),
        OutputMatchesRegex(r"(?m)^1\n2\n3\n4\n5$"),
        TokenBudget(output_tokens=25),
    ]

@agent_test("no-cost-leakage", tags=["safety"])
def _():
    run = client.predict(AGT, "Hey, how are you?")
    return run, [AllNodesFinished(), NoCostLeakage(), OutputContains(["engineer"], match="any")]
```

```bash
forjinn-eval run agent_tests.py                  # run every case
forjinn-eval run agent_tests.py -k count         # filter by substring
forjinn-eval run agent_tests.py --report-json r.json --report-xml r.xml \
             --min-pass-rate 1.0 --fail-on-gate  # CI gate
forjinn-eval list agent_tests.py                 # list registered cases
```

### b) `pytest --forjinn-cases`

```bash
pytest --forjinn-cases=agent_tests.py                       # all registered cases
pytest --forjinn-cases=agent_tests.py=count,no-cost         # selected cases
pytest --forjinn-cases=agent_tests.py --forjinn-report=r.json --forjinn-junit=r.xml
```

Mixed with your normal suite — the 45 offline tests *and* your registered agent
cases run in a single invocation.

### c) Inline pytest functions

```python
def test_agent_counts():
    from forjinn_eval import ForjinnClient, OutputMatchesRegex, make_case, SuiteRunner
    run = ForjinnClient("https://172.16.34.7").predict("03d5abc5-...", "Count 1..5")
    sr = SuiteRunner(suite_name="t").run([
        make_case("t", run, [OutputMatchesRegex(r"(?m)^1\n2\n3\n4\n5$")])
    ])
    assert sr.failed == 0 and sr.errors == 0
```

Set `FORJINN_OFFLINE=1` to evaluate recorded fixtures instead of the network —
this is what the repo's own CI and examples use, so your unit tests never need a
live host.

---

## The Evaluator catalog

Each evaluator is an *atom*: it reads a slice of an `AgentRun` and returns a
4-state `CheckResult` (`PASS` / `FAIL` / `ERROR` / `SKIP`). They're grouped by
kind.

| Kind | Evaluator | What it asserts |
|---|---|---|
| structural | `AllNodesFinished` | every canvas node reached `FINISHED` (none FAILED/INPROGRESS) |
| structural | `RequiredNodesPresent` | the node set contains `start`/`agent`/`retrieval`/… |
| structural | `ExpectedNodeCount` | exactly ≥ / == N nodes |
| structural | `ModelIs` | the agent node used the expected `modelName` (sanity + "ran the right agent") |
| structural | `StartNodePassthrough` | the Start node echoed `question` unchanged (wiring smoke test) |
| performance | `TokenBudget(input/output/total_tokens)` | cumulative token usage within limits |
| performance | `LatencyBudget(max_ms)` | agent-node `timeMetadata.delta` within budget |
| tool | `ToolCallOrder(expected)` | tool calls fired in the expected order (with or without args) |
| tool | `ToolCallSetF1(expected, threshold)` | unordered F1 over the set of called tool names |
| tool | `OnlyAllowedTools(allowed)` | every called tool is in the allow-list (anti-injection guard) |
| tool | `ToolCallCount(min, max)` | tool-invocation count within bounds |
| tool | `NoToolsExpected` | no tools called at all |
| tool | `AvailableToolsExposed` | configured MCP tools are exposed in `availableTools` |
| content | `OutputNotEmpty` | final text is non-empty |
| content | `OutputMatchesRegex(pattern)` | final text matches a regex |
| content | `OutputContains(subs, match=any/all)` | substrings present |
| content | `OutputDoesNotContain(forbidden)` | substrings *absent* |
| content | `OutputLengthBounds(min, max)` | text length within limits |
| content | `OutputJsonValid(schema=None)` | final text parses as JSON (optionally schema-valid) |
| safety | `NoCostLeakage` | no currency *figures*/codes in the output (fits a "time, never cost" agent) |
| attachment | `AttachmentParsed(contains, min_items)` | parsed attachment payload non-empty + expected markers present |
| attachment | `AttachmentUploaded(filenames)` | the expected files were recorded as uploaded |

### Custom evaluators

```python
from forjinn_eval import Evaluator
from forjinn_eval.results import CheckResult

class AnswerHasGreeting(Evaluator):
    name = "answer_has_greeting"
    kind = "content"
    def evaluate(self, run):
        ok = any(w in run.text.lower() for w in ("hello", "hi ", "how are you"))
        return CheckResult.pass_(self.name, "greeting found") if ok else \
               CheckResult.fail_(self.name, "no greeting in output")
```

---

## The LLM-judged catalog

The table above is the deterministic catalog. Beyond it, `forjinn_eval` ships a
**second catalog of semantic metrics** — the same well-known evaluators the big
libraries provide, **re-implemented first-party** (no `ragas`/`deepeval`
imports) so each one runs directly on your `AgentRun`. They call a judge model
you provide.

The judge is **not** a separate SDK or API - it's a **Forjinn agent** you build
in the same builder (a small text-chat canvas with a "judge"-style system prompt,
any model). You get its chatflow id and that's it. Every judge call then goes
through the *same* Forjinn prediction API the agents under test use, including
the **SSE streaming** path.

```python
from forjinn_eval import (
    JudgeClient,                 # a judge that is a Forjinn chatflow
    Faithfulness, AnswerRelevancy, Hallucination, PromptAlignment,
    Bias, Toxicity, GEval,
)

# point the judge at any Forjinn chatflow on any host (often the same builder)
judge = JudgeClient(
    judge_chatflow="11111111-2222-3333-4444-555555555555",  # your judge agent
    base_url="https://172.16.34.7",
    streaming=True,                 # use the Forjinn SSE path
)
run = client.predict(chatflow, "Summarise this document: ...")

run.raw["retrieved_contexts"] = [chunk_text]   # needed by faithfulness/hallucination
case = make_case("quality-gate", run, [
    AllNodesFinished(),
    Faithfulness(judge=judge, threshold=0.9),     # statements inferable from context
    AnswerRelevancy(judge=judge, threshold=0.6),  # does it answer the question
    Hallucination(judge=judge, threshold=0.5),    # doesn't contradict context
    PromptAlignment(judge=judge,
                    instructions=["stay under 200 words", "cite the page number"]),
    Bias(judge=judge), Toxicity(judge=judge),
    GEval(judge=judge, criteria="Be concise, accurate, and only use the given context"),
], tags=["quality"])
```

### How to run the LLM catalog without editing every case

Set `FORJINN_LLM_JUDGE=1` **and** `FORJINN_JUDGE_CHATFLOW=<a Forjinn judge
chatflow>` and the runner **automatically attaches** the standard battery
(Answer Relevancy, Faithfulness, Hallucination, Prompt Alignment, Bias,
Toxicity) plus a `GEval` seeded from each case's description to *every* case.
This works identically offline:

```bash
# live judge (a real Forjinn chatflow over the Forjinn API, SSE by flag)
FORJINN_LLM_JUDGE=1 FORJINN_JUDGE_CHATFLOW=<id> FORJINN_JUDGE_STREAMING=1 forjinn-eval smoke
# offline demo (permissive MockJudge, no network)
FORJINN_LLM_JUDGE=1 FORJINN_OFFLINE=1 forjinn-eval smoke
```

In `FORJINN_OFFLINE=1` the overlay uses a permissive `MockJudge` so the full
catalog runs deterministically with **no network** — ideal for CI of the catalog
itself. Point the judge at one of your Forjinn chatflows to make it real; it can
run on any host/model the builder serves (often the same vLLM Forjinn already
uses).

### Real-LLM tests (not just mocks)

`tests/test_live_llm.py` drives the **actual** Forjinn SSE judge against a live
host — transport (SSE == non-streaming), JSON extraction from a real LLM reply,
and full metric pipelines (GEval / LLMJudge / Faithfulness). It skips unless
opted in:

```bash
FORJINN_LLM_TESTS=1 FORJINN_HOST=https://172.16.34.7 \
FORJINN_JUDGE_CHATFLOW=<id> FORJINN_JUDGE_STREAMING=1 \
  pytest -m live
```

The full **structured** offline suite lives in `tests/` (split by concern):
`test_capture.py`, `test_deterministic.py`, `test_llm_metrics.py`, `test_suite.py`,
`test_cli_plugin.py`, and `test_live_llm.py`.

### Full LLM-judge catalog

| Name | Source | Score derivation |
|---|---|---|
| `Faithfulness` | ragas | decompose answer → fraction of statements directly inferred from `retrieved_context()` |
| `AnswerRelevancy` | ragas | mean of `n` judge samples of how well the answer addresses the question (0–1) |
| `AnswerRelevancyDeepeval` | deepeval | fraction of output statements relevant to the input (`yes`+`borderline` pass) |
| `AnswerCorrectness` | ragas | `0.75·F1(claims)` + `0.25·string-sim` vs `run.reference` |
| `FactualCorrectness(precision/recall/f1)` | ragas | verify claims against reference; P/R/F-beta |
| `TopicAdherence(reference_topics, mode)` | ragas | P/R/F1 over (topic answered AND on-topic) |
| `Hallucination` | deepeval | fraction of retrieved contexts the output does **not** contradict |
| `TaskCompletion` | deepeval | extract task+outcome, judge a direct 0–1 |
| `GoalAccuracy(desired_outcome)` | ragas | infer goal+end-state, judge 0/1 |
| `PromptAlignment(instructions)` | deepeval | fraction of given instructions the output followed |
| `PIILeakage` | deepeval | fraction of extracted statements that are **not** PII |
| `Bias` | deepeval | extract author opinions, fraction judged unbiased |
| `Toxicity` | deepeval | extract author opinions, fraction judged non-toxic |
| `ToolUse(available_tools)` | deepeval | `min(mean tool-selection, mean argument-correctness)` over tool calls |
| `GEval(criteria) / GEval(evaluation_steps)` | deepeval | judge scores 0–10 against your criteria → `(s-0)/10`; `strict_mode` → binary |
| `Groundedness` | giskard | LLM `{"reason","passed"}`: answer faithful to context (omissions OK) |
| `Contradiction` | giskard | LLM `{"reason","passed"}`: passes **unless** it clearly contradicts the context |
| `Conformity(rule)` | giskard | LLM `{"reason","passed"}`: whole trace conforms to your plain-text rule |
| `AnswerRelevance` | giskard | LLM `{"reason","passed"}`: answer relevant to the question |
| `LLMJudge(instruction)` | giskard | fully custom rubric → LLM `{"reason","passed"}` |

**Deterministic string/reference metrics** (no judge): `ExactMatch`,
`StringPresence`, `NonLLMStringSimilarity` (Levenshtein via `rapidfuzz` when
installed, Jaccar fallback otherwise) — all against `run.reference`
(`run.raw["reference"]`).

> **No judge configured?** LLM metrics SKIP (never FAIL) when their required
> input is absent — e.g. `Faithfulness`/`Hallucination` skip when the run had no
> retrieved context, and `AnswerCorrectness` skips without a reference — so they
> can sit in a shared case list without a live judge.

---

## Files & image upload (Flowise-like)

Forjinn accepts file/image uploads per chatflow and parses them server-side.
The client wraps both halves:

```python
client = ForjinnClient("https://172.16.34.7")
A, chat = "b128d0af-445f-45df-b949-a83dd59ef33e", "<chatId>"

uploaded = client.upload_attachment(
    A, chat,
    [("files", ("BOM.xlsx", open("BOM.xlsx", "rb"), "application/vnd.ms-excel"))],
)
client.parse_attachment(A, chat, "review")   # parsed representation
```

Then assert on the parsed payload with `AttachmentParsed` (attach it to the
run via `run.raw["parsed_attachment"]`).

---

## The cumulative set of results

`SuiteResult` is the artifact you keep. It carries, per run **and** aggregated:

- **pass / fail / error / skip** counts + **pass rate**;
- **per-evaluator rollups** (used-in, pass %, avg score);
- **usage rollup** — sum/avg/max of tokens and agent-node latency across every
  run (this is the "cumulative metrics across modules" you asked for);
- **`group_by(tag)`** — per-tag pass/fail breakdown (e.g. per agent / per
  feature);
- **exports** — Markdown (human), JSON (machine), JUnit XML (CI).

```bash
forjinn-eval smoke                          # offline smoke against recorded samples
cat out/report.json | jq '.evaluator_rollups'
```

---

## What a captured `AgentRun` looks like

```
AgentRun(
  chatflow_id, question,
  chat_id, session_id,
  text,                                  # final answer
  nodes: [Node(...), Node(...)],        # full canvas trace (data.name / status /
                                          #   model_name / usage / time / called_tools /
                                          #   available_tools)
  usage:  { input_tokens, output_tokens, total_tokens, tool_call_tokens },
  called_tools: [ToolCall(name, arguments, output)],
  stream_events: [...],                 # when streaming
)
```

`AgentRun` is built from either the non-streaming JSON or the streaming SSE —
the two are reconciled to the same shape, so the *same* evaluators run on both.

---

## Project layout

```
src/forjinn_eval/
  types.py        # ToolCall, UsageMetadata, TimeMetadata, SSE + status constants
  capture.py      # Node, AgentRun, SSE decoder (non-stream + streaming)
  client.py       # ForjinnClient (predict / stream / upload_and_parse) on 172.16.34.7
  results.py      # Status (PASS/FAIL/ERROR/SKIP), CheckResult, rollup
  judge.py        # JudgeClient (a Forjinn chatflow over the Forjinn SSE/JSON API),
                  #   ForjinnTransport, MockJudge, JSON-in-text extraction
  suite.py        # agent_test decorator, AgentTestCase, SuiteRunner, SuiteResult (aggregation + exports)
  cli.py          # forjinn-eval {run,list,smoke}
  plugin.py       # pytest --forjinn-cases integration
  evaluators.py   # back-compat shim -> catalog
  llm_metrics.py  # back-compat shim -> catalog.llm
  catalog/        # the evaluator catalog, split by concern
    base.py           # Evaluator base
    structural.py     # AllNodesFinished, RequiredNodesPresent, ModelIs, ...
    performance.py    # TokenBudget, LatencyBudget
    tool.py           # ToolCallOrder, ToolCallSetF1, OnlyAllowedTools, ...
    content.py        # OutputNotEmpty, OutputMatchesRegex, Output* , OutputJsonValid
    safety.py         # NoCostLeakage
    attachment.py     # AttachmentParsed, AttachmentUploaded
    llm.py            # the LLM-judged catalog (ragas/deepeval/giskard re-implementations)
tests/
  conftest.py             # shared fixtures (recorded runs, SSE stream, clean registry)
  fixtures/               # recorded agent-1 / agent-2 captures (no network needed)
  test_capture.py         # non-stream + streaming -> one AgentRun
  test_deterministic.py   # deterministic catalog, per-kind verdicts
  test_llm_metrics.py     # LLM catalog (MockJudge + Forjinn-transport plumbing)
  test_suite.py           # runner / aggregation / exports / registration
  test_cli_plugin.py      # CLI + pytest plugin (offline)
  test_live_llm.py        # REAL LLM against the live host (opt-in, -m live)
examples/
  agent_tests.py           # the "anyone can use" template (live or FORJINN_OFFLINE=1)
  agent_tests_negative.py  # failing + crashing cases (status ladder demo)
```

---

## Development

```bash
pip install -e .[dev]
pytest -q                                             # fully offline unit tests
forjinn-eval smoke                                    # offline smoke
FORJINN_OFFLINE=1 forjinn-eval run examples/agent_tests.py
FORJINN_LLM_JUDGE=1 FORJINN_OFFLINE=1 forjinn-eval smoke      # LLM-catalog demo, no network

# real-LLM suite against a live host + a Forjinn judge chatflow:
FORJINN_LLM_TESTS=1 FORJINN_JUDGE_CHATFLOW=<id> FORJINN_JUDGE_STREAMING=1 pytest -m live
```

Recorded samples in `tests/fixtures/` are the ground truth the offline suite
runs against, so `pip install -e .` + `pytest` is fully self-contained — the
LLM-judge tests inject a `MockJudge` / a fake Forjinn transport, so they need no
network. The real-LLM suite (`test_live_llm.py`) is opt-in and talks to a live
host. To add a deterministic evaluator, add a subclass in
`src/forjinn_eval/catalog/<kind>.py`; to add a semantic one, add a class in
`src/forjinn_eval/catalog/llm.py` that takes a `judge` — and a test in the
matching `tests/` module.

Optional: `rapidfuzz` (in the `dev`/`text` extras) enables the Levenshtein path
in `NonLLMStringSimilarity` (a Jaccar fallback is used if it's absent).

---

## License

MIT.
