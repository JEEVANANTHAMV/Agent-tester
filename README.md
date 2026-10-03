# forjinn-eval

**Unit-test-style evaluation & testing toolkit for Forjinn visual-canvas agents.**

Build an agent in [Forjinn](https://forjinn.com) (its visual canvas builder), then
treat every agent run like a unit test:

```
run the agent  →  capture the full node-by-node execution trace  →
assert cumulative metrics (deterministic + LLM-judged)  →
pass/fail report you can gate CI on (Markdown / JSON / JUnit)
```

Forjinn exposes each canvas agent as a REST endpoint you normally drive with
`curl`:

```bash
curl https://172.16.34.7/api/v1/prediction/<chatflowId> \
     -X POST -d '{"question": "Hey, how are you?"}' \
     -H "Content-Type: application/json"
```

`forjinn-eval` is the **programmatic, repeatable and cumulative** version of
that: it speaks that API (streaming SSE *and* non-streaming), normalises the rich
response into a single **`AgentRun`** artefact, and gives you **two catalogs** —

- **Deterministic evaluators** — fast, no LLM, reproducible, safe to gate CI on
  (structure, tokens, latency, tool order, content regex, safety).
- **LLM-judged evaluators** — the semantic metrics (Faithfulness, Answer
  Relevancy, Hallucination, Plan Quality, …) that point at *your own* judge
  model — usually the same self-hosted vLLM Forjinn already runs.

Plus pass-rate / token / latency rollups, per-tag grouping, and offline mode so
the whole suite runs in CI with **zero network**.

> **Where the results come from.** Each Forjinn run returns an
> `agentFlowExecutedData` array — one entry per canvas node (Start → Agent → …)
> with each node's `status`, `input`, `output`, per-node `usageMetadata`
> (token counts), `timeMetadata` (timing) and `calledTools`. That is the
> complete architecture of a run, and it is exactly what this library evaluates —
> so assertions go beyond the final text, down to *how* the workflow executed.

---

## Highlights

| Capability | What it means |
|---|---|
| **Trace-level assertions** | Evaluate every canvas node, not just the final text. |
| **Two-bar testing** | Deterministic baseline gate *plus* optional LLM-judged semantic bar. |
| **Your own judge** | The judge is a Forjinn chatflow driven over the *same* API/SSE path. |
| **Offline-first** `FORJINN_OFFLINE=1` | Replay fixtures + a permissive judge; full catalog runs with no network. |
| **Cumulative rollups** | Pass/fail/error/skip + per-evaluator + token/latency totals + per-tag. |
| **CI-native** | Markdown / JSON / JUnit exports, pass-rate gate, exit codes, a pytest plugin. |
| **Multi-turn** | `Conversation`/`Message` + `AgentRun.from_conversation` + conversational metrics. |
| **No heavy deps** | Runtime needs only `requests` + `rich`; LLM/embedding extras are optional. |

It deliberately mirrors the design of the big four — [deepeval](https://github.com/confident-ai/deepeval)
(assertion gates, pytest plugin), [ragas](https://github.com/explodinggradients/ragas)
(*metric = the atom*, per-field projection, mean aggregate),
[awslabs/agent-evaluation](https://github.com/awslabs/agent-evaluation)
(`Target.invoke → TestResult` contract, hooks, exit-code gate) and
[giskard](https://github.com/Giskard-AI/giskard-oss) (PASS/FAIL/ERROR/SKIP ladder,
trace, JUnit export) — with the LLM-judged catalog **re-implemented
first-party** (no `ragas`/`deepeval` imports) so each metric runs directly on your
`AgentRun`.

---

## Install

### From PyPI (once published)

```bash
pip install forjinn-eval          # minimal: requests + rich
pip install forjinn-eval[text]    # + Levenshtein string similarity (rapidfuzz)
pip install forjinn-eval[embeddings]  # + real semantic embeddings (sentence-transformers)
pip install forjinn-eval[all]     # everything a maintainer needs
```

### From this repository

```bash
pip install git+https://github.com/JEEVANANTHAMV/Agent-tester.git@main   # latest
pip install git+https://github.com/JEEVANANTHAMV/Agent-tester.git@v0.2.0 # a release tag
```

### Developer install (editable)

```bash
git clone https://github.com/JEEVANANTHAMV/Agent-tester.git && cd Agent-tester
pip install -e ".[all]"         # editable + dev tooling (pytest, mypy, ruff, …)
```

Requires **Python ≥ 3.9** (3.9–3.12 tested in CI).

> The version comes from the latest git tag (`setuptools-scm`); `v0.2.0` →
> `forjinn_eval.__version__ == "0.2.0"`. Untagged checkouts report a local
> `0.1.0.dev…` build.

---

## 30-second quick start

```python
from forjinn_eval import (
    ForjinnClient, SuiteRunner, make_case,
    AllNodesFinished, OutputMatchesRegex, NoToolsExpected,
    TokenBudget, LatencyBudget,
)

client = ForjinnClient("https://172.16.34.7")     # noproxy + no SSL-verify by default

# 1) run the agent (one line = what your curl does)
run = client.predict(
    "03d5abc5-6ecd-4891-a9a1-364aefb33a50",       # agent chatflow id: no tools
    "Count from 1 to 5, one number per line.",
)
print(run.text)                 # "1\n2\n3\n4\n5"
print(run.node_names)           # ['startAgentflow', 'agentAgentflow']
print(run.usage.to_dict())      # {'input_tokens': …, 'output_tokens': 10, …}

# 2) assert cumulative metrics (a single unit test)
case = make_case(
    "agent-counts-1-to-5", run,
    [
        AllNodesFinished(),                                  # every canvas node FINISHED
        OutputMatchesRegex(r"(?m)^1\n2\n3\n4\n5$"),          # exact expected output
        NoToolsExpected(),                                   # guard: no stray tool calls
        TokenBudget(output_tokens=25),                       # cumulative token budget
        LatencyBudget(max_ms=15000),                         # agent-node timing budget
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
identical node trace from the SSE event stream:

```python
run = client.predict(chatflow_id, "Count 1..5", streaming=True)
run.stream_events      # decoded SSE events (agentFlowEvent, nextAgentFlow, token, …)
```

> No live host yet? Set `FORJINN_OFFLINE=1` — the same code path replays the
> recorded fixtures in `tests/fixtures/`, so it all runs deterministically.

---

## Write your first test cases

The idiomatic pattern is **register cases with `@agent_test`, then run them via
the CLI or pytest** — the same file drives both. See a complete, runnable module
in [`examples/agent_tests.py`](https://github.com/JEEVANANTHAMV/Agent-tester/blob/master/examples/agent_tests.py).

```python
# my_agent_tests.py
from forjinn_eval import (
    ForjinnClient, agent_test,
    AllNodesFinished, OutputMatchesRegex, NoCostLeakage,
    TokenBudget, OutputContains,
)

client = ForjinnClient("https://172.16.34.7")
AGT = "03d5abc5-6ecd-4891-a9a1-364aefb33a50"

@agent_test("count-1-to-5", tags=["agent-1", "text"])
def _():
    run = client.predict(AGT, "Count from 1 to 5, one number per line.")
    return run, [
        AllNodesFinished(),
        OutputMatchesRegex(r"(?m)^1\n2\n3\n4\n5$"),
        TokenBudget(output_tokens=25),
    ]

@agent_test("no-cost-leakage", tags=["agent-1", "safety"])
def _():
    run = client.predict(AGT, "Hey, how are you?")
    return run, [AllNodesFinished(), NoCostLeakage(), OutputContains(["engineer"], match="any")]
```

Run it three ways:

```bash
forjinn-eval run my_agent_tests.py                 # run every registered case
forjinn-eval run my_agent_tests.py -k count        # filter by name substring
forjinn-eval list my_agent_tests.py                # list registered case names

# CI gate with reports
forjinn-eval run my_agent_tests.py \
     --report-json r.json --report-xml r.xml \
     --min-pass-rate 1.0 --fail-on-gate
```

Or inside pytest (the package installs a pytest plugin):

```bash
pytest --forjinn-cases=my_agent_tests.py                    # all registered cases
pytest --forjinn-cases=my_agent_tests.py=count,no-cost      # selected cases
pytest --forjinn-cases=my_agent_tests.py --forjinn-report=r.json --forjinn-junit=r.xml
```

Or inline plain pytest:

```python
def test_agent_counts():
    from forjinn_eval import ForjinnClient, OutputMatchesRegex, make_case, SuiteRunner
    run = ForjinnClient("https://172.16.34.7").predict("03d5abc5-…", "Count 1..5")
    sr = SuiteRunner(suite_name="t").run([
        make_case("t", run, [OutputMatchesRegex(r"(?m)^1\n2\n3\n4\n5$")])
    ])
    assert sr.failed == 0 and sr.errors == 0
```

A deep, worked guide with many patterns — including reference/LLM cases, custom
evaluators, composition, budgets, multi-turn and the CI gate — is in
[**docs/writing-tests.md**](https://github.com/JEEVANANTHAMV/Agent-tester/blob/master/docs/writing-tests.md).

---

## The CLI

```
forjinn-eval {run | list | smoke | web-scenarios}
```

| Command | Purpose |
|---|---|
| `run <module.py>` | Run `@agent_test` cases from a file. Options: `-k SUB` (repeatable filter), `--parallel N`, `--suite-name S`, `--report-json/--report-xml/--report-md`, `--min-pass-rate F`, `--fail-on-gate`. |
| `list <module.py>` | Print the registered case names in a module. |
| `smoke` | Offline smoke against the bundled recorded samples (no network). `--report PATH.json`. |
| `web-scenarios` | Drive the **full metric catalog** against Forjinn-web scenarios. `--offline` (default) replays fixtures; `--online --host URL --token T` hits the live builder; `--include-persona` adds a multi-turn persona conversation; same report/gate options. |

Exit codes: `0` = success (and gate met, if `--fail-on-gate`), `2` = failures/errors
or a missed pass-rate gate.

```bash
forjinn-eval smoke                                   # offline smoke, always works
forjinn-eval web-scenarios --offline                 # full catalog, offline
FORJINN_LLM_JUDGE=1 FORJINN_OFFLINE=1 forjinn-eval smoke   # LLM catalog demo (mock judge)
```

---

## Configuration (environment variables)

Everything is driven by environment variables — no config file required.

| Variable | Default | Purpose |
|---|---|---|
| `FORJINN_HOST` | `https://172.16.34.7` | Base URL of your Forjinn builder. |
| `FORJINN_TOKEN` | — | Optional JWT / API bearer token for the builder. |
| `FORJINN_OFFLINE` | `0` | `1` → replay recorded fixtures + a permissive judge. **No network.** |
| `FORJINN_LLM_JUDGE` | `0` | `1` → auto-attach the standard LLM battery (Answer Relevancy, Faithfulness, Hallucination, Prompt Alignment, Bias, Toxicity) + a description-seeded GEval to every case. |
| `FORJINN_JUDGE_CHATFLOW` | — | Chatflow id of a Forjinn judge agent (required for a *real* judge; `MockJudge` when `FORJINN_OFFLINE`). |
| `FORJINN_JUDGE_BASE_URL` | = `FORJINN_HOST` | Separate host for the judge (if it lives elsewhere). |
| `FORJINN_JUDGE_STREAMING` | `0` | `1` → drive the judge over the SSE streaming path. |
| `FORJINN_LLM_TESTS` | — | `1` → run the opt-in **real-LLM** tests (`pytest -m live`). Never on in default CI. |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Model for `Embeddings` *if* `sentence-transformers` is installed (`[embeddings]` extra). |

### Example environments

```bash
# 1) Live agent tests against a builder (no LLM judge)
export FORJINN_HOST="https://172.16.34.7"
export FORJINN_TOKEN="eyJ…"            # if the builder is authed
forjinn-eval run my_agent_tests.py

# 2) Same, but with a real LLM judge (a Forjinn chatflow that returns JSON)
export FORJINN_LLM_JUDGE=1
export FORJINN_JUDGE_CHATFLOW="11111111-2222-3333-4444-555555555555"
export FORJINN_JUDGE_STREAMING=1

# 3) Fully offline / CI (no network, permissive judge)
export FORJINN_OFFLINE=1
export FORJINN_LLM_JUDGE=1              # optional: also run the LLM catalog
pytest -q

# 4) Real-LLM suite against a live judge (opt-in, not in default CI)
FORJINN_LLM_TESTS=1 FORJINN_JUDGE_CHATFLOW=<id> FORJINN_JUDGE_STREAMING=1 pytest -m live
```

---

## The evaluator catalogs

Every evaluator is an **atom**: it reads a slice of an `AgentRun` and returns a
4-state `CheckResult` (`PASS` / `FAIL` / `ERROR` / `SKIP`). A `FAIL` is a real
assertion failure; a `SKIP` means the metric's input was absent (no judge, no
retrieved context, no reference) — never silently a pass.

### Deterministic catalog (no LLM — the CI baseline)

| Kind | Evaluator | Asserts |
|---|---|---|
| structural | `AllNodesFinished` | every canvas node reached `FINISHED` |
| structural | `RequiredNodesPresent(nodes)` | the node set contains the expected roles |
| structural | `ExpectedNodeCount(n, minimum=False)` | node count ≥ / == `n` |
| structural | `ModelIs(model)` | the agent node used the expected model |
| structural | `StartNodePassthrough` | Start node echoed `question` unchanged |
| performance | `TokenBudget(input/output/total/tool_call_tokens)` | cumulative token usage within limits |
| performance | `LatencyBudget(max_ms)` | agent-node `timeMetadata.delta` within budget |
| tool | `ToolCallOrder(expected)` | tool calls fired in the expected order |
| tool | `ToolCallSetF1(expected, threshold)` | unordered F1 over called tool names |
| tool | `OnlyAllowedTools(allowed)` | every called tool is in the allow-list (anti-injection) |
| tool | `ToolCallCount(min, max)` | tool-invocation count within bounds |
| tool | `NoToolsExpected` | no tools called at all |
| tool | `AvailableToolsExposed(tools)` | configured MCP tools are exposed |
| content | `OutputNotEmpty` | final text is non-empty |
| content | `OutputMatchesRegex(pattern)` | final text matches a regex |
| content | `OutputContains(subs, match=any/all)` | substrings present |
| content | `OutputDoesNotContain(forbidden)` | substrings absent |
| content | `OutputLengthBounds(min, max)` | text length within limits |
| content | `OutputJsonValid(schema=None)` | final text parses as JSON (optionally schema-valid) |
| text | `PatternMatch(pattern)` / `BleuScore` / `RougeScore` / `ChrfScore` | string quality vs reference |
| text | `SemanticSimilarity` / `AnswerSimilarity` / `PatternMatch` | embedding-based similarity (offline fallback) |
| loop | `AgentLoopDetection` | the agent did not get stuck repeating itself |
| tool | `ToolCallAccuracy` / `ToolCorrectness` / `ArgumentCorrectness` | deterministic tool/argument checks |
| safety | `NoCostLeakage` | no currency figures/codes in the output |
| attachment | `AttachmentParsed` / `AttachmentUploaded` | parsed upload payload / expected files |

### LLM-judged catalog (semantic bar — point at your judge)

The judge is **not** a separate SDK — it is a **Forjinn chatflow** you build in
the same builder (a small text-chat canvas with a "judge"-style system prompt,
any model). Every judge call goes through the *same* Forjinn prediction API
(including SSE streaming).

| Metric | Source | Score |
|---|---|---|
| `Faithfulness` | ragas | fraction of answer statements inferred from context |
| `AnswerRelevancy` | ragas | mean of `n` judge samples (0–1) |
| `AnswerRelevancyDeepeval` | deepeval | relevant output statements / total |
| `AnswerCorrectness` | ragas | `0.75·F1(claims) + 0.25·string-sim` vs reference |
| `FactualCorrectness` | ragas | verify claims vs reference; P/R/F-beta |
| `TopicAdherence` | ragas | P/R/F1 over (topic answered AND on-topic) |
| `AnswerAccuracy` | ragas | claim-level accuracy vs reference |
| `NoiseSensitivity` | ragas | robustness to irrelevant context chunks |
| `ContextualPrecision`/`Recall`/`Relevancy` | ragas/deepeval | per-chunk retrieval quality |
| `ContextEntityRecall` | deepeval | fraction of reference entities in context |
| `CitationFaithfulness` | — | per-citation support in context |
| `QuotedSpansAlignment` | — | quoted span ↔ source span alignment |
| `Hallucination` | deepeval | context chunks *not* contradicted by output |
| `Bias` / `Toxicity` | deepeval | fraction of opinions judged unbiased / non-toxic |
| `PIILeakage` | deepeval | fraction of statements that are not PII |
| `TaskCompletion` | deepeval | extract task+outcome, judge 0–1 |
| `GoalAccuracy(desired_outcome)` | ragas | infer goal+end-state, judge 0/1 |
| `PromptAlignment(instructions)` | deepeval | fraction of instructions followed |
| `PlanAdherence` / `PlanQuality` / `StepEfficiency` | deepeval | scale-scored plan/efficiency rubrics |
| `ConversationCompleteness` / `KnowledgeRetention` | — | multi-turn completeness & knowledge carry-over |
| `Summarization` | — | faithfulness + coverage of a summary |
| `ToolUse(available_tools)` | deepeval | `min(mean selection, mean arg-correctness)` |
| `ArgumentCorrectness` | — | arguments of called tools vs expected |
| `GEval(criteria)` / `GEval(evaluation_steps)` | deepeval | judge 0–10 → `/10`; `strict_mode` → binary |
| `Groundedness` / `Contradiction` / `Conformity` / `AnswerRelevance` | giskard | LLM `{"reason","passed"}` |
| `LLMJudge(instruction)` | giskard | fully custom rubric → LLM `{"reason","passed"}` |
| `Misuse` / `NonAdvice` / `RoleViolation` / `RoleAdherence` | — | safety & persona adherence |

> **No judge configured?** LLM metrics **SKIP** (never FAIL) when their required
> input is absent — so they can sit in a shared case list without a live judge.

### Composition

Combine evaluators into gates:

```python
from forjinn_eval import AllOf, AnyOf, Not, NoCostLeakage, OutputContains

gate_all = AllOf(NoCostLeakage(), OutputMatchesRegex(r"…"))   # every must pass
gate_any = AnyOf(OutputContains(["ok"]), OutputContains(["done"]))  # at least one
gate_not = Not(NoCostLeakage())                                # must NOT be true
```

### Custom evaluators

```python
from forjinn_eval import Evaluator
from forjinn_eval.results import CheckResult

class AnswerHasGreeting(Evaluator):
    name = "answer_has_greeting"
    kind = "content"
    def evaluate(self, run):
        ok = any(w in run.text.lower() for w in ("hello", "hi ", "how are you"))
        return (CheckResult.pass_(self.name, "greeting found") if ok
                else CheckResult.fail_(self.name, "no greeting in output"))
```

A full, annotated catalog + when-to-use guidance is in
[**docs/metrics.md**](https://github.com/JEEVANANTHAMV/Agent-tester/blob/master/docs/metrics.md).

---

## Run the LLM catalog without editing every case

Set `FORJINN_LLM_JUDGE=1` and `FORJINN_JUDGE_CHATFLOW=<judge chatflow>` and the
runner **automatically attaches** the standard battery + a `GEval` seeded from each
case's description to *every* case. Offline it uses a permissive `MockJudge`, so it
runs deterministically with no network:

```bash
# live judge (a real Forjinn chatflow over the Forjinn API, SSE by flag)
FORJINN_LLM_JUDGE=1 FORJINN_JUDGE_CHATFLOW=<id> FORJINN_JUDGE_STREAMING=1 forjinn-eval smoke
# offline demo (permissive MockJudge, no network)
FORJINN_LLM_JUDGE=1 FORJINN_OFFLINE=1 forjinn-eval smoke
```

Point the judge at one of your Forjinn chatflows to make it real — it can run on
any host/model the builder serves (often the same vLLM Forjinn already uses).

---

## Multi-turn / conversational testing

Build a transcript and run it through the single-turn catalog *or* the dedicated
conversational metrics:

```python
from forjinn_eval import (
    ForjinnClient, Conversation, Message, AgentRun,
    TurnFaithfulness, TurnRelevancy, MultiTurnTopicAdherence,
    make_case, SuiteRunner,
)

# multi-turn against a live agent (agent memory carries the history)
run = ForjinnClient(HOST).predict_multi_turn(
    AGT, ["What is our policy on refunds?", "And what about exchanges?"],
)

# or build a transcript directly (no network)
conv = Conversation()
conv.add_human("What is our refund policy?")
conv.add_ai("Refunds are available within 30 days.")
conv.add_human("So I can return it in a month?")
conv.add_ai("Yes, within 30 days of purchase.")
run = AgentRun.from_conversation(conv)

suite = SuiteRunner(suite_name="conv").run([make_case("refund-thread", run, [
    TurnFaithfulness(judge=judge, threshold=0.6),
    TurnRelevancy(judge=judge, threshold=0.5),
    MultiTurnTopicAdherence(reference_topics=["refunds"], judge=judge, threshold=0.5),
], tags=["conversation"])])
```

See [**docs/architecture.md**](https://github.com/JEEVANANTHAMV/Agent-tester/blob/master/docs/architecture.md) for the `AgentRun` /
`Conversation` / `Message` data model and [**docs/writing-tests.md**](https://github.com/JEEVANANTHAMV/Agent-tester/blob/master/docs/writing-tests.md#multi-turn--conversational-tests)
for conversational test patterns.

---

## The cumulative set of results

`SuiteResult` is the artefact you keep. It carries, per run **and** aggregated:

- **pass / fail / error / skip** counts + **pass rate**;
- **per-evaluator rollups** (used-in, pass %, avg score);
- **usage rollup** — sum/avg/max of tokens and agent-node latency across runs;
- **`group_by(tag)`** — per-tag pass/fail breakdown (e.g. per agent / feature);
- **exports** — Markdown (human), JSON (machine), JUnit XML (CI).

```python
sr.group_by("agent-1")          # per-tag breakdown
sr.to_markdown(); sr.to_dict(); sr.to_junit_xml()
sr.save_json("report.json"); sr.save_junit("report.xml")
```

---

## How to run / test the package

```bash
pip install -e ".[all]"

pytest -q                                            # fully offline unit + integration tests
forjinn-eval smoke                                   # offline smoke (recorded samples)
FORJINN_OFFLINE=1 forjinn-eval run examples/agent_tests.py
FORJINN_OFFLINE=1 forjinn-eval web-scenarios          # full catalog, offline
FORJINN_LLM_JUDGE=1 FORJINN_OFFLINE=1 forjinn-eval smoke   # LLM catalog demo, no network

# real-LLM (opt-in) — needs a live host + a Forjinn judge chatflow
FORJINN_LLM_TESTS=1 FORJINN_HOST=https://172.16.34.7 \
FORJINN_JUDGE_CHATFLOW=<id> FORJINN_JUDGE_STREAMING=1 pytest -m live
```

`tests/` is split by concern and runs **offline by default**:

```
tests/unit/          # capture, deterministic metrics, LLM metrics (MockJudge),
                     # embeddings, conversational, web-scenarios, suite/CLI/plugin
tests/integration/   # end-to-end client → case → suite (offline fixtures)
tests/live/          # REAL LLM against a live host (opt-in, -m live)
tests/fixtures/      # recorded agent captures (ground truth for offline mode)
examples/            # runnable sample test-case modules (live or FORJINN_OFFLINE=1)
```

## Documentation

| Document | Description |
|:---|:---|
| [**Quickstart**](https://github.com/JEEVANANTHAMV/Agent-tester/blob/master/docs/quickstart.md) | First assertions in 5 minutes (CLI, pytest, offline mock) |
| [**Writing Tests**](https://github.com/JEEVANANTHAMV/Agent-tester/blob/master/docs/writing-tests.md) | `@agent_test`, multi-turn, custom evaluators, fixtures |
| [**Metrics Catalog**](https://github.com/JEEVANANTHAMV/Agent-tester/blob/master/docs/metrics.md) | Complete reference of 30+ deterministic and LLM-judged metrics |
| [**Architecture**](https://github.com/JEEVANANTHAMV/Agent-tester/blob/master/docs/architecture.md) | `AgentRun`, `Conversation`, `Message`, and node execution model |
| [**Configuration**](https://github.com/JEEVANANTHAMV/Agent-tester/blob/master/docs/configuration.md) | Environment variables, CLI flags, pytest integration |
| [**API Reference**](https://github.com/JEEVANANTHAMV/Agent-tester/blob/master/docs/api-reference.md) | Public classes, methods, and entrypoints |
| [**Installation**](https://github.com/JEEVANANTHAMV/Agent-tester/blob/master/docs/installation.md) | Package installation, optional extras, and dependencies |

---

## Development

```bash
pip install -e ".[all]"
mypy            # type-check
ruff check src tests
ruff format --check src tests
pytest -q
```

To add a metric: a deterministic one goes in `src/forjinn_eval/catalog/<kind>.py`
(or `metrics/`), an LLM-judged one takes a `judge` argument. Export it from
`src/forjinn_eval/metrics/__init__.py` and re-export it in `src/forjinn_eval/__init__.py`,
then add a test in the matching `tests/unit/` module. Record fixtures under
`tests/fixtures/`.

---

## Project layout

```
src/forjinn_eval/
  types.py        # ToolCall, UsageMetadata, TimeMetadata, Message, Conversation, SSE/status constants
  capture.py      # Node, AgentRun (non-stream + streaming + from_conversation), SSE decoder
  client.py       # ForjinnClient (predict / predict_multi_turn / stream / upload_and_parse)
  results.py      # Status (PASS/FAIL/ERROR/SKIP), CheckResult, rollup
  judge.py        # JudgeClient (a Forjinn chatflow over the Forjinn API), ForjinnTransport,
                  #   MockJudge, Embeddings, queuing_judge, JSON-in-text extraction
  suite.py        # agent_test decorator, AgentTestCase, SuiteRunner, SuiteResult
  cli.py          # forjinn-eval {run,list,smoke,web-scenarios}
  plugin.py       # pytest --forjinn-cases integration
  web_scenarios.py# the Forjinn-web scenario battery (all catalogs, offline/online)
  catalog/        # deterministic evaluator catalog, split by concern
  metrics/        # the full metric catalog (deterministic + LLM-judged + conversational)
tests/            # unit (offline), integration (offline), live (opt-in), fixtures/
examples/         # runnable sample test-case modules
docs/             # installation, quickstart, architecture, metrics, writing-tests,
                  # configuration, api-reference, release
```

---

## License

MIT — see [LICENSE](https://github.com/JEEVANANTHAMV/Agent-tester/blob/master/LICENSE).
