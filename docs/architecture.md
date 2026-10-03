# Architecture

## The central object: `AgentRun`

Every assertion in `forjinn-eval` reads a single, read-only artefact: an
**`AgentRun`**. The library's job is to turn the rich Forjinn response (a per-node
`agentFlowExecutedData` array) into exactly one of these, from either transport.

```
                 ┌───────────────────────────────────────────────┐
  POST /api/v1/  │  AgentRun (the atom of data)                   │
  prediction/…   │                                               │
     / SSE   ──▶ │  chatflow_id, question, text                   │
     / JSON   dec│  nodes: [ Node, Node, … ]   ← the full trace   │
                 │  usage:  { input, output, total, tool_call }   │
                 │  called_tools: [ ToolCall, … ]                │
                 │  stream_events: [… ]   (when streaming)        │
                 │  raw: {…}        (retrieved_contexts,         │
                 │                   reference, parsed_attach…)  │
                 └───────────────────────────────────────────────┘
                              │ read-only
        ┌──────────┬──────────┼────────────────────────────┐
   AllNodesFinished  TokenBudget  Faithfulness  TurnFaithfulness  …
   (reads nodes)     (reads usage) (reads text+ctx) (reads turns)  every
                                                              evaluator
                                                              reads AgentRun
```

Both transports produce the same shape, so **the same evaluators run on a
streaming or a non-streaming run** — you choose transport, not metric.

### Where the data comes from

Each canvas node contributes one `Node` with:

- `status` — `FINISHED` / `FAILED` / `INPROGRESS` (the structural metrics read these),
- `model_name` — the model the agent node used,
- `input` / `output` — the node I/O (retrieval context usually surfaces here),
- `usageMetadata` — per-node token counts,
- `timeMetadata` — per-node timing (the agent node's is what `LatencyBudget` reads),
- `calledTools` / `availableTools` — the tools it called / had available.

`AgentRun` aggregates these: `usage` sums the per-node tokens; `tool_names` /
`called_tools` flatten the per-node tool calls; `retrieved_context()` pulls context
from the retrieval nodes and/or `raw["retrieved_contexts"]`; `reference` reads
`raw["reference"]` (you attach ground truth there).

### Key `AgentRun` members

| Member | Type | Meaning |
|---|---|---|
| `chatflow_id`, `question` | `str` | what was run, with what. |
| `text` | `str` | the final assistant answer. |
| `nodes` | `list[Node]` | the full node-by-node trace. |
| `node_names` / `node_names()`→list | `list[str]` | node names in order. |
| `agent_node` / `final_agent_node` | `Node\|None` | first / last agent node. |
| `usage` | `UsageMetadata` | summed token counts. `.to_dict()`. |
| `called_tools` / `all_tool_calls` / `tool_names` | `list[ToolCall]` / `list[str]` | tools that fired. |
| `available_tool_names` | `list[str]` | MCP tools exposed on the agent node. |
| `retrieved_context()` | `list[str]` | context chunks (nodes or `raw["retrieved_contexts"]`). |
| `reference` | `str\|None` | ground-truth answer, from `raw["reference"]`. |
| `reference_tool_calls` | `list[ToolCall]` | ground-truth tool calls, from `raw["reference_tool_calls"]`. |
| `system_prompt()` | `str` | the agent node's system prompt (if any). |
| `all_nodes_finished` | `bool` | every node `FINISHED`. |
| `to_dict()` | `dict` | serialisable snapshot (used in reports). |
| `stream_events` | `list[StreamEvent]` | the decoded SSE events (streaming runs). |

### Attaching ground-truth / extra facts

`raw` is a plain dict you can set **before** building a case, and metrics read it:

```python
run = client.predict(chatflow, question)
run.raw["reference"]            = "The correct answer."   # for *Correctness / AnswerAccuracy
run.raw["retrieved_contexts"]   = [chunk1, chunk2]         # for Faithfulness / Hallucination / Contextual*
run.raw["reference_tool_calls"] = ["search_db", ("get_price", {"sku": "A"})]   # for Tool*
run.raw["parsed_attachment"]    = {"text": "…"}            # for AttachmentParsed
```

## Multi-turn: `Conversation` and `Message`

A **`Conversation`** is an ordered list of **`Message`**s (roles `human` / `ai` /
`system`). Each `Message` carries `content`, `tool_calls` and per-turn
`retrieval_contexts`. Two ways to get one, both ending in a single `AgentRun` so
the whole single-turn catalog keeps working:

```python
# (a) build a transcript directly
conv = Conversation()
conv.add_human("What is our refund policy?")
conv.add_ai("Refunds are available within 30 days.")
conv.add_human("So I can return it in a month?")
conv.add_ai("Yes, within 30 days of purchase.")
run = AgentRun.from_conversation(conv)      # one AgentRun over the transcript

# (b) drive a live multi-turn conversation (agent memory carries the history)
run = client.predict_multi_turn(
    chatflow_id,
    ["What is our refund policy?", "And what about exchanges?"],
)
```

On a conversation `AgentRun`:

- `run.turns()` → the `Message` list;
- `run.context_upto(i)` → cumulative retrieved contexts up to turn `i`;
- `run.question` is the first human turn; `run.text` / `run.called_tools` are the
  **final** assistant turn.

That uniformity is the point: a `TurnFaithfulness`, `MultiTurnTopicAdherence`, etc.
reads the same `AgentRun` interface as a single-turn metric, just scoped to turns.

## Execution & rollup

```
  make_case(...) ──▶ AgentTestCase
                            │
                     SuiteRunner.run(cases[, parallel=N])
                            │ resolve_run() per case (client call *or* pre-captured run)
                            │
                     for each case: evaluate every evaluator → list[CheckResult]
                            │ roll per case → TestCaseResult (status = rollup of checks)
                            ▼
                     SuiteResult
                        ├─ counts (passed/failed/errors/skipped) + pass_rate
                        ├─ evaluator_rollups()  per-metric pass% / avg score
                        ├─ usage_rollup()       sum/avg/max tokens + latency
                        ├─ group_by(tag)        per-tag breakdown
                        └─ to_markdown()/to_dict()/to_junit_xml()/save_*()
```

- **`SuiteRunner.run`** may resolve each case's `AgentRun` lazily (a case can hold
  a `lambda` that predicts on demand — the `@agent_test` pattern) or take a
  pre-captured `AgentRun`. `parallel=N` fans out across the case pool.
- The **rollup ladder** is giskard-style: a `TestCaseResult` is `PASS` iff none of
  its checks is `FAIL`/`ERROR`; an `ERROR` (e.g. a crash, judge failure) outranks
  `FAIL`; `SKIP` is neutral. `rollup(statuses)` implements this per case and per suite.
- **`SuiteResult`** is the artifact you keep — it carries per-run *and* aggregate
  data and can be exported in three forms (see [writing-tests.md](writing-tests.md#reports-and-ci-gating)).

## The judge (LLM-judged metrics)

The judge is **another Forjinn chatflow**, driven over the *same* prediction API /
SSE path the agents under test use — not a separate SDK or API key.

```
  JudgeClient(judge_chatflow, base_url, streaming, …)
        └─ POST /api/v1/prediction/<judge_chatflow>  {"question": <rubric + data>}
                     │  (SSE or JSON, same client plumbing)
                     ▼
               extract_json(text)  → the {"verdict":…} / {"score":…} / … the
                                     rubric asked for
```

- `as_int01` coerces the verdict to a 0/1 (tolerating `{"verdict":1}` / `{"useful":[1]}` shapes).
- `queuing_judge(script=[…])` builds a `MockJudge` that answers a scripted reply
  per call (FIFO) — used by the offline catalog and by tests.
- In `FORJINN_OFFLINE=1`, `from .suite import _llm_judge_from_env` returns a
  **permissive shape-aware `MockJudge`** (`_offline_responder`), so the *entire*
  LLM catalog runs with no network and returns favourable-but-shaped verdicts.
- `set_default_judge(client)` installs a process-wide judge so metrics can be
  constructed without an explicit `judge=` (used by the overlay and tests).

## Embeddings (embedding-based metrics)

`Embeddings` resolves a backend in this order: (1) an injected
`embedder(text) -> list[float]`; (2) `sentence-transformers` (from the
`[embeddings]` extra, `EMBEDDING_MODEL` selects the model); (3) a **deterministic
hash bag-of-words** — dependency-free, offline, stable across processes. Cosine
over two of those vectors is a token-overlap similarity, which keeps
`SemanticSimilarity` / `AnswerSimilarity` / `ContextualRelevancy` *meaningful*
offline.

## Module map

```
types.py        primitives: ToolCall, UsageMetadata, TimeMetadata, Message,
                Conversation, SSE event + node status constants.
capture.py      Node, AgentRun (from_dict/from_nonstream/from_stream/
                from_conversation), SSE decoder, context/usage aggregation.
client.py       ForjinnClient (predict / predict_multi_turn / stream /
                upload_and_parse / parse_attachment).
results.py      Status, Metric, CheckResult, rollup ladder.
judge.py        JudgeClient, ForjinnTransport, MockJudge, queuing_judge,
                Embeddings, _cosine, extract_json, as_int01, default-judge accessors.
suite.py        agent_test decorator, AgentTestCase/TestCaseResult/SuiteResult/
                SuiteRunner, the LLM overlay from env.
metrics/        the full metric catalog (deterministic + LLM-judged + conversational).
catalog/        the deterministic evaluator catalog, split by concern (the
                "evaluator" surface re-exported as forjinn_eval).
web_scenarios.py the Forjinn-web scenario battery (every catalog, offline/online).
cli.py / plugin.py  the `forjinn-eval` command and the pytest plugin.
evaluators.py   back-compat shim → catalog.
llm_metrics.py  back-compat shim → metrics.
```

**Where does a new thing go?**
- A *deterministic* evaluator → `metrics/` (or `catalog/<kind>.py`), export in
  `metrics/__init__.py`, re-export in `__init__.py`, add a test in `tests/unit/`.
- An *LLM-judged* metric → a class in `metrics/` taking `judge`; same wiring.
- A *conversational* metric → `metrics/conversational.py` (or `_conversational.py`
  helpers), reading `run.turns()` / `run.context_upto()`.
- A new *report format* → a method on `SuiteResult`.
- A new *CLI subcommand* → a `cmd_*` in `cli.py` + a parser block in `main()`.
