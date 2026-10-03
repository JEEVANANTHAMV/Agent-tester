# API reference

The public surface, importable from the top level (`from forjinn_eval import …`)
unless noted. Sub-module notes are given where useful.

## Package

| Name | Notes |
|---|---|
| `__version__` | From the git tag (`setuptools-scm`); `"0.1.0.dev…"` on an untagged checkout. |

## Client

`ForjinnClient(base_url="https://172.16.34.7", token=None, cookie=None, timeout=120.0, verify_ssl=False, extra_headers=None, noproxy=True)`

| Method | Returns | Description |
|---|---|---|
| `predict(chatflow_id, question, *, streaming=False, extra_body=None)` | `AgentRun` | Run the agent (SSE or JSON). The primary entry point. |
| `predict_multi_turn(chatflow_id, turns, *, extra_body=None)` | `AgentRun` | Multi-turn; `turns` is a list of `str` or `{"role","content"}`. Human turns are separate `/prediction` calls (agent memory carries history); the final answers are stitched into one `AgentRun` via `AgentRun.from_conversation`. |
| `stream(chatflow_id, question, **kw)` | `Iterator[StreamEvent]` | Live streaming events. |
| `upload_attachment(chatflow_id, chat_id, files, …)` | `dict` | File/image upload (server-side parse). |
| `parse_attachment(chatflow_id, chat_id, kind, …)` | `dict`/`str` | The parsed representation. |

`ForjinnError` is raised on HTTP errors. See [installation.md](installation.md#proxy--corporate-network)
for proxy/verify behaviour.

## Capture

`AgentRun` — the atom of data. Build it from the client, or directly:

| Constructor | Purpose |
|---|---|
| `AgentRun.from_nonstream(chatflow_id, payload)` | From the non-streaming JSON. |
| `AgentRun.from_stream(chatflow_id, events)` | From decoded SSE events. |
| `AgentRun.from_conversation(conversation \| list)` | From a multi-turn `Conversation`/`Message` list. |

Attributes / members:

| Member | Value |
|---|---|
| `chatflow_id`, `question`, `text` | `str` — what/answer. |
| `nodes` | `list[Node]` — the full trace. |
| `node_names`, `tool_names`, `available_tool_names` | `list[str]`. |
| `agent_node`, `final_agent_node`, `agent_nodes()`, `nodes_by_type(prefix)` | node accessors. |
| `usage` | `UsageMetadata` (`.to_dict()` → `{input_tokens, output_tokens, total_tokens, tool_call_tokens}`). |
| `called_tools`, `all_tool_calls` | `list[ToolCall]`. |
| `retrieved_context()` | `list[str]` — context from nodes or `raw["retrieved_contexts"]`. |
| `reference` | `str\|None` — from `raw["reference"]`. |
| `reference_tool_calls` | `list[ToolCall]` — from `raw["reference_tool_calls"]`. |
| `system_prompt()` | `str`. |
| `all_nodes_finished` | `bool`. |
| `turns()`, `context_upto(i)`, `conversation` | multi-turn views. |
| `raw` | `dict` — attach `retrieved_contexts`, `reference`, `parsed_attachment`, `uploaded_files`, … |
| `to_dict()` | `dict` snapshot. |

`Node` (in the trace): `status`, `node_type()`, `is_agent`/`is_start`, `content`,
`model_name`, `usage`, `time`, `called_tools`, `input`, `output`.

`Message` / `Conversation`: multi-turn primitives. `Message(role, content,
tool_calls, retrieval_contexts, metadata)` with `is_human`/`is_ai`;
`Conversation(messages, metadata)` with `add_human`/`add_ai`/`append`, `turns`,
`human_turns`, `ai_turns`, `tool_calls`, `retrieval_contexts(upto)`, `from_messages`,
`from_runs`, `to_dict`.

## Results

`Status` — `PASS` / `FAIL` / `ERROR` / `SKIP`.

`CheckResult` — a single verdict: `status`, `name`, `reason`, `score`, `details`.
Constructors: `CheckResult.pass_(name, reason, **kw)`, `fail_(…, score=…)`,
`error_(…)`, `skip_(…)`; `ok` property (not FAIL/ERROR); `to_dict()`.

`rollup(statuses)` — the giskard ladder: any `ERROR` → `ERROR`, else any `FAIL` →
`FAIL`, else `PASS` (with skips).

## Judge

| Name | Notes |
|---|---|
| `JudgeClient(judge_chatflow, base_url=…, streaming=…, …)` / `.from_env()` | A judge that is a Forjinn chatflow over the Forjinn API/SSE. |
| `OpenAIJudge(api_key=…, base_url=…, model=…, …)` / `.from_env()` | A judge that is an **OpenAI-compatible** chat model (OpenAI, OpenRouter, Together, Groq, vLLM, Ollama, LM-Studio). Auto-reads `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_JUDGE_MODEL`. JSON-mode by default. |
| `MockJudge(responder=None)` | Offline stand-in. `.add(value)` queues a per-call reply (FIFO); `.calls` records questions. |
| `queuing_judge(script=None, fallback=None)` | A `MockJudge` that answers a scripted reply per call; `fallback` (default `_offline_responder`) kicks in when the script is exhausted. |
| `set_default_judge(client)` / `get_default_judge()` | Install / lazy-build a process-wide judge. |
| `Embeddings(embedder=None, model=None)` / `.from_env()` | `embed(text)`, `similarity(a,b)`, `cosine(vec_a,vec_b)`. Injected embedder → `sentence-transformers` → deterministic hash fallback. |
| `extract_json(text)` | Pull a JSON object out of a judge's prose. |
| `as_int01(x)` | Coerce a verdict to 0/1 (tolerates `{"verdict":1}`/`{"useful":[1]}`). |
| `_cosine(a, b)` | Cosine of two vectors. |

`JudgeError` is raised on judge failures.

## Suite / execution

| Name | Notes |
|---|---|
| `agent_test(name, tags=(), metadata=None)` | Decorator: registers a `@agent_test` case. The function returns `(run, [evaluators])` (or just a `run`). Its **docstring becomes the case description**. |
| `make_case(name, run, evaluators=None, tags=None, metadata=None)` | Build an `AgentTestCase`. `run` may be an `AgentRun` or a zero-arg `Callable[[], AgentRun]`. |
| `AgentTestCase` | A unit: `name`, run (or a `Callable[[], AgentRun]`), `evaluators`, `tags`, `metadata` (holds `description`). `.resolve_run()`. |
| `registered_cases()` | Names of all `@agent_test`-registered cases. |
| `build_cases(names=None)` | Resolve registered (or given) cases to runnable `AgentTestCase`s. |
| `SuiteRunner(parallel=1, suite_name="", llm_judge=None)` | `.run(cases) -> SuiteResult`. |
| `run_registered(names=None, parallel=1, suite_name="forjinn-eval", min_pass_rate=None)` | Run all registered cases; raises `ForjinnError` if under the gate. |
| `SuiteResult` | The report: `total`/`passed`/`failed`/`errors`/`skipped`/`pass_rate`, `results` (list of `TestCaseResult`), `evaluator_rollups()`, `usage_rollup()`, `group_by(tag)`, `to_markdown()`, `to_dict()`, `to_junit_xml()`, `save_json/save_junit/save_markdown`. |
| `TestCaseResult` | Per case: `name`, `tags`, `status`, `passed`, `check_results` (list of `CheckResult`), `to_dict()`. |
| `EvaluatorRollup` | Per evaluator: `used`, `passed`, `failed`, `skipped`, `pass_rate`, `avg_score`. |

## Evaluator base

`Evaluator` — base class. Subclasses set `name`/`kind` and implement
`evaluate(run) -> CheckResult`. (Re-exported from `forjinn_eval.catalog.base`.)

## Metric catalogs

Importable from the top level (see the full tables in
[metrics.md](metrics.md)):

- **Deterministic / structural**: `AllNodesFinished`, `RequiredNodesPresent`,
  `ExpectedNodeCount`, `ModelIs`, `StartNodePassthrough`.
- **Performance**: `TokenBudget`, `LatencyBudget`.
- **Tool**: `ToolCallOrder`, `ToolCallSetF1`, `OnlyAllowedTools`, `ToolCallCount`,
  `NoToolsExpected`, `AvailableToolsExposed`, `ToolCallAccuracy`, `ToolCorrectness`,
  `ArgumentCorrectness`.
- **Content**: `OutputNotEmpty`, `OutputMatchesRegex`, `OutputContains`,
  `OutputDoesNotContain`, `OutputLengthBounds`, `OutputJsonValid`, `NoCostLeakage`.
- **Text quality**: `ExactMatch`, `StringPresence`, `NonLLMStringSimilarity`,
  `PatternMatch`, `BleuScore`, `RougeScore`, `ChrfScore`, `SemanticSimilarity`,
  `AnswerSimilarity`, `AgentLoopDetection`.
- **Attachment**: `AttachmentParsed`, `AttachmentUploaded`.
- **LLM-judged (RAG answer)**: `Faithfulness`, `AnswerRelevancy`,
  `AnswerRelevancyDeepeval`, `AnswerCorrectness`, `FactualCorrectness`,
  `AnswerAccuracy`, `NoiseSensitivity`, `TopicAdherence`, `Hallucination`,
  `Groundedness`, `Contradiction`.
- **LLM-judged (retrieval/citations)**: `ContextualPrecision`, `ContextualRecall`,
  `ContextualRelevancy`, `ContextEntityRecall`, `CitationFaithfulness`,
  `QuotedSpansAlignment`.
- **LLM-judged (safety/behaviour)**: `Bias`, `Toxicity`, `PIILeakage`, `Misuse`,
  `NonAdvice`, `RoleViolation`, `RoleAdherence`, `Conformity`, `AnswerRelevance`.
- **LLM-judged (agent/plan/task)**: `TaskCompletion`, `GoalAccuracy`,
  `PromptAlignment`, `PlanAdherence`, `PlanQuality`, `StepEfficiency`,
  `Summarization`, `ToolUse`.
- **LLM-judged (generic)**: `GEval`, `LLMJudge`.
- **Conversational**: `TurnFaithfulness`, `TurnRelevancy`, `TurnContextualPrecision`,
  `TurnContextualRecall`, `TurnContextualRelevancy`, `MultiTurnTopicAdherence`,
  `MultiTurnToolUse`, `ConversationalGEval`, `GoalAccuracyMulti`,
  `ConversationCompleteness`, `KnowledgeRetention`.
- **Composition**: `AllOf`, `AnyOf`, `Not`.

### Batteries

| Function | Returns |
|---|---|
| `default_judge_metrics(threshold, judge)` | the standard single-turn LLM set. |
| `rag_judge_metrics(threshold, judge, embeddings)` | answer + retriever quality. |
| `safety_judge_metrics(threshold, judge)` | Bias/Toxicity/PII/Misuse/NonAdvice. |
| `agent_judge_metrics(threshold, judge)` | task/plan/loop. |

## Back-compat & shims

- `forjinn_eval.evaluators` → `forjinn_eval.catalog` (deterministic catalog).
- `forjinn_eval.llm_metrics` → `forjinn_eval.metrics` (LLM catalog).
- `forjinn_eval.catalog` / `forjinn_eval.metrics` are re-exported directly.

## CLI / plugin (console)

`forjinn-eval run|list|smoke|web-scenarios …` and `python -m forjinn_eval …` —
see the [CLI section of the README](../README.md#the-cli). The pytest plugin adds
`--forjinn-cases`, `--forjinn-report`, `--forjinn-junit`.

## Errors

| Exception | Raised when |
|---|---|
| `ForjinnError` | an HTTP call to the builder fails (and by `run_registered` on a missed gate). |
| `JudgeError` | a judge call / parse fails. |
