# Metrics catalog

Every metric is an **atom**: it reads a slice of an `AgentRun` and returns a
4-state `CheckResult`.

| Status | Meaning | Counts as failure? |
|---|---|---|
| `PASS` | the assertion held. | no |
| `FAIL` | the assertion was evaluated and did not hold. | **yes** |
| `ERROR` | the metric couldn't be evaluated (crash, judge error, bad input). | **yes** |
| `SKIP` | the metric's required input was absent (no judge, no context, no reference). | no (neutral) |

A case is `PASS` when none of its checks is `FAIL` or `ERROR`. `SKIP` is the
library's way of saying *"this metric needs something the run didn't have"* — it
never silently passes, and it keeps a shared case list runnable with or without a
live judge.

Metrics come in four families:

1. **Deterministic / structural** — no LLM, fast, reproducible. The CI baseline.
2. **Deterministic / content + text quality** — no LLM, compare to a reference.
3. **LLM-judged** — semantic, point at *your* judge chatflow.
4. **Conversational** — multi-turn, scoped to turns of a `Conversation`.

All of the below import from the top level: `from forjinn_eval import …`.

---

## 1. Structural (canvas shape)

Assert *how the workflow executed*, not just what it said.

| Metric | Asserts |
|---|---|
| `AllNodesFinished()` | every canvas node reached `FINISHED` (none `FAILED`/`INPROGRESS`). |
| `RequiredNodesPresent(["start","agent"])` | the node set contains the expected roles. |
| `ExpectedNodeCount(5, minimum=True)` | node count ≥ / == `n`. |
| `ModelIs("gpt-4o-mini")` | the agent node used the expected model. |
| `StartNodePassthrough()` | the Start node echoed `question` unchanged (wiring smoke test). |

When to use: the cheapest correctness gate. A "did the workflow even run the
way I designed it" check before anything semantic.

```python
case = make_case(name, run, [
    AllNodesFinished(),
    RequiredNodesPresent(["start", "agent"]),
    ModelIs("gpt-4o-mini"),
])
```

## 2. Performance (token / latency budgets)

Cumulative budgets across the whole run.

| Metric | Asserts |
|---|---|
| `TokenBudget(input_tokens=…, output_tokens=…, total_tokens=…, tool_call_tokens=…)` | summed usage within the given limits (each optional). |
| `LatencyBudget(max_ms=…)` | the agent node's `timeMetadata.delta` within budget. |

```python
[TokenBudget(output_tokens=25, total_tokens=20000), LatencyBudget(max_ms=15000)]
```

## 3. Tool-correctness

| Metric | Asserts |
|---|---|
| `ToolCallOrder(["search_db","get_price"])` | tools fired in that order (args ignored) — or `(("name",args),…)` to match args. |
| `ToolCallSetF1({"search_db"}, threshold=1.0)` | unordered F1 over the set of called tool names. |
| `OnlyAllowedTools({"search_db","get_price"})` | every called tool is in the allow-list (anti-injection). |
| `ToolCallCount(min=1, max=3)` | tool-invocation count within bounds. |
| `NoToolsExpected()` | no tools called at all. |
| `AvailableToolsExposed({"search_db"})` | configured MCP tools are exposed on the agent node. |
| `ToolCallAccuracy(reference)` | deterministic accuracy vs `reference` / `run.raw["reference_tool_calls"]`. |
| `ToolCorrectness(reference)` | LLM judge: were the right tools called for the task. |
| `ArgumentCorrectness(reference)` | LLM judge: are the tool arguments correct. |

```python
[ToolCallOrder([("search_db", {"table": "pricing"}), ("get_price", {"sku": "A-1"})])]
[OnlyAllowedTools({"search_db", "get_price"})]
```

## 4. Content

| Metric | Asserts |
|---|---|
| `OutputNotEmpty()` | final text is non-empty. |
| `OutputMatchesRegex(r"(?m)^1\n2\n3\n4\n5$")` | final text matches a regex. |
| `OutputContains(["ok"], match="any")` | substrings present (`"any"` / `"all"`). |
| `OutputDoesNotContain(["sorry i can't"])` | substrings absent. |
| `OutputLengthBounds(min=10, max=500)` | text length within limits. |
| `OutputJsonValid()` / `OutputJsonValid(schema)` | final text parses as JSON (optionally schema-valid). |
| `NoCostLeakage()` | no currency figures / codes in the output (fits a "time, never cost" agent). |

## 5. Text quality (against a reference)

No LLM. Set `run.raw["reference"] = "the correct answer"` first.

| Metric | Algorithm |
|---|---|
| `ExactMatch()` | exact string match (normalised). |
| `StringPresence(needle)` | `needle` is a substring of the answer. |
| `NonLLMStringSimilarity(threshold=0.8)` | Levenshtein similarity (via `rapidfuzz` when installed; Jaccard fallback). |
| `PatternMatch(regex, reference)` | a reference-derived pattern must match the answer. |
| `BleuScore(n=4, threshold=…)` | BLEU against the reference. |
| `RougeScore(n=2, threshold=…)` | ROUGE against the reference. |
| `ChrfScore(threshold=…)` | chrF against the reference. |
| `SemanticSimilarity(embeddings, threshold)` | cosine over `Embeddings` (offline fallback). |
| `AnswerSimilarity(embeddings, threshold)` | semantic similarity of answer vs reference. |
| `AgentLoopDetection(threshold)` | the agent did not get stuck repeating the same output. |

## 6. Attachments

| Metric | Asserts |
|---|---|
| `AttachmentParsed(contains=[…], min_items=1)` | the parsed upload payload (from `run.raw["parsed_attachment"]`) is non-empty and has the expected markers. |
| `AttachmentUploaded(["BOM.xlsx","manual.pdf"])` | the expected files were recorded as uploaded. |

```python
run.raw["parsed_attachment"] = {"text": "BOM: 42 rows"}
[AttachmentParsed(contains=["BOM"]), AttachmentUploaded(["BOM.xlsx"])]
```

---

## 7. LLM-judged (semantic)

Every metric here takes `judge=` (a `JudgeClient`, `MockJudge`, or any object
with `complete_json`). Without a judge, or without the run's required input, the
metric **SKIPs** rather than fails.

### 7.1 RAG — answer side

| Metric | Source | Reads | Score |
|---|---|---|---|
| `Faithfulness(threshold=0.9)` | ragas | `text` + `retrieved_context()` | fraction of answer statements inferred from context. |
| `AnswerRelevancy(threshold=0.6)` | ragas | `question` + `text` | mean of `n` judge samples (0–1). |
| `AnswerRelevancyDeepeval(threshold=…)` | deepeval | `question` + `text` | relevant output statements / total. |
| `AnswerCorrectness(threshold=…)` | ragas | `text` + `reference` | `0.75·F1(claims) + 0.25·string-sim`. |
| `FactualCorrectness(beta=1.0)` | ragas | `text` + `reference` | verify claims; P/R/F-beta. |
| `AnswerAccuracy(threshold=…)` | ragas | `text` + `reference` | claim-level accuracy. |
| `NoiseSensitivity(threshold=…)` | ragas | `text` + context with noise | robustness to irrelevant chunks. |
| `TopicAdherence(reference_topics, mode="f1")` | ragas | `text` + topics | P/R/F1 over (topic answered AND on-topic). |
| `Hallucination(threshold=0.5)` | deepeval | `text` + context | context chunks *not* contradicted. |
| `Groundedness` | giskard | `text` + context | `{"reason","passed"}` faithful to context. |
| `Contradiction` | giskard | `text` + context | passes **unless** it contradicts context. |

### 7.2 RAG — retrieval / citations

| Metric | Asserts |
|---|---|
| `ContextualPrecision(threshold)` | per retrieved chunk, is it the *right* rank to support the answer (ragas/deepeval). |
| `ContextualRecall(threshold)` | per reference fact, is it in the retrieved context (deepeval). |
| `ContextualRelevancy(embeddings, threshold)` | embedding similarity between context chunks and the question. |
| `ContextEntityRecall(threshold)` | fraction of reference entities that land in the context. |
| `CitationFaithfulness(threshold)` | each `[cite]` is supported by the cited context. |
| `QuotedSpansAlignment(threshold)` | quoted spans align with their source spans. |

### 7.3 Safety / behaviour

| Metric | Asserts |
|---|---|
| `Bias()` | extract author opinions; fraction judged unbiased. |
| `Toxicity()` | extract author opinions; fraction judged non-toxic. |
| `PIILeakage()` | fraction of extracted statements that are **not** PII. |
| `Misuse(threshold)` | the agent did not misuse its tools / scope. |
| `NonAdvice(threshold)` | the agent did not give unsolicited advice (fits time-only agents). |
| `RoleViolation(threshold)` | the agent stayed within its role. |
| `RoleAdherence(threshold)` | the agent stayed in character for the persona. |
| `Conformity(rule)` | the whole trace conforms to a plain-text rule. |

### 7.4 Agent / plan / task

| Metric | Asserts |
|---|---|
| `TaskCompletion(threshold)` | extract task+outcome, judge a direct 0–1. |
| `GoalAccuracy(desired_outcome, threshold)` | infer goal+end-state, judge 0/1. |
| `PromptAlignment(instructions, threshold)` | fraction of given instructions the output followed. |
| `PlanAdherence(threshold)` | scale-scored adherence to a multi-step plan. |
| `PlanQuality(threshold)` | scale-scored plan quality. |
| `StepEfficiency(threshold)` | scale-scored step efficiency. |
| `Summarization(threshold)` | summary faithful to + covers the source. |
| `ToolUse(available_tools, threshold)` | `min(mean selection, mean arg-correctness)` over tool calls. |

### 7.5 Generic

| Metric | Asserts |
|---|---|
| `GEval(criteria="…")` / `GEval(evaluation_steps=[…], strict_mode=False)` | judge scores 0–10 against your criteria → `/10`; `strict_mode` → binary. |
| `LLMJudge(instruction="…")` | fully custom rubric → `{"reason","passed"}`. |
| `AnswerRelevance()` | LLM: answer relevant to the question. |
| `Conformity(rule)` | whole trace conforms to a plain-text rule. |

### Attaching ground truth for LLM metrics

```python
run.raw["reference"]           = "ground-truth answer"     # *Correctness / AnswerAccuracy / Factual
run.raw["retrieved_contexts"]  = [c1, c2]                  # Faithfulness / Hallucination / Contextual*
run.raw["reference_tool_calls"]= ["tool", ("t2", {"k":1})] # *Correctness metrics
```

### Pointing at a judge

```python
from forjinn_eval import JudgeClient

judge = JudgeClient(
    judge_chatflow="11111111-2222-3333-4444-555555555555",  # a Forjinn judge agent
    base_url="https://172.16.34.7",                          # often the same builder
    streaming=True,                                          # use the SSE path
)
[
    Faithfulness(judge=judge, threshold=0.9),
    AnswerRelevancy(judge=judge, threshold=0.6),
    Hallucination(judge=judge, threshold=0.5),
    GEval(judge=judge, criteria="Be concise, accurate, only use the given context"),
]
```

Or use the **overlay** — set `FORJINN_LLM_JUDGE=1` + `FORJINN_JUDGE_CHATFLOW=<id>`
and the runner auto-attaches the standard battery + a description-seeded GEval to
*every* case (see [configuration.md](configuration.md)).

### Batteries

Pre-grouped lists for common gates:

```python
from forjinn_eval import default_judge_metrics, rag_judge_metrics, \
                          safety_judge_metrics, agent_judge_metrics, Embeddings

ev = default_judge_metrics(threshold=0.5, judge=judge)      # the standard LLM set
ev = rag_judge_metrics(threshold=0.5, judge=judge, embeddings=Embeddings.from_env())
ev = safety_judge_metrics(threshold=0.5, judge=judge)       # Bias/Toxicity/PII/Misuse/NonAdvice
ev = agent_judge_metrics(threshold=0.5, judge=judge)        # task/plan/loop
```

---

## 8. Conversational (multi-turn)

Scoped to the turns of a `Conversation`. Build the run via
`AgentRun.from_conversation(conv)` or `client.predict_multi_turn(...)`.

| Metric | Asserts |
|---|---|
| `TurnFaithfulness(judge, threshold)` | per-turn: the answer is faithful to the context seen up to that turn. |
| `TurnRelevancy(judge, threshold)` | per-turn: the answer addresses that turn's question. |
| `TurnContextualPrecision(judge, threshold)` | per-turn: retrieved context precision for that turn. |
| `TurnContextualRecall(judge, threshold)` | per-turn: retrieved context recall for that turn. |
| `TurnContextualRelevancy(embeddings, threshold)` | per-turn: embedding relevancy of retrieved context. |
| `MultiTurnTopicAdherence(reference_topics, judge, threshold)` | the whole conversation stays on the given topics. |
| `MultiTurnToolUse(judge, threshold)` | tools used correctly across the turns. |
| `ConversationalGEval(judge, criteria)` | GEval over the whole multi-turn transcript. |
| `GoalAccuracyMulti(judge, desired_outcome, threshold)` | the cumulative goal of the conversation was met. |
| `ConversationCompleteness(judge, threshold)` | the conversation reached a complete end-state. |
| `KnowledgeRetention(judge, threshold)` | facts stated early are remembered later. |

```python
from forjinn_eval import TurnFaithfulness, TurnRelevancy, MultiTurnTopicAdherence, \
                         Conversation, AgentRun, make_case, SuiteRunner

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

---

## 9. Composition

Combine evaluators into a single gate:

| Combinator | Semantics |
|---|---|
| `AllOf(*evaluators)` | every child must PASS. |
| `AnyOf(*evaluators)` | at least one child must PASS. |
| `Not(evaluator)` | the child must NOT pass. |

```python
from forjinn_eval import AllOf, AnyOf, Not, NoCostLeakage, OutputMatchesRegex

safe_gate = AllOf(NoCostLeakage(), Not(OutputMatchesRegex(r"\\d+\\s*USD")))
answer_gate = AnyOf(OutputContains(["ok"]), OutputContains(["done"]))
```

## 10. Writing a custom metric

```python
from forjinn_eval import Evaluator
from forjinn_eval.results import CheckResult

class OutputMentionsPolicy(Evaluator):
    name = "output_mentions_policy"     # stable, lowercase
    kind = "content"                    # structural/performance/tool/content/safety/…
    def evaluate(self, run) -> CheckResult:
        if not run.text.strip():
            return CheckResult.skip_(self.name, "no output")
        ok = "policy" in run.text.lower()
        return (CheckResult.pass_(self.name, "mentions policy") if ok
                else CheckResult.fail_(self.name, "does not mention policy"))
```

Rules of thumb:

- Return **`Skip`**, never `Pass`, when the metric's input is missing — this keeps
  shared case lists runnable without a judge / reference.
- Keep `name` stable; it is the key in reports and rollups.
- For LLM metrics, take `judge=` (and use `judge.complete_json(...)` + `as_int01` /
  `_scale` so you reuse the same coercion the catalog uses).
- Add a test in `tests/unit/` and export from `metrics/__init__.py` → `__init__.py`.
