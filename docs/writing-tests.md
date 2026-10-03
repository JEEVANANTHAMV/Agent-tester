# Writing test cases

This is the how-to for authoring agent tests. A **case** = a name + an `AgentRun`
+ a list of evaluators + tags. You write the *intent* as metrics; the runner does
the rest (run the agent, evaluate, roll up, export).

Three ways to define a case — pick by how you want to run it:

1. **`@agent_test` registered module** — best for a reusable suite the CLI and
   `pytest --forjinn-cases` both run. **Recommended.**
2. **`make_case` inline** — best for ad-hoc scripts or plain `def test_*()` pytest.
3. **`AgentTestCase` directly** — when you need full control (lazy run resolution,
   custom name/tags/references).

Start from the two runnable templates in [`examples/`](../examples/):
[`agent_tests.py`](../examples/agent_tests.py) (passing) and
[`agent_tests_negative.py`](../examples/agent_tests_negative.py) (the status ladder).

---

## 1. The `@agent_test` pattern (recommended)

A module of cases. The client is built once; each case is a zero-arg function that
returns `(run, [evaluators])`.

```python
# my_agent_tests.py
from forjinn_eval import (
    ForjinnClient, agent_test,
    AllNodesFinished, OutputMatchesRegex, NoToolsExpected,
    TokenBudget, OutputContains, NoCostLeakage,
)

HOST  = "https://172.16.34.7"
AGT   = "03d5abc5-6ecd-4891-a9a1-364aefb33a50"   # agent chatflow id
client = ForjinnClient(HOST)

@agent_test("agent-1-count-1-to-5", tags=["agent-1", "text"])
def count_1_to_5():
    run = client.predict(AGT, "Count from 1 to 5, one number per line.")
    return run, [
        AllNodesFinished(),                       # structural: workflow ran fully
        RequiredNodesPresent(["start", "agent"]), # structural: expected shape
        NoToolsExpected(),                        # tool guard
        OutputNotEmpty(),                         # content
        OutputMatchesRegex(r"(?m)^1\n2\n3\n4\n5$"),# exact expected output
        TokenBudget(output_tokens=25),            # performance
    ]

@agent_test("agent-1-no-cost-leakage", tags=["agent-1", "safety"])
def no_cost_leakage():
    run = client.predict(AGT, "Hey, how are you?")
    return run, [
        AllNodesFinished(),
        NoCostLeakage(),                          # safety: no currency figures
        OutputContains(["engineer"], match="any"),
    ]
```

`@agent_test(name, tags=[…], metadata={…})` registers the function. The `name`
must be unique; `tags` drive per-tag rollups (`group_by`); the function's
**docstring becomes the case description** (used to seed a GEval in the LLM
overlay, and available in reports). Extra keys go in `metadata`.

Run it:

```bash
forjinn-eval run my_agent_tests.py                     # all cases
forjinn-eval run my_agent_tests.py -k count            # substring filter (repeatable)
forjinn-eval list my_agent_tests.py                    # just list names
forjinn-eval run my_agent_tests.py --parallel 4        # fan out

# CI: reports + a gate
forjinn-eval run my_agent_tests.py \
     --report-json report.json --report-xml report.xml --report-md report.md \
     --min-pass-rate 1.0 --fail-on-gate
```

Or inside pytest (the package ships a pytest plugin):

```bash
pytest --forjinn-cases=my_agent_tests.py               # all registered
pytest --forjinn-cases=my_agent_tests.py=count-1,no-cost
pytest --forjinn-cases=my_agent_tests.py --forjinn-report=r.json --forjinn-junit=r.xml
```

### Designing a good case

- **One intent per case.** "counts 1..5", "never leaks cost", "uses the MCP tool in
  order" — not "does everything". Small cases fail for one obvious reason.
- **Layer the bar.** Structural first (`AllNodesFinished`, `RequiredNodesPresent`),
  then tool/content/performance, then (optionally) LLM-judged semantic. A structure
  failure short-circuits your attention before a semantic one can muddy it.
- **Use tags liberally.** `["agent-1","safety"]`, `["agent-2","tools"]`,
  `["conversation"]` — `suite.group_by("agent-1")` then slices results by concern.
- **Guard the negatives too.** If an agent must *not* call tools, assert
  `NoToolsExpected()`; if it must call exactly one, assert `ToolCallCount(min=1,max=1)`.

---

## 2. Inline `make_case` (scripts & plain pytest)

For ad-hoc runs or when you want ordinary `def test_*()` functions.

```python
from forjinn_eval import (
    ForjinnClient, SuiteRunner, make_case,
    AllNodesFinished, OutputMatchesRegex, TokenBudget,
)

def test_agent_counts_1_to_5():
    client = ForjinnClient("https://172.16.34.7")
    run = client.predict("03d5abc5-…", "Count from 1 to 5, one number per line.")
    sr = SuiteRunner(suite_name="inline").run([
        make_case(
            "count-1-to-5",
            run,
            [AllNodesFinished(), OutputMatchesRegex(r"(?m)^1\n2\n3\n4\n5$"),
             TokenBudget(output_tokens=25)],
            tags=["text"],
        )
    ])
    assert sr.failed == 0 and sr.errors == 0
```

`make_case(name, run, evaluators=None, tags=None, metadata=None)` — `run` may be an
`AgentRun` **or** a zero-arg `Callable[[], AgentRun]` (called lazily by the runner,
so a crashing agent is reported as `ERROR`, not a build crash). To set a
ground-truth answer, set `run.raw["reference"] = …` before the case (used by
`*Correctness` / `AnswerAccuracy`).

`@agent_test` is just syntactic sugar over this; the two are interchangeable.

---

## 3. Reference / ground-truth cases

Some semantics require a *reference*. Attach it to the run **before** building the
case, via `run.raw[...]`:

```python
run = client.predict(AGT, "What is the price of SKU A-1?")
run.raw["reference"] = "SKU A-1 is 42.50 USD."        # for AnswerAccuracy / *Correctness
run.raw["retrieved_contexts"] = [pricing_chunk]        # for Faithfulness / Hallucination / Contextual*
run.raw["reference_tool_calls"] = [                    # for Tool* correctness
    ("search_db", {"table": "pricing"}),
    ("get_price", {"sku": "A-1"}),
]

case = make_case(
    "price-lookup",
    run,
    [
        AllNodesFinished(),
        ToolCallOrder([("search_db", ...), ("get_price", ...)]),
        AnswerAccuracy(judge=judge, threshold=0.8),
        Faithfulness(judge=judge, threshold=0.9),
        Hallucination(judge=judge, threshold=0.5),
    ],
    tags=["rag", "tools"],
)
```

Metric → required input:

| You need… | Set on `run.raw` |
|---|---|
| a ground-truth answer | `run.raw["reference"] = str` |
| the retrieved context | `run.raw["retrieved_contexts"] = [str, …]` |
| ground-truth tool calls | `run.raw["reference_tool_calls"] = [ … ]` |
| a parsed attachment | `run.raw["parsed_attachment"] = dict/str` |

Without the input, the relevant metric **SKIPs** (never FAILs) — so you can keep
these in a shared list and they activate only once the input is present.

---

## 4. LLM-judged cases (two ways)

**Explicit** — attach a judge to the metrics you want:

```python
from forjinn_eval import JudgeClient
judge = JudgeClient(judge_chatflow="1111…", base_url=HOST, streaming=True)

[
    Faithfulness(judge=judge, threshold=0.9),
    AnswerRelevancy(judge=judge, threshold=0.6),
    GEval(judge=judge, criteria="Be concise and accurate"),
]
```

**Overlay (zero-edit)** — set the environment and the runner auto-attaches the
standard battery + a description-seeded GEval to *every* case:

```bash
export FORJINN_LLM_JUDGE=1
export FORJINN_JUDGE_CHATFLOW="11111111-2222-3333-4444-555555555555"
export FORJINN_JUDGE_STREAMING=1
forjinn-eval run my_agent_tests.py
```

Offline (CI) the overlay uses a permissive mock judge — the full LLM catalog runs
with no network:

```bash
FORJINN_LLM_JUDGE=1 FORJINN_OFFLINE=1 forjinn-eval smoke
```

> Use the **overlay** for a coarse "is this generally high quality?" bar across a
> whole suite; use **explicit** judges when a specific case needs a specific rubric
> (e.g. `GEval(criteria=<that case's rubric>)`).

---

## 5. Custom evaluators

Subclass `Evaluator` and return a `CheckResult`. Return `Skip` (not `Pass`) when
the input you need is missing.

```python
from forjinn_eval import Evaluator
from forjinn_eval.results import CheckResult

class OutputMentionsPolicy(Evaluator):
    name = "output_mentions_policy"
    kind = "content"
    def evaluate(self, run) -> CheckResult:
        if not run.text.strip():
            return CheckResult.skip_(self.name, "no output to inspect")
        ok = "policy" in run.text.lower()
        return (CheckResult.pass_(self.name, "mentions policy") if ok
                else CheckResult.fail_(self.name, "does not mention policy"))
```

Then drop it into any case. A custom *LLM* metric takes `judge=` and uses
`judge.complete_json(...)` + the shared coercion helpers (`as_int01`, `_scale`) so
it behaves like the catalog. See the last section of [metrics.md](metrics.md#writing-a-custom-metric).

---

## 6. Composition gates

Bundle a group of evaluators into one logical gate:

```python
from forjinn_eval import AllOf, AnyOf, Not, NoCostLeakage, OutputMatchesRegex, OutputContains

case = make_case("greeting-safe", run, [
    AllOf(                                   # every child must pass
        NoCostLeakage(),
        Not(OutputMatchesRegex(r"\bUSD\b")), # must NOT mention a currency
    ),
    AnyOf(                                   # at least one
        OutputContains(["hi"]),
        OutputContains(["hello"]),
    ),
])
```

---

## 7. Multi-turn / conversational tests

Build a transcript (`Conversation`) or drive a live multi-turn conversation, then
use the conversational metrics.

```python
from forjinn_eval import (
    ForjinnClient, Conversation, AgentRun,
    TurnFaithfulness, TurnRelevancy, MultiTurnTopicAdherence,
    make_case, SuiteRunner,
)

# (a) drive a live agent (its memory carries the history)
run = ForjinnClient(HOST).predict_multi_turn(
    AGT,
    ["What is our refund policy?", "And what about exchanges?"],
)

# (b) or build a transcript directly, no network:
conv = Conversation()
conv.add_human("What is our refund policy?")
conv.add_ai("Refunds are available within 30 days.", tool_calls=[], retrieval_contexts=[policy_chunk])
conv.add_human("So I can return it in a month?")
conv.add_ai("Yes, within 30 days of purchase.")
run = AgentRun.from_conversation(conv)

suite = SuiteRunner(suite_name="refund-thread").run([
    make_case("refund-thread", run, [
        TurnFaithfulness(judge=judge, threshold=0.6),
        TurnRelevancy(judge=judge, threshold=0.5),
        MultiTurnTopicAdherence(reference_topics=["refunds", "exchange"], judge=judge, threshold=0.5),
    ], tags=["conversation"]),
])
print(suite.to_markdown())
```

`run.turns()`, `run.context_upto(i)`, and `run.all_nodes_finished` work the same on
a conversational `AgentRun` — the single-turn metrics keep functioning too.

---

## 8. Streaming

Streaming is a flag on `predict`; the trace reconciles to the same `AgentRun`, so
the case/evaluators are unchanged:

```python
run = client.predict(AGT, "Count 1..5", streaming=True)
# run.stream_events  -> the decoded SSE events (agentFlowEvent, nextAgentFlow, token, …)
# the same evaluators run on `run`
```

---

## 9. Files & image upload

```python
uploaded = client.upload_attachment(
    AGT, chat_id,
    files=[("files", ("BOM.xlsx", open("BOM.xlsx", "rb"), "application/vnd.ms-excel"))],
)
parsed = client.parse_attachment(AGT, chat_id, "review")   # parsed representation
run.raw["parsed_attachment"] = parsed
run.raw["uploaded_files"] = ["BOM.xlsx"]

[
    AttachmentUploaded(["BOM.xlsx"]),
    AttachmentParsed(contains=["BOM"]),
]
```

---

## 10. Reports and CI gating

A `SuiteResult` gives you counts, per-evaluator rollups, usage rollup and per-tag
grouping. Export it in three forms:

```python
suite.to_markdown()        # human table
suite.to_dict()            # JSON-able
suite.to_junit_xml()       # CI
suite.save_json("report.json"); suite.save_junit("report.xml"); suite.save_markdown("report.md")
```

Per-tag: `suite.group_by("agent-1")` → `{tag_value: {passed, failed, …}}`.
Usage: `suite.usage_rollup()` → sum/avg/max of tokens and latency.

**CI gate** — fail the build on a pass rate or on any error:

- CLI: `forjinn-eval run my_tests.py --min-pass-rate 1.0 --fail-on-gate` → exit `2`
  when the gate is missed.
- `run_registered(min_pass_rate=…)` raises `ForjinnError` when below the gate:

```python
from forjinn_eval import run_registered
suite = run_registered(min_pass_rate=1.0)    # raises below the gate
```

The packaged CI (`.github/workflows/ci.yml`) runs the offline suite + smoke +
registered cases on every push, plus opt-in live e2e and live-LLM jobs gated on
secrets. See [configuration.md](configuration.md) for the env vars it uses.

---

## 11. The status ladder (when a case is not a clean pass)

Understand the four states and how a case rolls them up:

| State of a check | Why | Rolls the case to… |
|---|---|---|
| `PASS` | assertion held. | contributes nothing. |
| `FAIL` | evaluated, not met (e.g. the regex didn't match). | case **FAIL**. |
| `ERROR` | couldn't evaluate (agent raised, judge error, bad shape). | case **ERROR**. |
| `SKIP` | input missing (no judge / no reference / no context). | neutral (doesn't fail). |

A case is `PASS` iff no check is `FAIL`/`ERROR`. So:

- An agent that *crashed* → `ERROR` (distinct from a wrong answer).
- A missing judge with `FORJINN_LLM_JUDGE` unset → LLM metrics `SKIP` → the case
  still passes on its deterministic checks.

To *demonstrate* these states, `examples/agent_tests_negative.py` has a failing
case (wrong regex → `FAIL`) and a crashing case (`ERROR`).

```bash
FORJINN_OFFLINE=1 forjinn-eval run examples/agent_tests_negative.py   # shows FAIL + ERROR + SKIP
```

This is how you build a *meaningful* report: a pass-rate number plus the precise
reason each non-pass case is non-pass.
