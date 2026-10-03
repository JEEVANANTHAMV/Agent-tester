# Quickstart

The whole loop is: **run the agent → build a case → run the suite → read the report**.
One line each.

```python
from forjinn_eval import (
    ForjinnClient, SuiteRunner, make_case,
    AllNodesFinished, OutputMatchesRegex, NoToolsExpected,
    TokenBudget, LatencyBudget,
)

client = ForjinnClient("https://172.16.34.7")        # host of your builder

# 1) run the agent — the same thing your curl does
run = client.predict(
    "03d5abc5-6ecd-4891-a9a1-364aefb33a50",          # the agent's chatflow id
    "Count from 1 to 5, one number per line.",
)
print(run.text)                                      # '1\n2\n3\n4\n5'
print(run.node_names)                                # ['startAgentflow', 'agentAgentflow']
print(run.usage.to_dict())                           # {'input_tokens':…, 'output_tokens':10, …}

# 2) build a case = one unit test (name + run + evaluators + tags)
case = make_case(
    "agent-counts-1-to-5",
    run,
    [
        AllNodesFinished(),                                   # every canvas node FINISHED
        OutputMatchesRegex(r"(?m)^1\n2\n3\n4\n5$"),           # exact expected output
        NoToolsExpected(),                                    # no stray tool calls
        TokenBudget(output_tokens=25),                        # cumulative token budget
        LatencyBudget(max_ms=15000),                          # agent-node timing budget
    ],
    tags=["agent-1", "text"],
)

# 3) run the suite (one or many cases) and roll everything up
suite = SuiteRunner(suite_name="my-agent").run([case])

# 4) read / export
print(suite.to_markdown())          # human-readable pass/fail table
suite.save_json("report.json")      # machine-readable
suite.save_junit("report.xml")      # CI (Jenkins / GitHub / GitLab)
```

That's it. Everything else — LLM-judged metrics, the judge, multi-turn, file and
streaming — is just *more evaluators and options* on the same three lines.

## The four building blocks

| Block | Role |
|---|---|
| `ForjinnClient` | Talks to the builder (`predict`, `predict_multi_turn`, `stream`, `upload_attachment`). |
| `AgentRun` | The captured run — final text, the **node-by-node trace**, tokens, latency, tool calls, contexts. This is what every evaluator reads. |
| evaluator | One assertion (`AllNodesFinished()`…). Reads an `AgentRun`, returns a 4-state verdict. Compose freely. |
| `SuiteRunner` / `make_case` | Runs cases, rolls up pass/fail + tokens/latency + per-tag, and exports the report. |

- **`AgentRun`** is the atom of data. It is built once from the Forjinn response
  (streaming SSE *or* non-streaming JSON — identical shape) and then read as
  read-only by many evaluators. See [architecture.md](architecture.md).
- **evaluators** are atoms of assertion. Each returns a `CheckResult` with status
  `PASS` / `FAIL` / `ERROR` / `SKIP`. A case passes when **no** check is `FAIL` or
  `ERROR`. `SKIP` is a "this metric needs input the run didn't have" (no judge, no
  retrieved context, no reference) — it is not a silent pass.
- **`make_case(name, run, evaluators, tags=…, reference=…)`** bundles them.
- **`SuiteRunner`** runs cases (optionally in parallel) and produces a `SuiteResult`
  with counts, per-evaluator rollups, usage rollup, per-tag grouping and exports.

## Next

- **[writing-tests.md](writing-tests.md)** — register with `@agent_test`, run via CLI /
  pytest, reference & LLM cases, custom evaluators, budgets, multi-turn, the CI gate.
- **[metrics.md](metrics.md)** — the full deterministic + LLM-judged catalog.
- **[configuration.md](configuration.md)** — environment variables, the judge, offline mode.
- **[api-reference.md](api-reference.md)** — the public API surface, per symbol.
