# `hello_agent.py` — a single, minimal, self-contained agent test

This is the **smallest possible forjinn-eval test** — one agent case that works
**fully offline** (no network), so anyone can run it the moment they clone the repo:

```bash
forjinn-eval run examples/hello_agent.py
# or
forjinn-eval list examples/hello_agent.py
```

It uses one recorded fixture from `tests/fixtures/` as the "agent run" and asserts
a handful of deterministic metrics (structure + content + token budget). No live
host, no judge, no tokens.

## What's inside

- **One case** (`hello-works`) built from a recorded capture.
- **Layered metrics**: `AllNodesFinished` (structural) → `RequiredNodesPresent`
  (structural) → `OutputNotEmpty` + `OutputContains` (content/safety) →
  `TokenBudget` (performance).
- **Tags** (`["hello", "text", "offline"]`) you can slice with `suite.group_by`.
- A **`__main__`** block that runs the case and prints the report — so
  `python examples/hello_agent.py` also works.

## Make it talk to your real agent

Change the `predict` block to build a `ForjinnClient` and call `predict` (or
`predict_multi_turn`) against your builder — the rest of the file is unchanged.
The offline path only exists so the example runs in CI with zero credentials.

```python
client = ForjinnClient("https://YOUR_FORJINN_HOST")
run = client.predict("YOUR_CHATFLOW_ID", "Say hello.")
```

That's it. Copy any metric from [docs/metrics.md](../docs/metrics.md) into the
`[AllNodesFinished(), …]` list to assert more.
