# Configuration

Everything is configured through **environment variables** — there is no config
file to create. Set them in your shell, in a `.env` (loaded by your runner), or in
CI secrets.

## Variables

| Variable | Default | Scope | Purpose |
|---|---|---|---|
| `FORJINN_HOST` | `https://172.16.34.7` | client | Base URL of your Forjinn builder. |
| `FORJINN_TOKEN` | — | client | Bearer token for the builder (also sent as `Cookie: token=…`). |
| `FORJINN_OFFLINE` | `0` | client / judge | `1` → replay recorded fixtures in `tests/fixtures/` + a permissive judge. **No network.** |
| `FORJINN_LLM_JUDGE` | `0` | suite | `1` → auto-attach the standard LLM battery + a description-seeded GEval to *every* case. |
| `FORJINN_JUDGE_CHATFLOW` | — | judge | Chatflow id of a Forjinn **judge** agent. Required for a real judge. |
| `FORJINN_JUDGE_BASE_URL` | = `FORJINN_HOST` | judge | Separate host for the judge (if it lives elsewhere). |
| `FORJINN_JUDGE_STREAMING` | `0` | judge | `1` → drive the judge over the SSE streaming path. |
| `FORJINN_LLM_TESTS` | — | tests | `1` → run the opt-in **real-LLM** tests (`pytest -m live`). Off in default CI. |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | embeddings | Model for `Embeddings` **if** `sentence-transformers` is installed (`[embeddings]` extra). Ignored otherwise. |
| `FORJINN_AGENT_NO_TOOLS` / `FORJINN_AGENT_WITH_MCP` | (sample ids) | examples / live CI | Chatflow ids used by the example modules and the live e2e job. |

> There is intentionally **no** per-agent config file. A "config" is just: the host
> + token + (optionally) a judge chatflow + (optionally) offline. The rest is
> code — the evaluators in your test module.

---

## Scenarios

### 1. Live agent tests, no LLM judge

```bash
export FORJINN_HOST="https://172.16.34.7"
export FORJINN_TOKEN="eyJ…"          # only if the builder is authed
forjinn-eval run my_agent_tests.py --min-pass-rate 1.0 --fail-on-gate
```

### 2. Live + a real LLM judge (the same builder)

Build a small "judge" chatflow in Forjinn: a text-chat canvas with any model and a
system prompt that ends with *"respond with JSON only, exactly the requested keys"*.
Then:

```bash
export FORJINN_HOST="https://172.16.34.7"
export FORJINN_LLM_JUDGE=1
export FORJINN_JUDGE_CHATFLOW="11111111-2222-3333-4444-555555555555"
export FORJINN_JUDGE_STREAMING=1        # optional
forjinn-eval run my_agent_tests.py
```

Now every case **also** gets: Answer Relevancy, Faithfulness, Hallucination, Prompt
Alignment, Bias, Toxicity, and a `GEval` seeded from the case's description. Set
these **per case** instead (explicit `judge=` on each metric) when you want finer
control — see [writing-tests.md §4](writing-tests.md#4-llm-judged-cases-two-ways).

### 3. Fully offline / CI (zero network)

```bash
export FORJINN_OFFLINE=1
pytest -q                                        # unit + integration, offline
forjinn-eval smoke                               # recorded-sample smoke
forjinn-eval web-scenarios --offline             # full catalog, offline

# also run the LLM catalog offline (permissive mock judge):
export FORJINN_LLM_JUDGE=1
forjinn-eval smoke
```

This is what the packaged CI runs — the whole metric catalog executes with no
network, so the catalog *itself* is regression-tested.

### 4. Real-LLM suite (opt-in, not in default CI)

Drives the **actual** Forjinn SSE judge against a live host:

```bash
FORJINN_LLM_TESTS=1 \
FORJINN_HOST="https://172.16.34.7" \
FORJINN_TOKEN="eyJ…" \
FORJINN_JUDGE_CHATFLOW="1111…" \
FORJINN_JUDGE_STREAMING=1 \
  pytest -m live
```

### 5. Real semantic embeddings

```bash
pip install forjinn-eval[embeddings]          # installs sentence-transformers
export EMBEDDING_MODEL="all-MiniLM-L6-v2"     # optional
```

Without the extra, `SemanticSimilarity` / `AnswerSimilarity` /
`ContextualRelevancy` use the deterministic offline bag-of-words fallback.

---

## The judge in detail

The judge is **another Forjinn agent** — not a separate SDK or API key. Its output
is expected to be **JSON** with the keys the metric asked for (e.g.
`{"verdict": 1, "reason": "…"}` or `{"score": 9, "reason": "…"}`).

- **Resolution.** `JudgeClient.from_env()` / the suite overlay build the judge from
  `FORJINN_JUDGE_CHATFLOW` (+ `FORJINN_JUDGE_BASE_URL`, `FORJINN_JUDGE_STREAMING`).
- **Offline.** `FORJINN_OFFLINE=1` swaps in a **shape-aware permissive `MockJudge`**
  (`queuing_judge()` → `_offline_responder`): it scans each rubric for the JSON keys
  it names and returns a favourable value, sized to the rubric — so the entire LLM
  catalog runs offline and deterministically.
- **Per-call scripting.** For tests, `queuing_judge(script=[…], fallback=…)` builds a
  `MockJudge` that answers a scripted reply per call (FIFO) — see `tests/unit/`.
- **Default judge.** `set_default_judge(client)` installs a process-wide judge so a
  metric can be constructed without an explicit `judge=`; `get_default_judge()`
  builds one lazily from the environment (and returns a permissive mock when
  offline).

## Token & security notes

- Pass `FORJINN_TOKEN` as an env var or CI secret — never hard-code it.
- `ForjinnClient(..., verify_ssl=False, noproxy=True)` is the default because
  self-hosted Forjinn builders commonly run on a private VLAN with self-signed
  certs (mirrors `curl --noproxy *`). Set `verify_ssl=True` / `noproxy=False` if
  you need normal TLS/proxy behaviour — see [installation.md](installation.md).
- `.env`, `*.local` and `forjinn-eval.local.json` are git-ignored; generated reports
  under `out/` are git-ignored too.
