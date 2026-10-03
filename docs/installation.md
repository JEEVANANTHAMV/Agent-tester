# Installation

`forjinn-eval` targets **Python ≥ 3.9** (3.9–3.12 tested in CI). The only runtime
dependencies are `requests` and `rich`; everything semantic is an **optional
extra** you opt into.

## From this repository

Pick the install that matches your intent:

```bash
# Clone + developer install (editable, all tooling: pytest, mypy, ruff, cov, …)
git clone https://github.com/JEEVANANTHAMV/Agent-tester.git && cd Agent-tester
pip install -e ".[all]"
```

Or point `pip` straight at the repo (no clone needed):

```bash
pip install "forjinn-eval @ git+https://github.com/JEEVANANTHAMV/Agent-tester.git@main"          # latest
pip install "forjinn-eval @ git+https://github.com/JEEVANANTHAMV/Agent-tester.git@v0.2.0"          # a release tag
```

## From PyPI

Once a release has been published:

```bash
pip install forjinn-eval
```

> Releases are cut from **git tags** (`vX.Y.Z`). The version you get on `main`
> is a local dirty build; pin a tag for reproducible installs.

## Optional extras

| Extra | What it adds | Why it's optional |
|---|---|---|
| *(none)* | — | Minimal, fully functional. LLM metrics need only a judge chatflow (the `requests` dep). Embedding metrics use a dependency-free offline fallback. |
| `text` | `rapidfuzz` | Enables Levenshtein distance in `NonLLMStringSimilarity`; a Jaccard fallback is used if absent. |
| `embeddings` | `sentence-transformers` | Uses a real neural embedding model for `SemanticSimilarity` / `AnswerSimilarity` / `ContextualRelevancy`. Without it, a deterministic offline bag-of-words embedding is used (scoring stays meaningful, fully offline). |
| `dev` | `pytest`, `pytest-cov`, `httpx`, `mypy`, `ruff`, `rapidfuzz` | The test/type/check toolchain. |
| `all` | `text` + `embeddings` + `dev` + `build`/`twine` | Everything a maintainer needs. |

```bash
pip install forjinn-eval[text]            # string similarity
pip install forjinn-eval[embeddings]      # real semantic embeddings
pip install forjinn-eval[all]             # everything
```

## Verify the install

```bash
# the console command is on PATH
forjinn-eval --help
forjinn-eval smoke            # offline smoke against recorded samples (no network)
```

```python
import forjinn_eval
print(forjinn_eval.__version__)            # e.g. '0.2.0' or a local dev build
print(forjinn_eval.ForjinnClient)          # importable
```

## Using it in CI

The minimal CI install is just the dev extras (fully offline):

```bash
pip install -e ".[dev]"
pytest -q                 # offline unit + integration tests
forjinn-eval smoke        # recorded-sample smoke
```

Live e2e (optional, secrets-gated) uses the same install plus environment
variables — see [configuration.md](configuration.md) and
[../.github/workflows/ci.yml](../.github/workflows/ci.yml).

## Proxy / corporate network

`ForjinnClient` disables the system proxy and SSL verification by default (Forjinn
self-hosted builders commonly use self-signed certs on a private VLAN). If you
need to keep proxy/verify behaviour, construct the client with explicit settings:

```python
from forjinn_eval import ForjinnClient
client = ForjinnClient(
    base_url="https://172.16.34.7",
    token="eyJ…",        # optional bearer (also sent as Cookie: token=…)
    timeout=60,
    verify_ssl=True,     # keep SSL verification (False by default)
    extra_headers={...}, # extra request headers
    noproxy=False,       # honour the system proxy (True by default, = `curl --noproxy *`)
)
```

The constructor signature is
`ForjinnClient(base_url, token, cookie, timeout, verify_ssl, extra_headers, noproxy)`.
