# Releasing & versioning

`forjinn-eval` is versioned by **git tags** (via `setuptools-scm`). Cutting a
release = tagging a commit; the version propagates to the package, the built
wheel, and (if published) PyPI.

## How the version is derived

`pyproject.toml` configures `setuptools-scm`:

- **`tag_regex = "^v(?P<version>[0-9]+[0-9.]*)$"`** — only `vX.Y.Z` tags count
  (a `v0.2.0` tag → version `0.2.0`).
- **`write_to = "src/forjinn_eval/_version.py"`** — the resolved version is
  written into the package at build time; `forjinn_eval.__version__` reads it
  (falling back to `"0.1.0.dev0"` when there's no `_version.py`, e.g. some editable
  installs — in that case the tag still controls the *distribution* version).
- **untagged / dirty checkouts** build as `0.1.0.devN+<sha>` (dev build), so a
  `pip install …@main` never masquerades as a release.

## Tagging a release

```bash
# 1) make sure the working tree is clean and tests pass
git status --short
pytest -q
forjinn-eval smoke

# 2) tag the commit (use semantic versioning: major.minor.patch)
git tag -a v0.2.0 -m "v0.2.0 — expanded metric catalog + conversational metrics"
git push origin v0.2.0

# 3) verify the version resolves from the tag
python -m build --wheel --no-isolation     # → dist/forjinn_eval-0.2.0-*.whl
```

A tag with a `-` suffix (e.g. `v0.2.0rc1`) produces a pre-release (`0.2.0rc1`);
commits past the latest tag produce a `.devN` suffix.

## Building the artifacts (sdist + wheel)

```bash
pip install build
python -m build            # builds sdist + wheel into dist/
```

Inspect:

```bash
ls dist
unzip -l dist/forjinn_eval-0.2.0-py3-none-any.whl | head   # confirm the new modules are inside
```

## Publishing to PyPI (when ready)

```bash
pip install twine
# Test PyPI first:
twine upload --repository testpypi dist/*
# Production:
twine upload dist/*
```

The release GitHub workflow (`.github/workflows/release.yml`) does this
automatically when you push a `vX.Y.Z` tag (if you've configured a
`PYPI_API_TOKEN` repo secret) — see the next section.

## The release workflow

`.github/workflows/release.yml`:

1. Triggers on a `vX.Y.Z` tag push.
2. Builds the sdist + wheel.
3. Creates a **GitHub Release** with the tag (auto-generated notes).
4. If `secrets.PYPI_API_TOKEN` is set, runs `twine upload`.

So the maintainer flow is simply: tag, push — CI publishes.

## Changelog

Each release should note: new metrics, breaking changes, new CLI options, extra
install options. A concise bullet list in the GitHub release body is enough.

## Local "regular-interval" sync

The maintainer wants pushes at regular intervals. Keep the local `master` in sync
with `origin/master` (a plain push; if you prefer a named remote, `git remote
add origin https://github.com/JEEVANANTHAMV/Agent-tester.git`). On a corporate
network, set the system proxy for git (it does not read the Windows system proxy
automatically):

```bash
git config --local http.proxy http://<your-proxy>:<port>
git push origin master
```
