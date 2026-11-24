# Contributing to Vettor

Thanks for helping improve Vettor. This document covers the local workflow.

## Setup

```bash
git clone https://github.com/sobhit-singh/vettor.git
cd vettor
uv sync
```

Python 3.11 or newer and [`uv`](https://docs.astral.sh/uv/) are required.

## Before opening a pull request

1. Keep changes focused on one concern.
2. Add or update tests for behavioural changes.
3. Run the fast suite:

   ```bash
   uv run python -m unittest tests/test_vettor.py -v
   ```

4. Opt in to browser end-to-end tests when a browser runtime is available:

   ```bash
   QA_RUN_E2E=1 uv run python -m unittest tests/test_vettor.py -v
   ```

5. Confirm no generated artifacts, credentials, or local state are staged.
   The root `.gitignore` already excludes virtual environments, caches, logs,
   `artifacts/`, `failures/`, and runtime memory.

## Conventions

- Runtime dependencies belong in `pyproject.toml`; development-only tools in
  `[dependency-groups]`.
- Regenerate `uv.lock` whenever dependency metadata changes (`uv lock`).
- Use relative Markdown links so documentation renders on GitHub and PyPI.
- Never commit API keys, tokens, or `.env` files. Use environment variables or
  the CLI's configuration flags instead.

## Commit messages

Write a short, imperative summary (for example, `Add retry budget to model
router`). Explain the reason in the body when it is not obvious from the diff.
