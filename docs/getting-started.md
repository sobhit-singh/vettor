# Getting Started 🚀

This guide covers installing Vettor, configuring your model provider, and executing your first autonomous web QA test.

---

## Prerequisites

- **Python 3.11+** installed.
- **[uv](https://github.com/astral-sh/uv)** installed (recommended) or standard `pip`.
- Playwright system browser binaries.

---

## Installation

### 1. Clone & Sync

```bash
git clone https://github.com/vettor-ai/vettor.git
cd vettor

# Sync core runtime dependencies
uv sync

# Install Playwright browser dependencies (Chromium)
uv run playwright install chromium
```

### 2. Optional Development & Docs Groups

```bash
# Install testing & linting tools
uv sync --extra dev

# Install documentation build tools
uv sync --extra docs
```

---

## Running Vettor

### Interactive Terminal Mode (TUI)

Launching `vettor` with no subcommand starts the developer TUI:

```bash
uv run vettor
```

You can pass initial flags directly to configure your session:

```bash
uv run vettor --url https://staging.myapp.com --model openai/gpt-4o-mini
```

Inside the TUI, type instructions or slash commands into the prompt:

```text
> /plan Test checkout form with promo code 'SUMMER25'
> /run
```

---

### Headless CLI Mode

For headless regression tests or CI pipelines, pass the `--cli` flag:

```bash
uv run vettor --cli \
  --url https://example.com/checkout \
  --goal "Verify empty submission triggers credit card validation errors" \
  --model openai/gpt-4o-mini
```

When execution completes, Vettor prints a clean summary and saves results to `results.csv` and `artifacts/`.

---

## Your First Test Plan

A test case can be authored in JSON or generated autonomously using `/plan`.

Example `test_cases/login_test.json`:

```json
{
  "id": "case_01",
  "name": "Verify Login Validation",
  "url": "https://example.com/login",
  "steps": [
    {
      "action": "fill",
      "selector": "input[type='email']",
      "value": "invalid-email"
    },
    {
      "action": "click",
      "selector": "button[type='submit']"
    }
  ],
  "expected": {
    "contains": "Please enter a valid email address"
  }
}
```

Execute this test case directly:

```bash
uv run vettor --cli --test-case test_cases/login_test.json
```

---

## Next Steps

- Learn how to connect local SLMs in [Model Providers](providers.md).
- Understand how deterministic recipes work in [Architecture](architecture.md).
- Explore all interactive shortcuts in [Terminal UI](tui.md).
