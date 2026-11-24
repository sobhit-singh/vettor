# Configuration & CI/CD Pipelines ⚙️

Vettor is configured via a YAML file (`vettor.yaml`), environment variables, or CLI options.

---

## Configuration File (`vettor.yaml`)

Create `vettor.yaml` in your project root:

```yaml
target:
  url: "https://staging.app.example.com"
  timeout_ms: 30000
  local_storage:
    feature_flag_beta: "true"
  session_storage:
    region: "us-east-1"

model:
  active: "openai/gpt-4o-mini"
  temperature: 0.1
  privacy_lock: true
  providers:
    openai:
      type: "openai_compat"
      base_url: "https://api.openai.com/v1"
      api_key_env: "OPENAI_API_KEY"

    llamacpp:
      type: "llamacpp_local"
      base_url: "http://127.0.0.1:8080/v1"

    ollama:
      type: "openai_compat"
      base_url: "http://127.0.0.1:11434/v1"
      api_key: "ollama"

    litellm:
      type: "litellm"
      api_key_env: "ANTHROPIC_API_KEY"

browser:
  headless: true
  concurrency: 2
  slow_mo_ms: 0
  viewport_width: 1280
  viewport_height: 800

reporting:
  artifacts_dir: "artifacts"
  save_screenshots: true
  save_video: false
```

---

## CLI Options

All configuration options can be overridden using command-line arguments:

```bash
uv run vettor [OPTIONS]
```

| Flag | Type | Description |
| :--- | :--- | :--- |
| `--cli` | Flag | Run in headless CLI mode instead of launching the TUI |
| `--url <URL>` | String | Target portal URL |
| `--goal <TEXT>` | String | Natural language test goal to plan and execute |
| `--model <ID>` | String | Active provider and model (`provider/model_id`) |
| `--config <PATH>` | Path | Custom path to configuration file (default: `vettor.yaml`) |
| `--test-case <PATH>` | Path | Path to a specific test case JSON file |
| `--concurrency <INT>` | Integer | Number of parallel browser workers |
| `--private` | Flag | Enable strict local zero-egress PII sanitization |
| `--headless` / `--no-headless` | Flag | Run browsers in headless or headed mode |

---

## GitHub Actions CI/CD Integration

Add Vettor to your continuous integration workflow to catch regressions before merging:

```yaml
# .github/workflows/e2e-tests.yml
name: E2E Regression Suite

on:
  push:
    branches: [main]
  pull_request:

jobs:
  qa:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v4
        with:
          version: "latest"

      - name: Set up Python
        run: uv python install 3.12

      - name: Install Dependencies
        run: |
          uv sync
          uv run playwright install --with-deps chromium

      - name: Run Vettor E2E Tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          uv run vettor --cli \
            --url ${{ secrets.STAGING_URL }} \
            --goal "Verify critical signup, login, and billing settings" \
            --model openai/gpt-4o-mini

      - name: Upload Test Artifacts
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: vettor-artifacts
          path: artifacts/
```
