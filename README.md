# Vettor

**Autonomous agentic web QA harness** — self-healing, model-driven testing for any web application.

Vettor drives a browser through multi-step workflows, discovers form schemas at runtime, generates valid test data, triages blockers such as cookie banners and validation errors, and records what it learns in persistent procedural memory. It runs against local 2B models (for example LFM3-1.7B on llama.cpp), Ollama, OpenAI-compatible and frontier models through LiteLLM, or a deterministic heuristic fallback, using Rustwright or Playwright as the browser engine.

## Key capabilities

- **Fully configurable target**: point Vettor at any web application (e-commerce, SaaS, onboarding, checkout, multi-step portals) with custom authorization headers, pre-loaded cookies, and Playwright storage state.
- **Configurable testing brain**:
  - **llama.cpp (local 2B)**: auto-discovers GGUF weights in the Hugging Face cache and uses grammar-constrained JSON output for deterministic decisions.
  - **Ollama**: connects to any local endpoint such as `http://localhost:11434/v1`.
  - **OpenAI / OpenAI-compatible**: OpenAI, vLLM, LM Studio, DeepSeek, or Groq.
  - **Deterministic heuristic fallback**: zero-dependency path for offline or CI execution.
- **Autonomous exploration and self-healing**: live DOM inspection, synthetic data generation for required fields, and recovery from blockers, dialogs, and validation errors.
- **Persistent procedural memory** (`qa_memory.json`): remembers site quirks, required declarations, hidden fields, and recovery actions across runs, with trust scoring and reinforcement.
- **Development handoff** (`DEVELOPMENT_HANDOFF.md`): separates functional bugs (P0/P1) from UX friction (P2/P3) with evidence and suggested fixes.
- **Real-time JSONL streaming**: results are written line by line to `artifacts/test_results_YYYY-MM-DD.jsonl`, so progress survives timeouts and aborts.
- **Testability seams**: URL parameter state injection plus pre-seeded `localStorage` / `sessionStorage` for state-dependent UI.
- **Multiple suite formats**: structured `test_cases.jsonl`, `.json`, or `sample_cases.csv`.
- **Pre-flight diagnostics and scaffolding**: `check`, `init`, `handoff`, and `memory` subcommands.
- **Dual browser engines**: Rustwright for fast native replays and Playwright for Chromium, Firefox, and WebKit.
- **Flexible assertions**: numeric extraction with tolerance, text and URL matching, and console/JS exception monitoring.
- **Comprehensive reporting**: standalone interactive HTML report with KPI cards and embedded failure screenshots, plus `results.json` and `results.csv` exports.
- **Interactive TUI and headless CLI**: a Textual dashboard with live KPIs and streaming decision logs, or a non-interactive CLI for CI/CD.

## Requirements

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- Optional: a local GGUF model with llama.cpp for the `llamacpp` backend, and a browser runtime for end-to-end runs

## Installation

```bash
git clone https://github.com/sobhit-singh/vettor.git
cd vettor
uv sync
```

## Quick start

```bash
# Verify the environment and target readiness
uv run vettor check

# Scaffold a fresh workspace with config and starter cases
uv run vettor init

# Run the interactive terminal dashboard
uv run vettor

# Run headless (CI/CD)
uv run vettor --cli

# Override parameters on the command line
uv run vettor --url "https://my-app.local/flow" --suite test_cases.jsonl --engine rustwright --cli
```

### TUI keyboard shortcuts

| Key | Action |
| --- | --- |
| `r` | Run test suite |
| `s` | Stop run |
| `e` | Switch browser engine (Rustwright / Playwright) |
| `b` | Switch inference backend (llama.cpp / Ollama / OpenAI / heuristic) |
| `c` | Clear recipe cache |
| `m` | Start or verify the local mock testing portal |
| `Shift+M` | Switch mode (data-driven / autonomous / crawl-smoke) |
| `h` | Toggle headless / headed browser |
| `t` | Show the action trajectory (also logged to `artifacts/tui_trajectory.jsonl`) |
| `q` | Quit |

TUI action failures are reported inline with the failing control and root cause; full tracebacks go to `vettor.log`.

## Configuration

Everything is configurable through `harness_config.json`, CLI flags, or environment variables:

```json
{
  "target": {
    "url": "http://localhost:8000/step1",
    "base_url": "http://localhost:8000",
    "auth_headers": {},
    "cookies": [],
    "storage_state": null
  },
  "model": {
    "backend": "llamacpp",
    "endpoint": "http://127.0.0.1:8080/v1",
    "api_key": "none",
    "model_name": "LFM3-1.7B",
    "model_path": "",
    "auto_start_llama": true,
    "temperature": 0.0,
    "max_tokens": 500,
    "timeout_s": 20.0
  },
  "browser": {
    "engine": "rustwright",
    "browser_type": "chromium",
    "headless": true,
    "viewport_width": 1280,
    "viewport_height": 800
  },
  "suite": {
    "mode": "data_driven",
    "cases_path": "sample_cases.csv",
    "goal": "Complete the form flow, verify calculations, and validate quote output.",
    "max_pages_per_case": 15
  },
  "reporting": {
    "output_dir": "artifacts",
    "results_path": "results.csv",
    "screenshot_mode": "on_failure",
    "generate_html_report": true,
    "float_tolerance": 0.01
  }
}
```

Treat values such as `api_key`, `auth_headers`, and cookies as secrets: keep them out of commits and supply them at runtime instead.

## Multi-model support (LiteLLM plugin path)

The inference layer is provider-agnostic. `ModelRouter` in [model_provider.py](model_provider.py) builds an ordered provider chain and fails over with a cooldown. Install the optional extra once:

```bash
uv sync --extra litellm
```

Any LiteLLM-supported provider works by setting `model.litellm_model` to `provider/model`:

```bash
# OpenAI-compatible endpoint
uv run vettor --backend litellm --litellm-model openai/gpt-4o-mini \
  --api-base http://localhost:8000/v1 --cli

# Anthropic / Gemini / Ollama (keys read from the usual environment variables)
uv run vettor --backend litellm --litellm-model anthropic/claude-sonnet-4 --cli
uv run vettor --backend litellm --litellm-model gemini/gemini-2.5-flash --cli
uv run vettor --backend litellm --litellm-model ollama/llama3 --cli

# Local LFM3 first, cloud model only if local fails
uv run vettor --backend litellm --provider-chain llamacpp,litellm \
  --litellm-model openai/gpt-4o-mini --cli
```

`provider_chain` also accepts `fallback` (a second LiteLLM model via `fallback_model`). LiteLLM is optional; native OpenAI-compatible backends and the heuristic path work without it.

### LFM3 / llama.cpp note

The local server starts with `--jinja` so the GGUF's own LFM3 chat template is used. Recommended sampling is applied automatically, thinking-only responses are read from `reasoning_content`, and JSON extraction tolerates reasoning prefixes. Server output is captured in `artifacts/llama_server.log`.

## Testing

```bash
uv run python -m unittest tests/test_vettor.py -v
```

Browser end-to-end cases are skipped by default; enable them with `QA_RUN_E2E=1`.

## Project layout

| Path | Purpose |
| --- | --- |
| [main.py](main.py) | CLI and TUI entry point (`check`, `init`, `handoff`, `memory`, `run`) |
| [config.py](config.py) | Configuration schema for target, model, browser, suite, and reporting |
| [engine.py](engine.py) | Execution runner, browser lifecycle, self-healing pipeline, artifact export |
| [agent.py](agent.py) | AI brain: schema mapping, synthetic values, blocker triage, goal planning |
| [model_provider.py](model_provider.py) | Provider-neutral model router with failover |
| [llm_manager.py](llm_manager.py) | Local GGUF discovery and llama.cpp server lifecycle |
| [dom.py](dom.py) | Token-efficient DOM extraction for fields, modals, dialogs, and errors |
| [assertions.py](assertions.py) | Numeric tolerance, text/URL matching, and console error checks |
| [memory.py](memory.py) | Persistent procedural memory with trust calibration |
| [handoff.py](handoff.py) | Development handoff tracker and JSONL result streaming |
| [reporting.py](reporting.py) | Standalone HTML execution report generator |
| [tui.py](tui.py) | Textual dashboard with live KPIs, backend toggling, and log streaming |
| [mockinsurancesite.py](mockinsurancesite.py) | Bundled mock multi-step portal for local demos |
| [harness_config.json](harness_config.json) | Default configuration |
| [test_cases.jsonl](test_cases.jsonl) / [sample_cases.csv](sample_cases.csv) | Example test suites |
| [tests/test_vettor.py](tests/test_vettor.py) | Unit and opt-in browser end-to-end tests |

Generated output (`artifacts/`, `failures/`, `qa_memory.json`, `results.csv`, `vettor.log`, and `DEVELOPMENT_HANDOFF.md`) is intentionally not versioned; it is recreated by a run or by `uv run vettor init`.

## License

MIT — see [LICENSE](LICENSE).
