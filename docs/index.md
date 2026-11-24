# Vettor 🧪⚡

<div class="terminal-hero">
  <div class="terminal-header">
    <span class="terminal-dot red"></span>
    <span class="terminal-dot yellow"></span>
    <span class="terminal-dot green"></span>
    <span class="terminal-title">vettor v0.3.0 — active: openai/gpt-4o-mini</span>
  </div>
  <div class="terminal-content">
    <div><span class="c-dim">[14:22:00]</span> <span class="c-blue">INITIALIZE</span> Target: https://staging.app.example.com | Concurrency: 2</div>
    <div><span class="c-dim">[14:22:01]</span> <span class="c-green">REPLAY</span>     Step 1: Click 'Start Free Trial' <span class="c-dim">[vid=3, hit_cache=100%]</span></div>
    <div><span class="c-dim">[14:22:02]</span> <span class="c-green">REPLAY</span>     Step 2: Fill input 'Work Email' <span class="c-dim">[vid=7, hit_cache=100%]</span></div>
    <div><span class="c-dim">[14:22:03]</span> <span class="c-yellow">DRIFT</span>      Step 3: DOM changed! Expected `#plan-card-pro`, found `#plan-tier-pro`</div>
    <div><span class="c-dim">[14:22:04]</span> <span class="c-purple">HEAL</span>       SelfHealingAgent re-bound drifted node <span class="c-teal">[vid=11, confidence=0.96]</span></div>
    <div><span class="c-dim">[14:22:05]</span> <span class="c-blue">UIAgent</span>    Step 4: Click button 'Complete Onboarding' <span class="c-dim">[vid=15]</span></div>
    <div><span class="c-dim">[14:22:06]</span> <span class="c-green">PASS</span>       Test Case 'Enterprise Trial Signup' verified in 3.2s ✅</div>
  </div>
</div>

**Vettor** is an autonomous web QA agent harness designed for deterministic replay, zero-egress privacy, and instant self-healing.

Inspired by developer tools like `mini-swe-agent` and `omp`, Vettor decouples routine repetitive form execution from costly LLM token inference. It uses **compact local SLMs** (1.5B–4B) for high-speed Set-of-Mark interaction, compiles deterministic **trajectory recipes** for subsequent 0 ms replays, and escalates edge-case bug triage to **frontier LLMs** only when DOM drift or assertion failures occur.

---

## ⚡ Key Highlights

<div class="grid cards" markdown>

-   :material-chip: __Small + Frontier Routing__

    ---

    Run routine browser operations locally with `llama-server` or `Ollama` at 100+ tok/s. Automatically escalate ambiguous triage to GPT-4o, Claude 3.7, or DeepSeek R1.

-   :material-history: __Deterministic Trajectory Replay__

    ---

    Successful test journeys compile into normalized execution recipes. Regression runs replay directly against the DOM with zero LLM inference.

-   :material-bandage: __Instant Self-Healing__

    ---

    When UI redesigns mutate selectors or DOM structures, `SelfHealingAgent` analyzes visual marks and semantic intent to automatically patch recipes without manual script updates.

-   :material-shield-lock: __Local Privacy Lock__

    ---

    Zero cloud data egress when testing sensitive apps. Local PII sanitization redacts emails, tokens, and cookies before any prompt leaves your machine.

-   :material-console: __Developer-First TUI__

    ---

    High-density Textual interface featuring live telemetry counters, streaming trajectory transcripts, and inline Human-In-The-Loop (`HITL`) resolution.

-   :material-infinity: __Zero Domain Hardcoding__

    ---

    Generic, domain-agnostic web testing harness. Works seamlessly across SaaS dashboards, e-commerce checkouts, onboarding funnels, and enterprise portals.

</div>

---

## 🚀 Quick Install

```bash
# Clone the repository
git clone https://github.com/vettor-ai/vettor.git
cd vettor

# Install dependencies with uv
uv sync

# Launch the interactive terminal
uv run vettor
```

Target any web portal directly with a natural-language goal:

```bash
uv run vettor --url https://your-app.com --goal "Verify that signup with an invalid email shows a red error banner"
```

---

## 📖 Explore the Docs

- [Getting Started](getting-started.md): Installation, quick start guide, and running your first test.
- [Model Providers](providers.md): Local SLM setup (`llama-server`, `Ollama`) and frontier cloud routing.
- [Architecture](architecture.md): Deep dive into the multi-agent system, DOM observer, and recipe compilation.
- [Terminal UI](tui.md): Interactive command reference and keyboard shortcuts.
- [Configuration](configuration.md): Complete schema reference for `vettor.yaml` and CI/CD pipelines.
