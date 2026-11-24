# Terminal UI & Slash Commands 🕹️

Vettor features a developer-first, high-density Terminal User Interface (TUI) powered by [Textual](https://github.com/Textualize/textual).

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│ VETTOR │ URL: https://app.example.com │ MODEL: openai/gpt-4o-mini            │
│ STATUS: IDLE │ PASS: 5 │ FAIL: 0 │ ERR: 0 │ BUGS: 0 │ CACHE: 80%              │
├──────────────────────────────────────────────────────────────────────────────┤
│ 14:30:01 [INIT]      Loaded test suite: 6 cases                              │
│ 14:30:02 [REPLAY]    Case #1 'Login Flow' -> Step 1 [vid=1, hit_cache=100%]  │
│ 14:30:03 [REPLAY]    Case #1 'Login Flow' -> Step 2 [vid=2, hit_cache=100%]  │
│ 14:30:04 [PASS]      Case #1 'Login Flow' PASSED in 1.2s                     │
│ 14:30:05 [LLM]       Case #2 'Search Inventory' -> Generating trajectory... │
├──────────────────────────────────────────────────────────────────────────────┤
│ [HITL REQUESTED] Agent needs hint: 'Which billing plan should be chosen?'    │
├──────────────────────────────────────────────────────────────────────────────┤
│ > /plan Verify enterprise checkout with promo discount                       │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Layout Components

1. **Top Telemetry Bar**:
   - Shows active URL, loaded model ID, execution status (`IDLE`, `RUNNING`, `PAUSED`, `HITL`), pass/fail/bug counters, and recipe cache hit rate.
2. **Streaming Transcript (`RichLog`)**:
   - Displays real-time, color-coded trajectory logs grouped by case and agent (`REPLAY`, `LLM`, `HEAL`, `UIAgent`, `TriageAgent`).
3. **Inline HITL Resolution Bar**:
   - Stays hidden during normal execution.
   - When an agent detects ambiguity or requests Human-In-The-Loop assistance, this bar slides in directly above the command input.
4. **Unified Command Bar (`>`)**:
   - Accepts both natural language goals and slash commands.

---

## Slash Commands Reference

| Command | Arguments | Description |
| :--- | :--- | :--- |
| `/run` | `[file.json]` | Run the active test suite or a specific test case file |
| `/plan` | `<goal>` | Autonomously decompose a high-level goal into an actionable test plan |
| `/model` | `[provider/model_id]` | Switch active model (`llamacpp/...`, `openai/...`, `ollama/...`) |
| `/workers` | `<int>` | Adjust browser pool concurrency (default: `1`) |
| `/visual` | — | Toggle Set-of-Mark (SoM) visual screenshot mode |
| `/private` | — | Toggle local zero-egress PII sanitization |
| `/handoff` | — | View active Human-In-The-Loop requests and hints |
| `/memory` | — | Inspect discovered DOM selectors, form mappings, and bug signatures |
| `/clear` | — | Clear current transcript logs |
| `/stop` | — | Abort currently running test execution |
| `/help` | — | Display command help and active configuration |
| `/quit` | — | Gracefully close browser sessions and exit Vettor |

---

## Human-in-the-Loop (`HITL`) Resolution

When Vettor encounters an unknown prompt, CAPTCHA gate, or 2FA challenge, it publishes a `HITLRequested` event:

1. The inline HITL bar turns amber with the agent's question.
2. Type your response directly into the command prompt:
   - Natural language guidance: `Select the third dropdown item and click Submit`
   - Explicit mark assignment: `vid:3=john@example.com`
   - Selector override: `#submit-button=click`
3. Press <kbd>Enter</kbd> to immediately resume execution.
