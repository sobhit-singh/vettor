# Vettor Roadmap: From QA Script Runner to an OMP-Style QA Harness

Prepared for Sobhit Singh. Based on a static read of `vettor-main` v0.3.0 (about 4,900 lines of source plus tests). Nothing was executed, because dependencies were not installable in the review sandbox. Items marked **\[verified\]** were confirmed by reading the exact code path. Items marked **\[suspected\]** need a failing test to confirm.

## 1. Goal and design principles

**Goal.** Vettor should be to QA what a coding harness (omp, pi, Claude Code style) is to programming: a thin, model-agnostic agent loop wrapped in strong tools, persistent sessions, plan mode, subagents, hooks and project memory. It must run with a 1B to 2B local model, give identical correctness guarantees with any model, and scale up to frontier models without code changes.

**Principles to hold every decision against.**

1. **The harness is the intelligence, the model is the tiebreaker.** For a 2B model, code does the matching, verification and recovery. The model only picks among a few constrained options.
2. **Verification is never delegated to the model that acted.** PASS comes from code-checked oracles, not from the model saying `done`.
3. **Tiered execution.** Deterministic replay, then rule-based heuristics, then a constrained small model, then a bigger local model, then (only if allowed) a cloud model. Each tier is cheaper and more private than the next.
4. **Privacy is enforced at the network layer**, not by convention. Offline means offline.
5. **Everything is a session.** Every run is a replayable, resumable, diffable trace.
6. **Eval first.** No change to prompts, schemas or agents lands without a benchmark number before and after.

A note on scope: this maps the *concepts* of coding harnesses onto QA. It does not copy any specific harness's internals.

## 2. Current state: honest assessment

**What is good and should be kept.** The `src/` layout and layer split (`core`, `env`, `agents`, `reporting`, `tui`). The vid (integer element id) approach so models never write selectors. Pydantic schemas with JSON-schema constrained decoding. The compact observation format. The host-based privacy check. The event bus. The recipe and memory concepts. The mock portal fixture.

**What is missing or only claimed.**

| Claim in README or docs | Reality in code |
| --- | --- |
| `SelfHealingAgent` rebinds drifted steps | No such class. No drift event. Healing is a TriageAgent call that clicks something. |
| Privacy lock redacts PII, tokens, cookies | No redaction code exists. Only endpoint host checking. |
| Small plus frontier routing | `use_frontier` exists in the router but is never invoked. |
| Heuristic (no-LLM) backend | Router returns zero providers, so any LLM call raises `ModelError`. |
| Recipe replay matches URL and DOM | Matches normalized URL only. |
| Successful runs compile to recipes | Steps are recorded as they execute, including in failed runs. |
| Benchmark script | `scripts/benchmark.py` imports v1 modules (`engine`, `agent`, `mockinsurancesite`) and cannot run. |
| Tests | All agent tests mock the router. No real small model has ever driven the loop. |

## 3. Bug register

Severity: **S0** produces wrong QA verdicts or data loss. **S1** breaks a core feature. **S2** degrades reliability or misleads. **S3** cleanup.

### S0: wrong verdicts (fix first)

| ID | Where | Problem | Fix | Proving test |
| --- | --- | --- | --- | --- |
| B-01 \[verified\] | `agents/functional.py`, terminal block | When an expected key is not found on the page, code sets `extracted[k] = exp_val`. The numeric assertion then compares the expected value to itself and passes. | Never default extracted values. Missing means `None`, and the assertion fails with "not found". | Mock portal that omits the premium: case must FAIL. |
| B-02 \[verified\] | `functional.py`, `done` branch and `is_terminal_page` | `done` from the model returns PASS with no verification. A start URL containing `/dashboard` or text like "thank you" passes at step 1. | Remove auto-PASS. `done` and terminal detection only trigger the oracle phase. PASS requires every oracle to pass. Terminal check only counts after at least one state-changing action. | Stub model that says `done` immediately: FAIL. |
| B-22 \[verified\] | `reporting/assertions.py` rule 4 | General string containment passes if the expected text appears anywhere in the page body, even unrelated text. | Scope matches to a named region or extracted field. Report where it matched. Warn on body-wide matches. | Expected "500" present only in an unrelated footer. |
| B-23 \[verified\] | `orchestrator.py` `run_single` | Assertions run on a fresh `session.observe()` after the case ends, including ERROR cases, and the page may no longer be the verification page. | Capture the verification observation inside the agent at the moment of completion and attach it to the result. Skip assertions for ERROR. | Case that navigates away after completing. |
| B-06 \[verified\] | `functional.py`, `core/recipe.py` | Recipes are keyed by URL only (not goal, not DOM signature) and are recorded during execution regardless of outcome. A negative test replays the happy path, and failed runs poison the cache. | Record to a pending trace. Promote to a recipe only after the case PASSes with all oracles. Key by (page signature, case intent). | Run failing case, then passing case, confirm cache has only the passing path. |

### S1: core feature failures

| ID | Where | Problem | Fix |
| --- | --- | --- | --- |
| B-03 \[verified\] | `functional.py` step loop | `session.execute_action` and `decide_action` exceptions (bad vid, timeout, `ModelError`) are uncaught in the loop, so one hallucinated vid ends the case as ERROR. | Wrap each step. Convert failures into an observation line ("action failed: no element vid 14") and let the model retry up to N times, then triage. |
| B-04 \[verified\] | `env/observer.py` `PRUNE_JS` | `data-vettor-id` attributes are never cleared. An element that became hidden or disabled keeps its old id, and `querySelector` can return it instead of the current element. | Clear all old `data-vettor-id` at the start of each observe. Better: keep an in-page registry and resolve vids through it. |
| B-05 \[verified\] | `core/llm.py` `detect_model_profile` | Substring match: `32b`, `72b` contain `2b`; `14b`, `24b` contain `4b`; so large models are treated as small. | Parse parameter count with a regex, and fall back to an explicit `profile` in a per-model profile file. |
| B-12 \[verified\] | `core/llm.py` `_build` | `heuristic` backend returns no providers, then every LLM call raises. | Implement the heuristic agent (see Phase 2), and make the router return it as a real provider tier. |
| B-13 \[verified\] | `core/llm.py` | Frontier router is never used by any agent. | Implement escalation policy (Phase 2) with privacy gate. |
| B-07 / B-19 \[verified\] | `functional.py` cautions | Cautions store a `vid`, which changes every observation, and every caution replays on every loop iteration (can click the wrong element repeatedly). | Store a locator fingerprint (role, label, text, selector). Apply once per page entry. |
| B-08 \[verified\] | `agents/ui.py`, `core/...` | `_audited_signatures` lives on a shared agent, so with `workers > 1` case 2 skips the audit that case 1 did. `session.network_errors` and `page_errors` are never cleared, so findings duplicate across steps. | Per-session audit state. Drain error lists after each audit. |
| B-09 / B-24 \[verified\] | `orchestrator.py` | `stop_requested` only breaks the collection loop. Running tasks are not cancelled, and `break` inside `as_completed` leaves tasks running with the browser pool closing under them. | Track tasks, cancel on stop, `gather(..., return_exceptions=True)`, flush partial results. |
| B-16 \[verified\] | `env/browser.py` `execute_action` | No wait for navigation, network idle or DOM mutation after an action, so the next observation races the page. | Post-action settle: wait for load/network quiet and a short DOM mutation quiet window, with a cap. |
| B-25 \[verified\] | `functional.py` | Every case always navigates to the single configured URL. A case cannot have its own start URL, preconditions or auth state. | Add `start_url`, `setup` and `storage_state` per case. |

### S2: reliability and honesty

| ID | Problem | Fix |
| --- | --- | --- |
| B-10 \[verified\] | `assert` is in `ActionType` but `execute_action` silently does nothing for it. | Remove from the action schema, or implement as a real oracle call. |
| B-11 \[verified\] | `execute_action` handles `"type"` but the schema has no `type`. | Align schema and executor from one shared table. |
| B-17 \[verified\] | Observer ignores off-viewport and covered elements, iframes, shadow DOM, `contenteditable`, `[tabindex]`, ARIA widgets (`checkbox`, `combobox`, `switch`), and gives no page text outside errors. | Observer v2 (Phase 1). |
| B-18 \[verified\] | `RecipeStep.default_value` stores real test data (names, values) in `recipe.json`. | Store input keys and placeholders only. Redact at write time. |
| B-20 \[suspected\] | Any provider failure sets a 30 s cooldown. With a single provider, one transient error makes every call fail for 30 s. | Retry with backoff before cooldown. Cooldown only when another provider exists. |
| B-26 \[suspected\] | `json_schema` fallback to `json_object` triggers on error-text keywords only, and `strict: True` with Pydantic `anyOf` may be rejected by some llama.cpp versions. | Capability probe per provider at startup, cached in the model profile. |
| B-27 \[verified\] | `TestCaseAgent` generates a whole suite in one JSON call with 60 elements of context. A 2B model cannot do this reliably. | Plan one case per call from a field list. |

### S3: cleanup

- **B-14** Delete or rewrite `scripts/benchmark.py` against the v3 API (becomes the Phase 7 eval runner).
- **B-15** Align README and docs with reality: `concurrency` versus `workers` config key, the clone URL organisation, `SelfHealingAgent`, redaction claims. Add a "status" column (implemented, partial, planned) to the feature list.
- **B-28** `sample_cases.jsonl` and `benchmark.py` are insurance-portal specific. Move to `examples/` and make fixtures generic.

## 4. Target architecture

```
vettor/
  harness/          agent loop, tool registry, context manager, session store
  tools/            observe, act, navigate, scroll, wait, extract, assert, screenshot, note_bug
  oracles/          text, url, numeric, element, network, a11y, visual, judge (LLM)
  tiers/            replay, heuristic, small_llm, local_large, cloud
  agents/           explorer, executor, judge, triage (subagents, each with its own prompt and tool subset)
  models/           providers, capability probe, profiles/*.yaml
  privacy/          redactor, egress guard, audit log
  session/          jsonl trace, resume, fork, replay
  project/          VETTOR.md loader, skills, hooks
  interfaces/       tui, headless json/rpc, mcp server, ci
  evals/            fixture sites, tasks, metrics, matrix runner
```

**The agent loop (omp-style, minimal).** Observe, decide, act, settle, verify, record. One loop for all models. Differences between a 2B and a frontier model live in the *profile*, not in code branches: context budget, element cap, history window, whether to batch actions, which tiers are allowed.

**Model profiles.** A YAML per model family: `max_elements`, `history_turns`, `prompt_style` (json, line format, grammar), `supports_json_schema`, `supports_vision`, `temperature`, `max_tokens`, `batch_actions`, `allowed_tiers`. Detected by a capability probe on first use and cached.

## 5. Roadmap

Each phase has an exit gate. Do not start the next phase until the gate passes. Durations assume one developer working part time with a coding agent doing the bulk of the implementation.

### Phase 0: Stabilize and make results trustworthy (about 1 week)

Fix S0 bugs and the cheap S1 bugs: B-01, B-02, B-22, B-23, B-06, B-03, B-04, B-05, B-08, B-09/24, B-10, B-11, B-14, B-15. Write one failing test per bug *before* the fix. Add a `ScriptedModel` test double that returns canned actions, so the full loop is tested without mocking the router away.

**Gate:** Mock portal runs: a scripted model that says `done` early FAILS; a portal with a wrong premium FAILS; a negative case does not poison recipes; CI green on lint, tests and a 3-case e2e.

### Phase 1: Harness core and Observer v2 (about 2 weeks)

1. **Tool registry.** Define each tool once with name, JSON schema, executor and result formatter. The action schema, prompt tool list and executor are generated from the registry, which removes B-10/B-11 permanently.
2. **Agent loop.** Replace the monolithic `run_case` with a loop that is a thin function over the registry, tiers and oracles. Failures become observations.
3. **Observer v2.**
   - In-page vid registry, cleared per observation (finishes B-04).
   - Shadow DOM and same-origin iframe traversal.
   - ARIA widgets, `contenteditable`, `[tabindex]`, clickable `div` heuristics (cursor pointer plus handler).
   - Viewport and covered-element detection (`elementFromPoint`), with a `scroll` tool and page regions (header, main, dialog).
   - A short "visible text" section (headings, status messages, result values) so verification is possible from the observation.
   - DOM diff between observations (added, removed, changed) so the model and loop detector see what an action did.
4. **Settle logic** (B-16) and **loop detection**: identical action and unchanged DOM hash twice triggers a recovery strategy, then escalation.

**Gate:** Eval suite from Phase 7 (minimal version: 5 fixture pages) shows no regressions and at least 90 percent on the scripted model.

### Phase 2: Small-model mode (about 2 to 3 weeks)

This phase is what makes sub-2B practical. Order matters.

1. **Heuristic tier (no LLM).** Match input keys to elements by normalized label, name, id, placeholder and type, using token overlap plus simple synonyms. Handle selects by option matching, checkboxes by boolean parse, and "next/submit/continue" buttons by role and text. This alone should complete most plain forms with zero model calls.
2. **Constrained decoding.**
   - Inject the current vid list as an `enum` into the action schema each step.
   - Provide a GBNF grammar for llama.cpp as an alternative.
   - Add a line format (`fill 3 "Aarav"`) as a fallback for models that fail JSON.
3. **Prompt compression.** Profile-driven element cap (about 12 to 20 for 2B), ranked by relevance to the case inputs. System prompt under 150 tokens. One or two few-shot examples per profile. History reduced to a one-line-per-step summary.
4. **Decomposition.** Break goals into micro-decisions: "which element for field X?" rather than "what should I do next?". The loop asks only the question the heuristic could not answer.
5. **Planner per case.** Fix B-27 by generating one case per call from a field list, with a deterministic generator for boundary values (empty, max length, negative, unicode, injection strings) so the model is not needed for data.
6. **Escalation policy.** Define `tiers` and a rule table: confidence below a threshold or two failed attempts moves up one tier. Cloud tiers are blocked when `confidential` or `--offline` is set, and the run reports "stuck, needs human" instead.
7. **Model profiles and capability probe** (B-05, B-26).

**Gate:** On the eval suite, a 1.5B to 2B model reaches at least 85 percent task success with the heuristic tier enabled and at least 60 percent with it disabled. Report both numbers, because the gap tells you whether the harness or the model is the weak link.

### Phase 3: Test spec and oracles (about 2 weeks)

1. **Test spec format.** YAML or Markdown with front matter. Fields: `id`, `start_url`, `setup`, `goal`, `inputs`, `steps` (optional, for scripted flows), `expect` (list of oracles), `tags`, `tolerance`, `negative: true`. Keep CSV and JSONL importers.
2. **Oracle library.** Text in region, URL pattern, numeric with tolerance, element present or absent, element state, network call made (method, URL, status, JSON path), console clean, a11y (axe-core), visual diff against a baseline, and an optional **LLM judge** oracle that is always secondary and always logged as "model-judged".
3. **Negative and boundary testing.** `expect_error` oracles (the app should reject input X with a message matching Y).
4. **Finding quality.** Severity rules, dedup by signature, reproduction steps attached from the trace, and a suggested minimal repro.

**Gate:** The mock portal has seeded bugs (wrong total, missing validation, 500 on submit, broken image, overflow). Vettor must find at least 90 percent of them with zero false positives on the clean version.

### Phase 4: Recipes and real self-healing (about 2 weeks)

1. **Locator fingerprint** per step: role, accessible name, label, text, tag, type, neighbor labels, selector, and position region.
2. **Recipe key** = page signature plus case intent. Promotion only after a PASS (B-06). Store placeholders, not data (B-18).
3. **Drift detection.** Before each replayed step, check that the fingerprint resolves uniquely. Emit a `Drift` event when it does not.
4. **Healing ladder.** Exact selector, then role and name, then label similarity, then heuristic tier, then small model, then human. A healed step is a *proposal*; it is committed to the recipe only if the case then passes.
5. **Recipe versioning** with a diff viewer so a human can review what healing changed.

**Gate:** Mutation test: automatically rename ids, classes and button text in the mock portal. At least 80 percent of recipes self-heal without a model call, and none heal to a wrong element and pass.

### Phase 5: Privacy and zero-egress (about 1 to 2 weeks)

1. **Egress guard** at the Playwright route level and at the HTTP client level. In `--offline`, only the target host and the local model endpoint are reachable. Everything else is blocked and logged.
2. **Redactor.** Detect emails, phones, card numbers, tokens, JWTs, cookies, and configured regexes. Apply before any prompt leaves the process, before anything is written to traces, recipes or memory, and in reports.
3. **Tier gating.** `confidential` forces local-only tiers, disables cloud escalation and disables any screenshot upload.
4. **Audit log.** Every model call: tier, provider, host, token counts, redaction count. A `vettor audit` command prints it.
5. **Hardening.** Disable LiteLLM telemetry, verify no outbound calls at startup, and document the threat model.

**Gate:** A network-sandbox test (no route except localhost) completes the full local suite. A red-team test with fake secrets in the page confirms none appear in prompts, traces or reports.

### Phase 6: Harness experience, the omp-style layer (about 3 weeks)

This is what turns it from a tool into a harness.

1. **Sessions.** Every run writes a JSONL trace (observations, model calls, actions, oracle results). Commands: `resume`, `fork` (branch from step N with a different hint), `replay` (deterministic, no model), `diff` (compare two runs).
2. **Plan mode.** `/plan` explores the app read-only (no submissions) and produces a reviewable test plan. The human edits it, then `/run` executes. Plan mode never performs destructive actions.
3. **Permission modes.** Read-only, safe (no submit or delete), full. Destructive-action guard with an allowlist of URLs and button patterns.
4. **Subagents.** Explorer (maps the app), Executor (runs cases), Judge (secondary verdicts), Triage (root cause). Each has its own prompt, tool subset and model tier, so a 2B model can run Executor while a bigger local model only handles Triage.
5. **Project memory file.** `VETTOR.md` at repo root: app description, test accounts (references only), known quirks, forbidden actions, terminal markers, and style rules. Loaded into every session within the token budget.
6. **Skills.** Reusable procedures stored as small files ("login with SSO", "add item to cart", "dismiss cookie banner") that the loop can invoke as a single tool. Skills are also how healed cautions graduate from memory.
7. **Hooks.** `before_action`, `after_action`, `on_finding`, `on_run_end` as shell or Python callbacks (Slack, Jira, GitHub issue creation).
8. **Interfaces.** Keep the TUI. Add a headless `--json` event stream and a small RPC/MCP server so other agents (including coding agents) can call Vettor as a tool.
9. **Checkpoints.** Save browser storage state at named points so a long flow can resume from step N.

**Gate:** Kill a run mid-way, resume it, get identical results. A coding agent can invoke Vettor through MCP, run a smoke suite and read back findings.

### Phase 7: Evaluation harness (start in Phase 0, finish here, about 2 weeks total)

Build this early. It is how every earlier phase proves itself.

1. **Fixture sites** (local, deterministic): login, multi-step wizard, checkout with totals, searchable table with pagination, SPA with modals and toasts, file upload, custom dropdowns and date pickers, iframe widget, and a page with seeded bugs. Each has a hidden ground truth.
2. **Task set.** At least 40 tasks with machine-checkable success and, for bug pages, a seeded-bug list.
3. **Metrics.** Task success rate, false PASS rate (the most important), false FAIL rate, bug recall and precision, steps per task, tokens per task, wall time, cache hit rate, heal success rate.
4. **Model matrix.** Run the same suite across 0.5B, 1.5B, 3B, 7B and one frontier model, with and without each tier. Output a table and a JSON file.
5. **Regression gates in CI.** Heuristic-only and scripted runs on every PR. A nightly small-model run. Fail the PR if false PASS rate rises at all.

### Phase 8: Release hardening (about 1 to 2 weeks)

Packaging (pip, uv, optional Docker with browsers), a config schema with validation and clear errors, `vettor doctor` (checks browser, model server, privacy settings), docs that mark each feature as implemented, partial or planned, a security policy, an example repo, versioned releases and changelog discipline.

## 6. Milestones

| Milestone | Contents | Target |
| --- | --- | --- |
| M0 Trustworthy | Phase 0 plus minimal eval | Week 1 |
| M1 Harness core | Phase 1 | Week 3 |
| M2 Small-model ready | Phase 2 | Week 6 |
| M3 Real QA | Phases 3 and 4 | Week 10 |
| M4 Private by construction | Phase 5 | Week 11 |
| M5 Harness experience | Phase 6 | Week 14 |
| M6 Public release | Phases 7 and 8 | Week 16 |

Phases 3 to 5 can overlap if you have a second contributor. Phases 0 and 7 must not be skipped or deferred.

## 7. How to carry out the development

**Working method.**

1. **One bug or feature per branch**, one PR, one failing test first. The PR description links the bug ID from this document and shows the before and after eval numbers.
2. **Eval before features.** Stand up the minimal fixture sites and the `ScriptedModel` in the first days. Every later change is judged by the benchmark, not by a demo.
3. **Trace-driven debugging.** When a small model fails, read the JSONL trace and fix the harness (prompt, observation, constraint) before blaming the model. Most 2B failures are harness failures.
4. **Keep the loop small.** If the agent loop grows past a few hundred lines, move logic into tools, oracles or tiers.
5. **Decision log.** A `docs/decisions/` folder with one short file per major choice (grammar versus JSON, tier thresholds, recipe key design). Future you will need it.
6. **Definition of done.** Tests pass, lint passes, the eval suite shows no worse false PASS rate, docs and README status column updated, trace format version bumped if it changed.

**Using a coding agent to build it.** Use a coding agent for implementation and keep yourself as reviewer and spec author. Give it one phase at a time with: this document's section, the relevant bug rows, the exit gate, and the instruction to write the failing test first. Review the tests more carefully than the code, since tests encode what QA correctness means here. Do not let an agent change eval fixtures and the code under test in the same PR.

**Model choices for development.** Develop against a ladder: a scripted model for logic, a 1.5B to 2B model as the primary target, a 7B as a sanity ceiling, and one frontier model only to separate harness bugs from model limits. If it works on the 2B with the heuristic tier, it will work everywhere above it.

## 8. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| False PASS erodes trust in the tool | Oracles in code, false PASS rate as the top CI metric, model-judged results labeled as such |
| Small model too weak even with constraints | Heuristic tier handles the common path; unresolved items go to human-in-the-loop, not guesses |
| Self-healing heals to the wrong element and passes | Healing is a proposal, committed only after full oracle pass, and reviewed through recipe diffs |
| Real sites change or block automation | Fixtures for CI; real-site runs are opt-in smoke tests with allowlists |
| Privacy claims outrun implementation | Network-level enforcement plus a red-team test; docs label each guarantee with its test |
| Scope creep into a full test platform | Keep the core loop small; push integrations to hooks and MCP |
| Destructive actions on real environments | Permission modes, URL allowlist, read-only plan mode, confirmation prompts |

## 9. First seven days

1. **Day 1.** Add `ScriptedModel`, the early-`done` test and the wrong-premium test. Both should currently fail, proving B-01 and B-02.
2. **Day 2.** Fix B-01, B-02, B-22, B-23 (verdict correctness).
3. **Day 3.** Fix B-03 and B-04 (loop resilience and stale vids).
4. **Day 4.** Fix B-06 (recipe promotion on pass only) and B-18 (no data in recipes).
5. **Day 5.** Fix B-05, B-08, B-09/B-24, B-10/B-11.
6. **Day 6.** Replace `scripts/benchmark.py` with the minimal eval runner and 5 fixture pages.
7. **Day 7.** Update README and docs with a status column (B-15), tag v0.3.1, record the baseline numbers for the 2B model.

After day 7, start Phase 1 with the tool registry, since it removes a whole class of bugs by construction.
