# Changelog

All notable changes to Vettor will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-11-25

### Added
- **Full Async `src/`-Layout Architecture**: Complete redesign separating `vettor.core`, `vettor.env`, `vettor.agents`, `vettor.reporting`, and `vettor.tui`.
- **Small-to-Frontier Model Support**: Integer Visual Element IDs (`vid`) paired with constrained decoding (`json_schema` / GBNF), eliminating fragile CSS selector synthesis and enabling reliable autonomous execution on lightweight local models (e.g., `LFM3-1.7B`, `Qwen2.5-3B`) and frontier models (`GPT-4o`, `Claude 3.5 Sonnet`).
- **Specialized Multi-Agent Squad**:
  - `FunctionalityAgent`: Workflow driver with deterministic fast-path replay acceleration.
  - `UIAgent`: Zero-LLM contrast, accessibility, layout stability, console, and network probes, plus optional Set-of-Mark (SoM) visual screenshot audits.
  - `TriageAgent`: Stalled workflow diagnostics distinguishing actionable heals from genuine bugs.
  - `TestCaseAgent`: Autonomous discovery planner generating structured test suites from scratch.
- **Unified Provider Registry**: Pre-configured support for `openai`, `openrouter`, `ollama`, `llamacpp`, `litellm`, and custom self-hosted VPS/vLLM endpoints with live TUI registration (`/provider add`).
- **Zero-Trust Privacy Lock**: Host validation enforcing local/loopback, RFC1918 private subnets, Tailscale CGNAT IPs, and explicitly allowed private hosts (`allowed_private_hosts`) while strictly blocking public cloud endpoints in confidential mode.
- **Developer-First Terminal UI**: `omp.sh`-inspired keyboard-driven Textual interface featuring a 1-line top telemetry bar, indented streaming trajectory transcript (`TranscriptLog`), inline human-in-the-loop prompt (`HITLBar`), and dual-mode slash command / natural language goal prompt.
- **Deterministic Recipe Replay Cache**: Trajectory compiler that stores reliable form workflows, achieving 0-LLM token replays on subsequent runs and automated self-healing upon DOM mutations.
- **Domain-Agnostic Assertions & Reporting**: Substring matching, numeric float tolerance comparison, automatic issue deduplication in `DEVELOPMENT_HANDOFF.md`, JSONL streaming, and dark-mode standalone HTML reports.
- **CI/CD & Documentation**: GitHub Actions pipelines for automated linting, unit testing, and GitHub Pages documentation deployments via MkDocs Material.
