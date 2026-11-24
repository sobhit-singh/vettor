import argparse
import json
from pathlib import Path
import sys
import urllib.request
import urllib.error
import time

from config import QAConfig
from engine import QAEngine, _get_playwright, _find_chromium_executable
from handoff import HandoffManager, HANDOFF_HEADER
from memory import QAMemory
from tui import VettorTUI
import llm_manager


def cmd_init(args):
    """Scaffold a full agentic QA workspace with config, tests, memory, and handoff."""
    print("=" * 60)
    print("Vettor: Initializing Workspace Scaffolding")
    print("=" * 60)

    cfg_path = Path("harness_config.json")
    if not cfg_path.exists() or getattr(args, "force", False):
        cfg = QAConfig()
        cfg.save(cfg_path)
        print(f"[+] Created configuration file: {cfg_path}")
    else:
        print(f"[i] Configuration file already exists: {cfg_path}")

    cases_path = Path("test_cases.jsonl")
    if not cases_path.exists() or getattr(args, "force", False):
        sample_cases = (
            '{"case_id": "1", "name": "Standard Non-Smoker Base Quote", "full_name": "Aarav Sharma", "age": 30, "smoker": "no", "coverage_amount": 500000, "expected_premium": 345.00}\n'
            '{"case_id": "2", "name": "Smoker Surcharge High Coverage", "full_name": "Priya Nair", "age": 45, "smoker": "yes", "coverage_amount": 1000000, "expected_premium": 955.00}\n'
            '{"case_id": "3", "name": "Young Adult Medium Coverage", "full_name": "Rohan Verma", "age": 28, "smoker": "no", "coverage_amount": 750000, "expected_premium": 462.00}\n'
        )
        cases_path.write_text(sample_cases, encoding="utf-8")
        print(f"[+] Created structured test cases: {cases_path}")
    else:
        print(f"[i] Test suite already exists: {cases_path}")

    handoff_path = Path("DEVELOPMENT_HANDOFF.md")
    if not handoff_path.exists() or getattr(args, "force", False):
        handoff_path.write_text(HANDOFF_HEADER, encoding="utf-8")
        print(f"[+] Initialized development handoff tracker: {handoff_path}")
    else:
        print(f"[i] Development handoff tracker already exists: {handoff_path}")

    memory_path = Path("qa_memory.json")
    if not memory_path.exists() or getattr(args, "force", False):
        mem = QAMemory(memory_path)
        mem.save()
        print(f"[+] Initialized persistent knowledge repository: {memory_path}")
    else:
        print(f"[i] Memory repository already exists: {memory_path}")

    print("\n[OK] Workspace initialized successfully!")
    print("Next steps:")
    print("  1. Run pre-flight environment check:  uv run vettor check")
    print("  2. Start test suite in TUI mode:      uv run vettor")
    print("  3. Run headless in CLI mode:          uv run vettor --cli")


def cmd_check(cfg: QAConfig):
    """Run pre-flight environment readiness and diagnostics checks."""
    print("=" * 65)
    print("Vettor: Pre-flight Readiness Diagnostics")
    print("=" * 65)

    all_passed = True

    # 1. Target URL Check
    target_url = cfg.start_url
    print(f"\n[1/5] Target Web Application: {target_url}")
    try:
        t0 = time.perf_counter()
        req = urllib.request.Request(target_url, headers={"User-Agent": "Vettor/1.0"})
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            code = resp.status
            dur_ms = round((time.perf_counter() - t0) * 1000, 1)
            print(f"      Status: [PASS] HTTP {code} ({dur_ms}ms)")
    except Exception as exc:
        print(f"      Status: [WARN/FAIL] Could not reach target URL: {exc}")
        print("      Hint: If running local tests, start mock server: uv run python mockinsurancesite.py")
        all_passed = False

    # 2. Model Backend Check
    print(f"\n[2/5] Model Inference Backend: {cfg.model.backend}")
    if cfg.model.backend == "llamacpp":
        llama_bin = llm_manager.find_llama_executable(cfg.model.llama_bin)
        model_path = llm_manager.find_model_gguf(cfg.model.model_path)
        bin_ok = bool(llama_bin and Path(llama_bin).exists())
        model_ok = bool(model_path and Path(model_path).exists())
        print(f"      Llama binary: {'[PASS] ' + str(llama_bin) if bin_ok else '[FAIL] Not found'}")
        print(f"      Model file:   {'[PASS] ' + str(model_path) if model_ok else '[FAIL] Not found'}")
        is_live = llm_manager.is_server_alive(cfg.model.endpoint)
        print(f"      Server alive: {'[PASS] Active at ' + cfg.model.endpoint if is_live else '[INFO] Server idle (auto_start=' + str(cfg.model.auto_start_llama) + ')'}")
        if not (bin_ok and model_ok) and not is_live:
            all_passed = False
    elif cfg.model.backend in ("ollama", "openai"):
        print(f"      Endpoint: {cfg.model.endpoint}")
        print(f"      Model:    {cfg.model.model_name}")

    # 3. Browser Engine Check
    print(f"\n[3/5] Browser Engine: {cfg.browser.engine}")
    try:
        sync_pw, active_engine, chrom_exe = _get_playwright(cfg.browser.engine, cfg.browser.browser_type)
        print(f"      Active Engine: [PASS] {active_engine}")
        if chrom_exe:
            print(f"      Chromium Exe:  [PASS] {chrom_exe}")
    except Exception as exc:
        print(f"      Engine Status: [FAIL] {exc}")
        all_passed = False

    # 4. Test Suite Cases Check
    resolved_cases = cfg.get_resolved_cases_path()
    print(f"\n[4/5] Test Suite Cases: {resolved_cases}")
    if resolved_cases.exists():
        ext = resolved_cases.suffix.lower()
        check_cfg = QAConfig.from_dict(cfg.to_dict())
        check_cfg.model.auto_start_llama = False
        engine_dummy = QAEngine(check_cfg)
        cases = engine_dummy._load_test_cases()
        print(f"      Format: [PASS] {ext.upper()} ({len(cases)} test cases loaded)")
    else:
        print(f"      Status: [FAIL] Suite file not found at {resolved_cases}")
        all_passed = False

    # 5. Memory & Handoff Knowledge Base
    print(f"\n[5/5] Knowledge Base & Handoff Tracking")
    mem = QAMemory(cfg.reporting.memory_path)
    mem_stats = mem.summary_stats()
    print(f"      Memory Store:  [PASS] {mem_stats['total_records']} records ({mem_stats['facts']} facts, {mem_stats['cautions']} cautions) at {cfg.reporting.memory_path}")
    handoff_mgr = HandoffManager(cfg.reporting.handoff_path)
    open_items = handoff_mgr.list_open_items()
    print(f"      Handoff Track: [PASS] {len(open_items)} open items in {cfg.reporting.handoff_path}")

    print("\n" + "=" * 65)
    if all_passed:
        print("[SUCCESS] All pre-flight diagnostics PASSED. Ready to execute QA harness!")
    else:
        print("[NOTICE] Some pre-flight checks flagged warnings. Review items above.")
    print("=" * 65)


def cmd_handoff(cfg: QAConfig):
    """Display open issues and recommendations from DEVELOPMENT_HANDOFF.md."""
    handoff_mgr = HandoffManager(cfg.reporting.handoff_path)
    items = handoff_mgr.list_open_items()
    print("=" * 65)
    print(f"Development Handoff Tracker: {cfg.reporting.handoff_path}")
    print(f"Total Open Items: {len(items)}")
    print("=" * 65)

    if not items:
        print("No open issues or UX friction points currently recorded.")
        return

    for idx, item in enumerate(items, start=1):
        sev = item.get("severity", "P2")
        itype = item.get("type", "Bug")
        title = item.get("title", "Untitled")
        print(f"\n[{idx}] [{sev}] [{itype}] {title}")
        if "observed" in item:
            print(f"    Observed:      {item['observed']}")
        if "expected" in item:
            print(f"    Expected:      {item['expected']}")
        if "suggested fix" in item:
            print(f"    Suggested Fix: {item['suggested fix']}")
        if "evidence" in item:
            print(f"    Evidence:      {item['evidence']}")


def cmd_memory(args, cfg: QAConfig):
    """Inspect or manage the agent's persistent memory."""
    mem = QAMemory(cfg.reporting.memory_path)
    action = getattr(args, "memory_action", "inspect") or "inspect"

    if action == "clear":
        mem.clear()
        print(f"Cleared all memory records from {cfg.reporting.memory_path}")
        return

    stats = mem.summary_stats()
    print("=" * 65)
    print(f"Agent Procedural Memory: {cfg.reporting.memory_path}")
    print(f"Total Records: {stats['total_records']} | Facts: {stats['facts']} | Cautions: {stats['cautions']} | Avg Trust: {stats['avg_trust']}")
    print("=" * 65)

    if not mem.records:
        print("Memory store is empty. Records are learned automatically during test runs.")
        return

    for r in mem.records.values():
        badge = f"[{r.kind.upper()}] [{r.status}]"
        print(f"\n* {badge} {r.id} | Trust: {r.trust} (Confirmed: {r.confirmed}, Contradicted: {r.contradicted})")
        print(f"  Target:  {r.domain}{r.path}")
        print(f"  Summary: {r.summary}")
        if r.data:
            print(f"  Data:    {json.dumps(r.data)}")


def main():
    parser = argparse.ArgumentParser(
        description="Vettor: Autonomous Agentic Web QA Harness",
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Available subcommands")

    # Subcommand: init
    sub_init = subparsers.add_parser("init", help="Scaffold a new QA workspace")
    sub_init.add_argument("--force", "-f", action="store_true", help="Overwrite existing configuration and test files")

    # Subcommand: check
    subparsers.add_parser("check", help="Run pre-flight environment and readiness diagnostics")

    # Subcommand: handoff
    subparsers.add_parser("handoff", help="View development handoff issues and friction points")

    # Subcommand: memory
    sub_mem = subparsers.add_parser("memory", help="Inspect or clear persistent agent memory")
    sub_mem.add_argument("memory_action", nargs="?", choices=["inspect", "clear"], default="inspect", help="Memory action")

    # Common run flags (also supported as top-level arguments for backwards compatibility)
    for p in (parser, subparsers.add_parser("run", help="Run QA test suite")):
        p.add_argument("--config", "-c", default=None, help="Path to JSON configuration file")
        p.add_argument("--save-config", default=None, help="Save current configuration to JSON file and exit")
        p.add_argument("--url", default=None, help="Target starting URL")
        p.add_argument("--base-url", default=None, help="Target base URL")
        p.add_argument("--suite", "--csv", default=None, dest="suite", help="Path to test cases CSV/JSON/JSONL")
        p.add_argument("--mode", choices=["data_driven", "autonomous_goal", "crawl_smoke"], default=None, help="Execution mode")
        p.add_argument("--goal", default=None, help="Autonomous QA goal description")
        p.add_argument("--engine", choices=["rustwright", "playwright"], default=None, help="Browser engine")
        p.add_argument("--browser", choices=["chromium", "firefox", "webkit"], default=None, help="Browser type")
        p.add_argument("--headless", action="store_true", default=None, help="Run browser in headless mode")
        p.add_argument("--headed", action="store_false", dest="headless", help="Run browser in visible mode")
        p.add_argument("--slow-mo", type=int, default=None, help="Slow down browser interactions by N ms")
        p.add_argument("--backend", choices=["llamacpp", "ollama", "openai", "litellm", "auto", "heuristic"], default=None, help="Inference backend or model router")
        p.add_argument("--endpoint", default=None, help="Inference API endpoint")
        p.add_argument("--model", default=None, help="Model name")
        p.add_argument("--litellm-model", default=None, help="LiteLLM model id, e.g. openai/gpt-4o-mini, anthropic/claude-sonnet-4, ollama/llama3")
        p.add_argument("--api-base", default=None, help="API base URL for the LiteLLM model")
        p.add_argument("--fallback-model", default=None, help="LiteLLM model id used when the primary backend fails")
        p.add_argument("--provider-chain", default=None, help="Comma ordered provider chain, e.g. llamacpp,litellm")
        p.add_argument("--temperature", type=float, default=None, help="Model sampling temperature")
        p.add_argument("--max-tokens", type=int, default=None, help="Maximum completion tokens")
        p.add_argument("--model-path", default=None, help="Direct path to GGUF model file")
        p.add_argument("--api-key", default=None, help="API key for inference provider")
        p.add_argument("--cli", action="store_true", help="Run in headless CLI mode without interactive TUI")
        p.add_argument("--output-dir", default=None, help="Directory to save test reports and artifacts")
        p.add_argument("--memory-path", default=None, help="Path to qa_memory.json file")
        p.add_argument("--handoff-path", default=None, help="Path to DEVELOPMENT_HANDOFF.md file")
        p.add_argument("--app-context", default=None, help="Application lifecycle context and flow guidance for AI agent")

    args = parser.parse_args()

    # Route init directly before reading config
    if args.subcommand == "init":
        cmd_init(args)
        return

    # 1. Load config from file or default
    config_file = getattr(args, "config", None)
    if config_file:
        cfg = QAConfig.from_file(config_file)
    else:
        default_cfg = Path("harness_config.json")
        if default_cfg.exists():
            try:
                cfg = QAConfig.from_file(default_cfg)
            except Exception:
                cfg = QAConfig()
        else:
            cfg = QAConfig()

    # 2. Apply CLI overrides
    if getattr(args, "url", None):
        cfg.target.url = args.url
    if getattr(args, "base_url", None):
        cfg.target.base_url = args.base_url
    if getattr(args, "suite", None):
        cfg.suite.cases_path = args.suite
    if getattr(args, "mode", None):
        cfg.suite.mode = args.mode
    if getattr(args, "goal", None):
        cfg.suite.goal = args.goal
    if getattr(args, "engine", None):
        cfg.browser.engine = args.engine
    if getattr(args, "browser", None):
        cfg.browser.browser_type = args.browser
    if getattr(args, "headless", None) is not None:
        cfg.browser.headless = args.headless
    if getattr(args, "slow_mo", None) is not None:
        cfg.browser.slow_mo_ms = args.slow_mo
    if getattr(args, "backend", None):
        cfg.model.backend = args.backend
    if getattr(args, "endpoint", None):
        cfg.model.endpoint = args.endpoint
    if getattr(args, "model", None):
        cfg.model.model_name = args.model
    if getattr(args, "litellm_model", None):
        cfg.model.litellm_model = args.litellm_model
    if getattr(args, "api_base", None):
        cfg.model.litellm_api_base = args.api_base
    if getattr(args, "fallback_model", None):
        cfg.model.fallback_model = args.fallback_model
    if getattr(args, "provider_chain", None):
        cfg.model.provider_chain = [part.strip() for part in args.provider_chain.split(",") if part.strip()]
    if getattr(args, "temperature", None) is not None:
        cfg.model.temperature = args.temperature
    if getattr(args, "max_tokens", None) is not None:
        cfg.model.max_tokens = args.max_tokens
    if getattr(args, "model_path", None):
        cfg.model.model_path = args.model_path
    if getattr(args, "api_key", None):
        cfg.model.api_key = args.api_key
    if getattr(args, "output_dir", None):
        cfg.reporting.output_dir = args.output_dir
    if getattr(args, "memory_path", None):
        cfg.reporting.memory_path = args.memory_path
    if getattr(args, "handoff_path", None):
        cfg.reporting.handoff_path = args.handoff_path
    if getattr(args, "app_context", None):
        cfg.flow.app_context = args.app_context

    # Save config option
    if getattr(args, "save_config", None):
        cfg.save(args.save_config)
        print(f"Saved configuration to: {args.save_config}")
        return

    # Route subcommands
    if args.subcommand == "check":
        cmd_check(cfg)
        return
    elif args.subcommand == "handoff":
        cmd_handoff(cfg)
        return
    elif args.subcommand == "memory":
        cmd_memory(args, cfg)
        return

    # Default execution: CLI or TUI
    if getattr(args, "cli", False):
        engine = QAEngine(
            cfg,
            on_event=lambda e: print(f"[{e['type']}] {e.get('msg', e.get('result', ''))}")
            if e["type"] in ("log", "row_done", "done")
            else None,
        )
        engine.run_all()
    else:
        VettorTUI(cfg).run()


if __name__ == "__main__":
    main()
