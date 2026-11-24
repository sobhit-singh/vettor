from __future__ import annotations

import argparse
import asyncio
import sys
import time
import traceback
import urllib.request
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from vettor.config import VettorConfig
from vettor.core.llama_server import find_llama_executable, find_model_gguf, is_loopback_url, is_server_alive
from vettor.core.memory import QAMemory
from vettor.core.recipe import RecipeCache
from vettor.orchestrator import Orchestrator
from vettor.reporting.handoff import HANDOFF_TEMPLATE, HandoffManager

console = Console()


def _load_config(config_path: str | None = None) -> VettorConfig:
    if config_path:
        p = Path(config_path)
        if p.exists():
            try:
                return VettorConfig.from_file(p)
            except Exception as exc:
                console.print(f"[yellow]Warning: Could not parse {p} ({exc}), using defaults.[/yellow]")
    return VettorConfig.load_default()


def cmd_init(args: argparse.Namespace) -> int:
    """Scaffolds vettor.json, qa_memory.json, and DEVELOPMENT_HANDOFF.md."""
    console.print(Panel.fit("[bold cyan]Vettor: Initializing Workspace Scaffolding[/bold cyan]"))

    force = getattr(args, "force", False)

    cfg_path = Path("vettor.json")
    if not cfg_path.exists() or force:
        cfg = VettorConfig()
        cfg.save(cfg_path)
        console.print(f"[green]+[/green] Created configuration file: [bold]{cfg_path}[/bold]")
    else:
        console.print(f"[dim]• Configuration file already exists: {cfg_path}[/dim]")

    memory_path = Path("qa_memory.json")
    if not memory_path.exists() or force:
        mem = QAMemory(memory_path)
        mem.save()
        console.print(f"[green]+[/green] Initialized persistent memory: [bold]{memory_path}[/bold]")
    else:
        console.print(f"[dim]• Memory file already exists: {memory_path}[/dim]")

    handoff_path = Path("DEVELOPMENT_HANDOFF.md")
    if not handoff_path.exists() or force:
        handoff_path.write_text(HANDOFF_TEMPLATE, encoding="utf-8")
        console.print(f"[green]+[/green] Created handoff tracker: [bold]{handoff_path}[/bold]")
    else:
        console.print(f"[dim]• Handoff tracker already exists: {handoff_path}[/dim]")

    console.print("\n[bold green]✓ Workspace initialized successfully![/bold green]")
    console.print("Next steps:")
    console.print("  1. Run diagnostics:      [cyan]vettor check[/cyan]")
    console.print("  2. Start CLI run:         [cyan]vettor run --cli[/cyan]")
    console.print("  3. Start interactive TUI: [cyan]vettor run[/cyan]")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    """Runs pre-flight readiness diagnostics across target, model, browser, suite, and memory."""
    cfg = _load_config(getattr(args, "config", None))
    if getattr(args, "url", None):
        cfg.target.url = args.url
    if getattr(args, "model", None):
        cfg.model.active = args.model
    if getattr(args, "api_base", None):
        prov, _ = cfg.model.parse_active()
        if prov in cfg.model.providers:
            cfg.model.providers[prov].base_url = args.api_base

    console.print(Panel.fit("[bold cyan]Vettor: Pre-flight Readiness Diagnostics[/bold cyan]"))

    table = Table(title="System & Environment Status", show_header=True, header_style="bold magenta")
    table.add_column("Component", style="bold", width=24)
    table.add_column("Target / Resource", width=36)
    table.add_column("Status", width=16)

    all_ok = True

    # 1. Target URL
    target_url = cfg.target.url
    if not target_url:
        table.add_row("Target Application", "(none)", "[cyan]NOT CONFIGURED[/cyan]")
        console.print("[dim][INFO] No target URL configured. Set via --url or /target <url> in TUI.[/dim]")
    else:
        target_status = "[red]UNREACHABLE[/red]"
        try:
            req_headers = {"User-Agent": "Vettor/1.0", **cfg.target.auth_headers}
            req = urllib.request.Request(target_url, headers=req_headers)
            t0 = time.monotonic()
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                ms = round((time.monotonic() - t0) * 1000, 1)
                target_status = f"[green]PASS ({resp.status}, {ms}ms)[/green]"
        except Exception as exc:
            all_ok = False
            target_status = f"[yellow]FAIL ({type(exc).__name__})[/yellow]"
        table.add_row("Target Application", target_url, target_status)

    # 2. Model Backend / Provider
    provider_name, model_id = cfg.model.parse_active()
    p_cfg = cfg.model.providers.get(provider_name)

    if p_cfg and p_cfg.type == "llamacpp_local":
        is_loop = is_loopback_url(p_cfg.base_url)
        alive = is_server_alive(p_cfg.base_url, timeout=1.0, api_key=p_cfg.resolve_api_key())
        bin_path = find_llama_executable(cfg.model.llama_bin)
        model_gguf = find_model_gguf(cfg.model.model_path)
        if alive:
            model_status = f"[green]LIVE ({model_id})[/green]"
        elif is_loop and bin_path and model_gguf:
            model_status = "[cyan]READY TO LAUNCH[/cyan]"
        else:
            all_ok = False
            model_status = "[red]MISSING MODEL/BIN[/red]"
        table.add_row("Model Provider", f"{provider_name} ({p_cfg.base_url})", model_status)

    elif p_cfg and p_cfg.type == "litellm":
        try:
            import litellm  # noqa: F401
            model_status = f"[green]READY ({model_id})[/green]"
        except ImportError:
            all_ok = False
            model_status = "[red]MISSING LITELLM[/red]"
        table.add_row("Model Provider", f"litellm ({model_id})", model_status)

    elif p_cfg and p_cfg.type == "openai_compat":
        api_key = p_cfg.resolve_api_key()
        has_key = bool(api_key and api_key != "none")
        is_loop = is_loopback_url(p_cfg.base_url)
        if is_loop:
            alive = is_server_alive(p_cfg.base_url, timeout=1.0, api_key=api_key)
            model_status = f"[green]READY ({model_id})[/green]" if alive else f"[red]OFFLINE ({model_id})[/red]"
            if not alive:
                all_ok = False
        else:
            alive = is_server_alive(p_cfg.base_url, timeout=1.5, api_key=api_key) if not has_key else True
            model_status = f"[green]READY ({model_id})[/green]" if (alive or has_key) else f"[yellow]MISSING API KEY ({model_id})[/yellow]"
            if not (alive or has_key):
                all_ok = False
        table.add_row("Model Provider", f"{provider_name} ({model_id})", model_status)

    else:
        table.add_row("Model Provider", f"{provider_name} ({model_id})", "[green]CONFIGURED[/green]")

    # 3. Browser Engine
    browser_eng = cfg.browser.engine
    browser_status = "[green]INSTALLED[/green]"
    try:
        import playwright  # noqa: F401
    except ImportError:
        all_ok = False
        browser_status = "[red]MISSING PLAYWRIGHT[/red]"
    table.add_row("Browser Engine", f"{browser_eng} ({cfg.browser.browser_type})", browser_status)

    # 4. Test Suite
    cases_path = Path(cfg.suite.cases_path)
    suite_status = "[dim]NOT FOUND[/dim]"
    if cases_path.exists():
        try:
            sample_cases = Orchestrator(cfg).load_cases(cases_path)
            suite_status = f"[green]PASS ({len(sample_cases)} cases)[/green]"
        except Exception:
            suite_status = "[yellow]PARSE ERROR[/yellow]"
    else:
        suite_status = "[yellow]MISSING FILE[/yellow]"
    table.add_row("Test Suite", str(cases_path), suite_status)

    # 5. Memory & Cache
    mem_path = Path(cfg.reporting.memory_path)
    mem_cnt = 0
    if mem_path.exists():
        mem = QAMemory(mem_path)
        mem_cnt = len(mem.get_all())
    recipe_path = Path(cfg.reporting.recipe_path)
    recipe_cnt = 0
    if recipe_path.exists():
        rc = RecipeCache(recipe_path)
        recipe_cnt = len(rc.recipes)
    table.add_row("Memory & Recipes", f"Mem: {mem_cnt} | Recipes: {recipe_cnt}", "[green]READY[/green]")

    console.print(table)
    if all_ok:
        console.print("[bold green]✓ All core systems ready for test execution.[/bold green]")
        return 0
    else:
        console.print("[bold yellow]! Some checks failed or are unready. Review table above.[/bold yellow]")
        return 1


def cmd_run(args: argparse.Namespace) -> int:
    """Executes the test suite either in CLI mode or interactive TUI mode."""
    cfg = _load_config(getattr(args, "config", None))

    # Apply CLI overrides
    if getattr(args, "workers", None) is not None:
        cfg.browser.workers = args.workers
    if getattr(args, "url", None):
        cfg.target.url = args.url
    if getattr(args, "target", None):
        cfg.target.url = args.target
    if getattr(args, "model", None):
        cfg.model.active = args.model
    if getattr(args, "api_base", None):
        prov, _ = cfg.model.parse_active()
        if prov in cfg.model.providers:
            cfg.model.providers[prov].base_url = args.api_base
    if getattr(args, "suite", None):
        cfg.suite.cases_path = args.suite

    run_headless = getattr(args, "cli", False)

    if run_headless:
        if not cfg.target.url:
            console.print("[yellow]No target set. Specify --url <url> or run without --cli to use the TUI.[/yellow]")
            return 1

        console.print(f"[cyan]Starting Vettor CLI run against [bold]{cfg.target.url}[/bold] (workers={cfg.browser.workers})...[/cyan]")
        orchestrator = Orchestrator(cfg)
        summary = asyncio.run(orchestrator.run())

        table = Table(title="Run Summary", show_header=True, header_style="bold green")
        table.add_column("Case ID", width=12)
        table.add_column("Status", width=10)
        table.add_column("Duration", width=10)
        table.add_column("Steps", width=8)
        table.add_column("Cache Hits", width=10)
        table.add_column("Heals", width=8)
        table.add_column("Note", width=30)

        for res in summary.results:
            color = "green" if res.status == "PASS" else ("red" if res.status == "FAIL" else "yellow")
            table.add_row(
                res.case_id,
                f"[{color}]{res.status}[/{color}]",
                f"{res.duration_s:.2f}s",
                str(res.steps),
                str(res.cache_hits),
                str(res.heals),
                res.note[:30],
            )

        console.print(table)
        console.print(
            f"[bold]Total:[/bold] {summary.total} | "
            f"[green]Passed:[/green] {summary.passed} | "
            f"[red]Failed:[/red] {summary.failed} | "
            f"[yellow]Errors:[/yellow] {summary.errors} | "
            f"[cyan]Findings:[/cyan] {summary.findings} | "
            f"[dim]Duration: {summary.duration_s:.2f}s[/dim]"
        )
        return 0 if summary.failed == 0 and summary.errors == 0 else 1

    # TUI Mode (lazy import)
    try:
        from vettor.tui.app import VettorApp  # type: ignore[import-not-found]
        app = VettorApp(cfg)
        app.run()
        return 0
    except Exception as exc:
        console.print(f"[bold red]Failed to start TUI:[/bold red] {exc}")
        traceback.print_exc()
        return 1


def cmd_plan(args: argparse.Namespace) -> int:
    """Autonomous test suite generator from a target URL."""
    cfg = _load_config(getattr(args, "config", None))
    url = args.url
    out_file = Path(args.out)
    instruction = getattr(args, "instruction", "")

    console.print(f"[cyan]Planning test suite from [bold]{url}[/bold]...[/cyan]")
    orchestrator = Orchestrator(cfg)
    cases = asyncio.run(orchestrator.plan_suite(url, instruction=instruction))

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(c.model_dump_json() + "\n")

    console.print(f"[green]✓ Generated {len(cases)} test cases and saved to [bold]{out_file}[/bold][/green]")
    return 0


def cmd_handoff(args: argparse.Namespace) -> int:
    """Lists open items from the development handoff tracker."""
    cfg = _load_config(getattr(args, "config", None))
    handoff_path = getattr(args, "file", None) or cfg.reporting.handoff_path
    mgr = HandoffManager(handoff_path)
    items = mgr.list_open_items()

    console.print(Panel.fit(f"[bold cyan]Development Handoff ({handoff_path})[/bold cyan]"))
    if not items:
        console.print("[green]No open items found in handoff tracker.[/green]")
        return 0

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Severity", width=10)
    table.add_column("Title", width=40)
    table.add_column("Detail", width=50)

    for it in items:
        sev = it.get("severity", "P2")
        color = "red" if sev == "P0" else ("yellow" if sev == "P1" else "cyan")
        table.add_row(f"[{color}]{sev}[/{color}]", it.get("title", ""), it.get("detail", ""))

    console.print(table)
    return 0


def cmd_memory(args: argparse.Namespace) -> int:
    """Inspects or resets persistent QA memory."""
    cfg = _load_config(getattr(args, "config", None))
    action = getattr(args, "action", "inspect") or "inspect"
    memory_path = Path(cfg.reporting.memory_path)
    mem = QAMemory(memory_path)

    if action == "clear":
        mem.clear()
        console.print(f"[green]✓ Persistent memory cleared at {memory_path}[/green]")
        return 0

    records = mem.get_all()
    console.print(Panel.fit(f"[bold cyan]QA Memory Inspection ({memory_path})[/bold cyan]"))
    console.print(f"Total stored records: [bold]{len(records)}[/bold]")

    if records:
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Key / Page", width=30)
        table.add_column("Kind", width=12)
        table.add_column("Confirmed", width=10)
        table.add_column("Contradicted", width=12)
        table.add_column("Trust Score", width=12)

        for rec in records[:20]:
            table.add_row(
                rec.page_key[:30],
                rec.kind,
                str(rec.confirmed),
                str(rec.contradicted),
                f"{rec.trust:.2f}",
            )
        console.print(table)
        if len(records) > 20:
            console.print(f"[dim]... and {len(records) - 20} more records.[/dim]")

    return 0


def main(argv: list[str] | None = None) -> int:
    """Primary entrypoint for the vettor CLI."""
    parser = argparse.ArgumentParser(
        prog="vettor",
        description="Vettor: Model-Agnostic Autonomous Web QA Harness",
    )
    parser.add_argument("--config", "-c", help="Path to vettor.json configuration file")
    parser.add_argument("--cli", action="store_true", help="Run headless in CLI mode without TUI")
    parser.add_argument("--url", "-u", type=str, help="Target application URL")
    parser.add_argument("--target", "-t", type=str, help="Target application URL")
    parser.add_argument("--model", "-m", type=str, help="Active model string (e.g. openai/gpt-4o-mini)")
    parser.add_argument("--api-base", type=str, help="API base URL override")
    parser.add_argument("--workers", "-w", type=int, help="Number of concurrent browser workers")
    parser.add_argument("--suite", "-s", type=str, help="Test cases file path (.jsonl, .json, .csv)")

    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # init
    p_init = subparsers.add_parser("init", help="Scaffold a new Vettor workspace")
    p_init.add_argument("--force", "-f", action="store_true", help="Overwrite existing workspace files")

    # check
    p_check = subparsers.add_parser("check", help="Run pre-flight environment diagnostics")
    p_check.add_argument("--url", "-u", type=str, help="Target application URL")
    p_check.add_argument("--model", "-m", type=str, help="Active model string (e.g. openai/gpt-4o-mini)")
    p_check.add_argument("--api-base", type=str, help="API base URL override")

    # run
    p_run = subparsers.add_parser("run", help="Run the test suite")
    p_run.add_argument("--cli", action="store_true", help="Run headless in CLI mode without TUI")
    p_run.add_argument("--workers", "-w", type=int, help="Number of concurrent browser workers")
    p_run.add_argument("--target", "-t", type=str, help="Target application URL")
    p_run.add_argument("--url", "-u", type=str, help="Target application URL")
    p_run.add_argument("--model", "-m", type=str, help="Model name or active provider/model override")
    p_run.add_argument("--api-base", type=str, help="API base URL override")
    p_run.add_argument("--suite", "-s", type=str, help="Test cases file path (.jsonl, .json, .csv)")

    # plan
    p_plan = subparsers.add_parser("plan", help="Generate an autonomous test suite from a target URL")
    p_plan.add_argument("url", type=str, help="URL of the target application to explore")
    p_plan.add_argument("--out", "-o", type=str, default="cases.jsonl", help="Output path for generated suite")
    p_plan.add_argument("--instruction", "-i", type=str, default="", help="Focus instruction for the planner")

    # handoff
    p_handoff = subparsers.add_parser("handoff", help="Display open items in DEVELOPMENT_HANDOFF.md")
    p_handoff.add_argument("--file", type=str, help="Path to handoff tracker file")

    # memory
    p_mem = subparsers.add_parser("memory", help="Inspect or clear QA procedural memory")
    p_mem.add_argument("action", nargs="?", choices=["inspect", "clear"], default="inspect")

    parsed = parser.parse_args(argv)

    if parsed.command is None:
        # Default behavior: launch TUI (or headless CLI run if --cli passed)
        return cmd_run(parsed)

    dispatch = {
        "init": cmd_init,
        "check": cmd_check,
        "run": cmd_run,
        "plan": cmd_plan,
        "handoff": cmd_handoff,
        "memory": cmd_memory,
    }

    handler = dispatch.get(parsed.command)
    if handler:
        return handler(parsed)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
