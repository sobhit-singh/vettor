from __future__ import annotations

import re
import shlex
from typing import Any

from pydantic import BaseModel, Field

from vettor.config import ProviderConfig, VettorConfig
from vettor.core.llama_server import find_llama_executable, find_model_gguf, is_loopback_url, is_server_alive
from vettor.core.memory import QAMemory
from vettor.core.recipe import RecipeCache
from vettor.reporting.handoff import HandoffManager


class CommandResult(BaseModel):
    """Result of parsing and executing a slash command or autonomous goal."""

    handled: bool = True
    output: str = ""
    action: str = ""
    args: dict[str, Any] = Field(default_factory=dict)


HELP_TEXT = """[bold cyan]Available Commands & Controls:[/bold cyan]
  [green]/target <url>[/green]                             Set target web application URL
  [green]/run [suite_file] [--workers N] [--url URL][/green]  Run data-driven or existing test suite
  [green]/plan <url> [instruction...][/green]               Autonomously explore & generate a test suite
  [green]/model <provider/model> [--profile P][/green]        Change active model and profile
  [green]/provider add <name> <base_url> [key_env][/green]  Register a self-hosted or remote provider
  [green]/provider list[/green]                             List all configured model providers
  [green]/check[/green]                                     Run pre-flight environment diagnostics
  [green]/config [show|save][/green]                        Display or save current configuration
  [green]/workers <N>[/green]                                Change concurrent browser worker count
  [green]/visual [on|off][/green]                            Toggle visual screenshot auditing
  [green]/private [on|off][/green]                           Toggle confidential data mode
  [green]/headless [on|off][/green]                          Toggle headless browser execution
  [green]/handoff[/green]                                    View open findings from handoff tracker
  [green]/memory [clear][/green]                             Inspect or clear persistent QA memory
  [green]/clear[/green]                                     Clear deterministic recipe cache and log
  [green]/stop[/green]                                      Stop the actively running test suite
  [green]/help[/green]                                      Display this command reference
  [green]/quit[/green] or [green]/exit[/green]                              Exit Vettor terminal

[bold yellow]Natural Language Goals:[/bold yellow]
  Type any instruction without a leading slash to run an autonomous goal directly:
  [dim]> Test checkout flow on http://localhost:8000/shop with a $50 gift card[/dim]"""


def _run_diagnostics_text(cfg: VettorConfig) -> str:
    """Generates formatted pre-flight readiness text for display in the TUI transcript."""
    lines = ["[bold cyan]Pre-flight Readiness Diagnostics:[/bold cyan]"]

    # 1. Target URL
    if not cfg.target.url:
        lines.append("  • [cyan]Target Application:[/cyan] [yellow]NOT SET[/yellow] (Use /target <url>)")
    else:
        try:
            import urllib.request
            req = urllib.request.Request(cfg.target.url, headers={"User-Agent": "Vettor/1.0", **cfg.target.auth_headers})
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                lines.append(f"  • [cyan]Target Application:[/cyan] [green]PASS (HTTP {resp.status})[/green] {cfg.target.url}")
        except Exception as exc:
            lines.append(f"  • [cyan]Target Application:[/cyan] [yellow]WARN/FAIL ({type(exc).__name__})[/yellow] {cfg.target.url}")

    # 2. Model Provider
    provider_name, model_id = cfg.model.parse_active()
    p_cfg = cfg.model.providers.get(provider_name)
    if p_cfg and p_cfg.type == "llamacpp_local":
        is_loop = is_loopback_url(p_cfg.base_url)
        alive = is_server_alive(p_cfg.base_url, timeout=1.0, api_key=p_cfg.resolve_api_key())
        bin_ok = bool(find_llama_executable(cfg.model.llama_bin))
        model_ok = bool(find_model_gguf(cfg.model.model_path))
        if alive:
            status = f"[green]LIVE ({model_id})[/green]"
        elif is_loop and bin_ok and model_ok:
            status = f"[cyan]READY TO LAUNCH ({model_id})[/cyan]"
        else:
            status = "[red]MISSING BIN/MODEL[/red]"
        lines.append(f"  • [cyan]Model Provider:[/cyan] {provider_name} -> {status}")
    elif p_cfg and p_cfg.type == "litellm":
        try:
            import litellm  # noqa: F401
            lines.append(f"  • [cyan]Model Provider:[/cyan] litellm -> [green]READY ({model_id})[/green]")
        except ImportError:
            lines.append("  • [cyan]Model Provider:[/cyan] litellm -> [red]MISSING LITELLM PKG[/red]")
    elif p_cfg and p_cfg.type == "openai_compat":
        key = p_cfg.resolve_api_key()
        key_label = "API_KEY: SET" if bool(key and key != "none") else "NO_KEY"
        lines.append(f"  • [cyan]Model Provider:[/cyan] {provider_name} -> [green]READY ({model_id}, {key_label})[/green] ({p_cfg.base_url})")
    else:
        lines.append(f"  • [cyan]Model Provider:[/cyan] {provider_name} ({model_id}) -> [green]CONFIGURED[/green]")

    # 3. Browser
    try:
        import playwright  # noqa: F401
        lines.append(f"  • [cyan]Browser Engine:[/cyan] [green]READY[/green] ({cfg.browser.engine}, {cfg.browser.browser_type})")
    except ImportError:
        lines.append("  • [cyan]Browser Engine:[/cyan] [red]MISSING PLAYWRIGHT[/red]")

    return "\n".join(lines)


def parse_and_dispatch(raw_input: str, cfg: VettorConfig) -> CommandResult:
    """Parses and executes slash commands or dispatches an autonomous goal."""
    text = (raw_input or "").strip()
    if not text:
        return CommandResult(handled=False)

    # 1. Plain text -> Autonomous Goal
    if not text.startswith("/"):
        # Auto-extract URL if present in goal text and target is unset
        if not cfg.target.url:
            m = re.search(r"https?://[^\s\"'>]+", text)
            if m:
                cfg.target.url = m.group(0)

        cfg.suite.mode = "autonomous_goal"
        cfg.suite.goal = text
        return CommandResult(
            handled=True,
            output=f"[cyan]Target Goal:[/cyan] {text}\n[dim]Launching autonomous goal agent...[/dim]",
            action="run",
            args={"goal": text},
        )

    # 2. Slash command tokenization
    try:
        parts = shlex.split(text)
    except ValueError as exc:
        return CommandResult(handled=True, output=f"[red]Command syntax error: {exc}[/red]")

    cmd = parts[0].lower()
    args = parts[1:]

    # /help
    if cmd == "/help":
        return CommandResult(handled=True, output=HELP_TEXT)

    # /target <url>
    if cmd == "/target":
        if not args:
            cur = cfg.target.url or "(none)"
            return CommandResult(handled=True, output=f"Current Target URL: [bold]{cur}[/bold]")
        cfg.target.url = args[0].strip()
        return CommandResult(
            handled=True,
            output=f"[green]✓ Target URL set to [bold]{cfg.target.url}[/bold][/green]",
            action="target",
            args={"url": cfg.target.url},
        )

    # /provider [add|list]
    if cmd == "/provider":
        sub = args[0].lower() if args else "list"
        if sub == "list":
            active_prov, _ = cfg.model.parse_active()
            lines = ["[bold cyan]Configured Inference Providers:[/bold cyan]"]
            for name, p_cfg in cfg.model.providers.items():
                is_active = " [bold green][ACTIVE][/bold green]" if name == active_prov else ""
                lines.append(f"  • [bold]{name}[/bold]{is_active}: type={p_cfg.type}, url={p_cfg.base_url or 'n/a'}")
            return CommandResult(handled=True, output="\n".join(lines))
        if sub == "add":
            if len(args) < 3:
                return CommandResult(handled=True, output="[red]Usage: /provider add <name> <base_url> [api_key_env][/red]")
            p_name = args[1]
            p_url = args[2]
            p_env = args[3] if len(args) > 3 else ""
            cfg.model.providers[p_name] = ProviderConfig(
                type="openai_compat",
                base_url=p_url,
                api_key_env=p_env,
            )
            return CommandResult(
                handled=True,
                output=f"[green]✓ Registered provider [bold]{p_name}[/bold] ({p_url})[/green]",
            )
        return CommandResult(handled=True, output="[yellow]Usage: /provider list or /provider add <name> <base_url> [api_key_env][/yellow]")

    # /check
    if cmd == "/check":
        diag = _run_diagnostics_text(cfg)
        return CommandResult(handled=True, output=diag)

    # /config [show|save]
    if cmd == "/config":
        sub = args[0].lower() if args else "show"
        if sub == "save":
            cfg.save("vettor.json")
            return CommandResult(handled=True, output="[green]✓ Configuration saved to vettor.json[/green]")
        prov, model = cfg.model.parse_active()
        summary = (
            f"[bold cyan]Current Vettor Configuration:[/bold cyan]\n"
            f"  Target URL:  {cfg.target.url or '(none)'}\n"
            f"  Active Model: {cfg.model.active} (provider={prov}, model={model})\n"
            f"  Workers:     {cfg.browser.workers} (headless={cfg.browser.headless})\n"
            f"  Confidential: {cfg.model.confidential}\n"
            f"  Visual Audit: {cfg.model.enable_visual_audit}\n"
            f"  Suite Path:  {cfg.suite.cases_path}"
        )
        return CommandResult(handled=True, output=summary)

    # /quit, /exit
    if cmd in ("/quit", "/exit"):
        return CommandResult(handled=True, output="[dim]Exiting Vettor...[/dim]", action="quit")

    # /stop
    if cmd == "/stop":
        return CommandResult(handled=True, output="[yellow]Stopping active test run...[/yellow]", action="stop")

    # /clear
    if cmd == "/clear":
        rc = RecipeCache(cfg.reporting.recipe_path)
        rc.clear()
        return CommandResult(
            handled=True,
            output="[green]✓ Deterministic recipe cache cleared.[/green]",
            action="clear_log",
        )

    # /workers <N>
    if cmd == "/workers":
        if not args:
            return CommandResult(handled=True, output=f"Current workers: [bold]{cfg.browser.workers}[/bold]")
        try:
            val = max(1, int(args[0]))
            cfg.browser.workers = val
            return CommandResult(handled=True, output=f"[green]✓ Browser workers set to {val}[/green]")
        except ValueError:
            return CommandResult(handled=True, output="[red]Workers count must be a positive integer.[/red]")

    # /visual [on|off]
    if cmd == "/visual":
        if not args:
            cfg.model.enable_visual_audit = not cfg.model.enable_visual_audit
        elif args[0].lower() in ("on", "true", "1"):
            cfg.model.enable_visual_audit = True
        elif args[0].lower() in ("off", "false", "0"):
            cfg.model.enable_visual_audit = False
        state = "ENABLED" if cfg.model.enable_visual_audit else "DISABLED"
        return CommandResult(handled=True, output=f"[green]✓ Visual audit is now {state}[/green]")

    # /private [on|off]
    if cmd == "/private":
        if not args:
            cfg.model.confidential = not cfg.model.confidential
        elif args[0].lower() in ("on", "true", "1"):
            cfg.model.confidential = True
        elif args[0].lower() in ("off", "false", "0"):
            cfg.model.confidential = False
        state = "ENABLED" if cfg.model.confidential else "DISABLED"
        return CommandResult(handled=True, output=f"[green]✓ Confidential/Private mode is now {state}[/green]")

    # /headless [on|off]
    if cmd == "/headless":
        if not args:
            cfg.browser.headless = not cfg.browser.headless
        elif args[0].lower() in ("on", "true", "1"):
            cfg.browser.headless = True
        elif args[0].lower() in ("off", "false", "0"):
            cfg.browser.headless = False
        state = "HEADLESS" if cfg.browser.headless else "HEADED"
        return CommandResult(handled=True, output=f"[green]✓ Browser mode set to {state}[/green]")

    # /model <provider/model> [--profile P]
    if cmd == "/model":
        if not args:
            prov, m_id = cfg.model.parse_active()
            return CommandResult(
                handled=True,
                output=f"Active Model: [bold cyan]{cfg.model.active}[/bold cyan] (provider={prov}, model={m_id}, profile={cfg.model.profile})",
            )
        new_active = args[0]
        cfg.model.active = new_active
        if len(args) > 1:
            i = 1
            while i < len(args):
                if args[i] == "--profile" and i + 1 < len(args):
                    cfg.model.profile = args[i + 1]  # type: ignore[assignment]
                    i += 2
                else:
                    i += 1
        return CommandResult(
            handled=True,
            output=f"[green]✓ Model updated:[/green] [bold]{cfg.model.active}[/bold] (profile={cfg.model.profile})",
        )

    # /handoff
    if cmd == "/handoff":
        mgr = HandoffManager(cfg.reporting.handoff_path)
        items = mgr.list_open_items()
        if not items:
            return CommandResult(handled=True, output="[green]No open items in development handoff tracker.[/green]")
        lines = [f"[bold cyan]Open Handoff Items ({len(items)}):[/bold cyan]"]
        for it in items:
            sev = it.get("severity", "P2")
            color = "red" if sev in ("P0", "P1") else "yellow"
            lines.append(f"  [{color}]{sev}[/{color}] [bold]{it.get('title', '')}[/bold]: {it.get('detail', '')}")
        return CommandResult(handled=True, output="\n".join(lines))

    # /memory [clear]
    if cmd == "/memory":
        mem = QAMemory(cfg.reporting.memory_path)
        if args and args[0].lower() == "clear":
            mem.clear()
            return CommandResult(handled=True, output="[green]✓ Persistent QA memory cleared.[/green]")
        stats = mem.summary_stats()
        return CommandResult(
            handled=True,
            output=(
                f"[bold cyan]QA Procedural Memory:[/bold cyan]\n"
                f"  Total: {stats['total']} | Active: {stats['active']} | Deprecated: {stats['deprecated']}\n"
                f"  Facts: {stats['facts']} | Cautions: {stats['cautions']} | Procedures: {stats['procedures']}"
            ),
        )

    # /plan <url> [instruction...]
    if cmd == "/plan":
        if not args:
            return CommandResult(handled=True, output="[red]Usage: /plan <target_url> [instruction...][/red]")
        target_url = args[0]
        instruction = " ".join(args[1:]) if len(args) > 1 else ""
        return CommandResult(
            handled=True,
            output=f"[cyan]Planning suite for [bold]{target_url}[/bold]...[/cyan]",
            action="plan",
            args={"url": target_url, "instruction": instruction},
        )

    # /run [suite_path] [--workers N] [--url URL]
    if cmd == "/run":
        if not cfg.target.url and "--url" not in args:
            return CommandResult(
                handled=True,
                output="[yellow]No target set. Use /target <url> first.[/yellow]",
            )
        cfg.suite.mode = "data_driven"
        i = 0
        while i < len(args):
            token = args[i]
            if token == "--workers" and i + 1 < len(args):
                try:
                    cfg.browser.workers = max(1, int(args[i + 1]))
                except ValueError:
                    pass
                i += 2
            elif token == "--url" and i + 1 < len(args):
                cfg.target.url = args[i + 1]
                i += 2
            elif not token.startswith("-"):
                cfg.suite.cases_path = token
                i += 1
            else:
                i += 1
        return CommandResult(
            handled=True,
            output=f"[cyan]Running test suite [bold]{cfg.suite.cases_path}[/bold] against [bold]{cfg.target.url}[/bold] (workers={cfg.browser.workers})...[/cyan]",
            action="run",
        )

    return CommandResult(handled=False, output=f"[yellow]Unknown command: {cmd}. Type /help for assistance.[/yellow]")
