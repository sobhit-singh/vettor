import functools
import json
import logging
import os
from collections import deque
from datetime import datetime
from pathlib import Path
import time
import urllib.request
from typing import Any
from rich.markup import escape
from rich.text import Text
from textual import work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Footer, Header, Input, Label, ProgressBar, RichLog, Static

from config import QAConfig
from engine import QAEngine, RunStats
from mockinsurancesite import start_server_in_thread

logger = logging.getLogger("vettor.tui")


class TUIActionError(RuntimeError):
    def __init__(self, source: str, action: str, cause: BaseException):
        self.source = source
        self.action = action
        self.cause = cause
        super().__init__(f"{source} -> {action} failed: {type(cause).__name__}: {cause}")


ACTION_ROUTES = {
    "start_run": "UI -> action_start_run -> _run_worker -> QAEngine.run_all",
    "stop_run": "UI -> action_stop_run -> QAEngine.stop_requested",
    "toggle_engine": "UI -> action_toggle_engine -> QAConfig.engine -> BrowserConfig.engine",
    "toggle_backend": "UI -> action_toggle_backend -> QAConfig.llm_backend -> ModelConfig.backend",
    "toggle_headless": "UI -> action_toggle_headless -> QAConfig.headless -> BrowserConfig.headless",
    "toggle_mode": "UI -> action_toggle_mode -> SuiteConfig.mode",
    "clear_cache": "UI -> action_clear_cache -> RecipeCache.clear",
    "save_config": "UI -> action_save_config -> QAConfig.save",
    "ensure_mock_site": "UI -> action_ensure_mock_site -> start_server_in_thread",
    "show_trajectory": "UI -> action_show_trajectory -> TrajectoryRecorder.render",
}


class TrajectoryRecorder:
    def __init__(self, storage_path: Path, max_entries: int = 100):
        self.storage_path = storage_path
        self.entries: deque[dict[str, Any]] = deque(maxlen=max_entries)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, source: str, action: str, status: str, detail: str = "", duration_ms: float = 0.0) -> dict[str, Any]:
        entry = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "source": source,
            "action": action,
            "route": ACTION_ROUTES.get(action, "UI -> VettorTUI -> handler"),
            "status": status,
            "detail": str(detail)[:500],
            "duration_ms": round(duration_ms, 3),
        }
        self.entries.append(entry)
        try:
            with self.storage_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry) + "\n")
        except Exception as exc:
            logger.warning("Could not persist TUI trajectory entry: %s", exc)
        return entry

    def render(self) -> str:
        if not self.entries:
            return "[yellow]No TUI actions recorded yet.[/]"
        lines = [f"[bold cyan]TUI ACTION TRAJECTORY[/] ({len(self.entries)} recorded, newest last)"]
        for idx, entry in enumerate(self.entries, start=1):
            marker = {"ok": "[green]OK[/]", "error": "[red]ERROR[/]"}.get(entry["status"], "[yellow]REQUEST[/]")
            lines.append(f"  {idx:>2}. {entry['timestamp']} {marker} {entry['source']} -> {entry['action']}")
            lines.append(f"      route: {entry['route']}")
            if entry["detail"]:
                lines.append(f"      detail: {escape(entry['detail'])}")
        return "\n".join(lines)


def guarded_action(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        action = func.__name__.removeprefix("action_")
        source = getattr(self, "_pending_action_source", None) or "key"
        self._pending_action_source = None
        recorder = getattr(self, "trajectory", None)
        if recorder is not None:
            recorder.record(source, action, "request", "action invoked")
        started = time.perf_counter()
        try:
            result = func(self, *args, **kwargs)
        except Exception as exc:
            error = TUIActionError(source, action, exc)
            logger.exception("TUI action failed | source=%s action=%s", source, action)
            if recorder is not None:
                recorder.record(source, action, "error", str(error))
            self._notify_action_error(error)
            raise error from exc
        duration_ms = (time.perf_counter() - started) * 1000
        detail = self._action_state()
        if recorder is not None:
            recorder.record(source, action, "ok", detail, duration_ms)
        logger.info(
            "TUI action completed | source=%s action=%s duration_ms=%.2f state=%s",
            source,
            action,
            duration_ms,
            detail,
        )
        return result

    return wrapper


def is_url_reachable(url: str, timeout: float = 0.8) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Vettor/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status < 500
    except Exception:
        return False


class TUIEventLogHandler(logging.Handler):
    """Custom logging handler to pipe Python log records into the Textual UI safely."""

    def __init__(self, tui_app: "VettorTUI"):
        super().__init__()
        self.tui_app = tui_app

    def emit(self, record: logging.LogRecord):
        try:
            msg = self.format(record)
            if record.levelno >= logging.ERROR:
                formatted = f"[bold red]{escape(msg)}[/]"
            elif record.levelno >= logging.WARNING:
                formatted = f"[bold yellow]{escape(msg)}[/]"
            else:
                formatted = escape(msg)
            self.tui_app.call_from_thread(self.tui_app._safe_write, formatted)
        except Exception as exc:
            logging.getLogger("tui_internal").warning("Failed to relay log record to TUI: %s", exc)


class VettorTUI(App):
    TITLE = "Vettor: Autonomous Web QA Harness"
    SUB_TITLE = "Configurable, Domain-Agnostic Agentic Testing"

    CSS = """
    #kpi-bar {
        height: 3;
        padding: 0 1;
        background: $boost;
        content-align: left middle;
    }
    .kpi {
        width: 1fr;
        content-align: center middle;
        text-style: bold;
    }
    #config-panel {
        height: auto;
        padding: 0 1;
        background: $surface;
        border-bottom: solid $primary;
        margin-bottom: 1;
    }
    .config-row {
        height: auto;
        margin: 0;
        padding: 0;
    }
    .config-input {
        margin: 0 1 1 0;
    }
    #input-url {
        width: 3fr;
    }
    #input-suite {
        width: 2fr;
    }
    #input-context {
        width: 3fr;
    }
    #input-goal {
        width: 2fr;
    }
    #controls {
        height: auto;
        padding: 0 1;
        margin-bottom: 1;
    }
    .control-row {
        height: 3;
    }
    #controls Button {
        margin-right: 1;
        min-width: 14;
    }
    #main-split {
        height: 1fr;
    }
    #results-table {
        width: 3fr;
        height: 1fr;
        border: solid $primary;
    }
    #agent-log {
        width: 2fr;
        height: 1fr;
        border: solid $accent;
    }
    #progress {
        margin: 0 1 1 1;
    }
    """

    BINDINGS = [
        ("r", "start_run", "Run Suite"),
        ("s", "stop_run", "Stop"),
        ("e", "toggle_engine", "Toggle Engine"),
        ("b", "toggle_backend", "Toggle Backend"),
        ("h", "toggle_headless", "Toggle Headless"),
        ("m", "ensure_mock_site", "Mock Site"),
        ("M", "toggle_mode", "Toggle Mode"),
        ("c", "clear_cache", "Clear Cache"),
        ("t", "show_trajectory", "Trajectory"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, cfg: QAConfig):
        super().__init__()
        self.cfg = cfg
        self.engine: QAEngine | None = None
        self._suite_running = False
        self._pending_action_source: str | None = None
        self._log_handler: TUIEventLogHandler | None = None
        self.trajectory = TrajectoryRecorder(Path(cfg.reporting.output_dir) / "tui_trajectory.jsonl")

        log_file = Path("vettor.log")
        logging.basicConfig(
            filename=str(log_file),
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            encoding="utf-8",
        )

    def compose(self) -> ComposeResult:
        yield Header()

        # Top KPI Metrics Bar
        with Horizontal(id="kpi-bar"):
            yield Static("PASS: 0", id="kpi-pass", classes="kpi")
            yield Static("FAIL: 0", id="kpi-fail", classes="kpi")
            yield Static("ERR: 0", id="kpi-err", classes="kpi")
            yield Static("CACHE HITS: 0", id="kpi-cache", classes="kpi")
            yield Static("HEALS: 0", id="kpi-heals", classes="kpi")
            yield Static(f"ENGINE: {self.cfg.engine.upper()}", id="kpi-engine", classes="kpi")
            yield Static(f"MODEL: {self.cfg.llm_backend.upper()}", id="kpi-model", classes="kpi")

        # Interactive Config Editor Panel
        with Vertical(id="config-panel"):
            with Horizontal(classes="config-row"):
                yield Input(
                    value=self.cfg.start_url,
                    placeholder="Target Application URL (e.g. http://localhost:3000/flow)",
                    id="input-url",
                    classes="config-input",
                )
                yield Input(
                    value=self.cfg.suite.cases_path,
                    placeholder="Test Suite Cases File (CSV or JSONL)",
                    id="input-suite",
                    classes="config-input",
                )
            with Horizontal(classes="config-row"):
                yield Input(
                    value=self.cfg.flow.app_context,
                    placeholder="Application Lifecycle / Flow Context (e.g. E-commerce checkout funnel)",
                    id="input-context",
                    classes="config-input",
                )
                yield Input(
                    value=self.cfg.suite.goal,
                    placeholder="Autonomous QA Goal / Verification Objective",
                    id="input-goal",
                    classes="config-input",
                )

        # Dynamic Action Buttons
        with Vertical(id="controls"):
            with Horizontal(classes="control-row"):
                yield Button("▶ Run Suite (r)", id="btn-run", variant="success")
                yield Button("■ Stop (s)", id="btn-stop", variant="error", disabled=True)
                yield Button(f"Engine: {self.cfg.engine.upper()} (e)", id="btn-engine", variant="primary")
                yield Button(f"Backend: {self.cfg.llm_backend.upper()} (b)", id="btn-backend", variant="default")
                yield Button(f"Browser: {'Headless' if self.cfg.headless else 'Headed'} (h)", id="btn-headless", variant="default")
            with Horizontal(classes="control-row"):
                yield Button(f"Mode: {self.cfg.suite.mode} (Shift+M)", id="btn-mode", variant="default")
                yield Button("⌫ Clear Cache (c)", id="btn-cache", variant="warning")
                yield Button("💾 Save Config", id="btn-save", variant="default")
                yield Button("⚡ Mock Site", id="btn-mock", variant="default")
                yield Button("Trajectory (t)", id="btn-trajectory", variant="default")

        yield ProgressBar(total=100, show_eta=False, id="progress")

        with Horizontal(id="main-split"):
            yield DataTable(id="results-table")
            yield RichLog(id="agent-log", highlight=True, markup=True)

        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#results-table", DataTable)
        table.cursor_type = "row"
        table.zebra_stripes = True
        table.add_columns("Case ID", "Status", "Actual", "Expected", "Time", "Details / Note")
        table.focus()

        resolved_suite = self.cfg.get_resolved_cases_path()
        self._safe_write(
            f"[bold cyan]Universal Web QA Harness Ready.[/]\n"
            f"  Target:  [bold]{self.cfg.start_url}[/]\n"
            f"  Suite:   [bold]{resolved_suite}[/]\n"
            f"  Engine:  [bold]{self.cfg.engine}[/] | Headless: [bold]{self.cfg.headless}[/]\n"
            f"  Backend: [bold]{self.cfg.llm_backend}[/] ({self.cfg.local_model})\n"
            f"  Context: [italic]{self.cfg.flow.app_context or 'Universal multi-step workflow'}[/]"
        )

        self._log_handler = TUIEventLogHandler(self)
        logging.getLogger("vettor").addHandler(self._log_handler)

    def _safe_write(self, msg: str) -> None:
        logger.debug("TUI output: %s", msg)
        try:
            log = self.query_one("#agent-log", RichLog)
            try:
                log.write(Text.from_markup(msg))
            except Exception as exc:
                logger.debug("Markup rendering failed, writing plain text: %s", exc)
                log.write(Text(str(msg)))
        except Exception as exc:
            logging.getLogger("tui_internal").error("Could not write to TUI log: %s", exc)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        iid = event.input.id
        val = event.value.strip()
        if iid == "input-url":
            self.cfg.target.url = val
            self._safe_write(f"[bold green]Updated Target URL:[/] {val}")
        elif iid == "input-suite":
            self.cfg.suite.cases_path = val
            resolved = self.cfg.get_resolved_cases_path()
            status = f"[green]found ({resolved})[/]" if resolved.exists() else "[red]file not found[/]"
            self._safe_write(f"[bold green]Updated Test Suite Path:[/] {val} -> {status}")
        elif iid == "input-context":
            self.cfg.flow.app_context = val
            self._safe_write(f"[bold green]Updated Application Context:[/] {val}")
        elif iid == "input-goal":
            self.cfg.suite.goal = val
            self._safe_write(f"[bold green]Updated QA Goal:[/] {val}")

    def _sync_inputs_to_config(self) -> None:
        try:
            url_val = self.query_one("#input-url", Input).value.strip()
            if url_val:
                self.cfg.target.url = url_val

            suite_val = self.query_one("#input-suite", Input).value.strip()
            if suite_val:
                self.cfg.suite.cases_path = suite_val

            context_val = self.query_one("#input-context", Input).value.strip()
            if context_val:
                self.cfg.flow.app_context = context_val

            goal_val = self.query_one("#input-goal", Input).value.strip()
            if goal_val:
                self.cfg.suite.goal = goal_val
        except Exception:
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        routes = {
            "btn-run": "start_run",
            "btn-stop": "stop_run",
            "btn-engine": "toggle_engine",
            "btn-backend": "toggle_backend",
            "btn-headless": "toggle_headless",
            "btn-mode": "toggle_mode",
            "btn-cache": "clear_cache",
            "btn-save": "save_config",
            "btn-mock": "ensure_mock_site",
            "btn-trajectory": "show_trajectory",
        }
        action = routes.get(bid)
        if action is None:
            error = TUIActionError(
                f"button:{bid}", "unknown_button", ValueError("No handler is mapped to this button")
            )
            logger.error("Unmapped TUI button pressed: %s", bid)
            self._notify_action_error(error)
            return
        self._pending_action_source = f"button:{bid}"
        try:
            getattr(self, f"action_{action}")()
        except TUIActionError:
            pass

    @guarded_action
    def action_toggle_engine(self) -> None:
        if self._suite_running:
            self._safe_write("[yellow]Engine is locked while a suite is running.[/]")
            return
        before = self.cfg.engine
        self.cfg.engine = "playwright" if before == "rustwright" else "rustwright"
        self.query_one("#btn-engine", Button).label = f"Engine: {self.cfg.engine.upper()} (e)"
        self.query_one("#kpi-engine", Static).update(f"ENGINE: {self.cfg.engine.upper()}")
        self._safe_write(f"[yellow]Switched browser engine: {before} -> [bold]{self.cfg.engine}[/][/]")

    @guarded_action
    def action_toggle_backend(self) -> None:
        if self._suite_running:
            self._safe_write("[yellow]Backend is locked while a suite is running.[/]")
            return
        backends = ["llamacpp", "ollama", "openai", "litellm", "auto", "heuristic"]
        current_idx = backends.index(self.cfg.llm_backend) if self.cfg.llm_backend in backends else 0
        before = self.cfg.llm_backend
        next_backend = backends[(current_idx + 1) % len(backends)]
        self.cfg.llm_backend = next_backend
        self.query_one("#btn-backend", Button).label = f"Backend: {next_backend.upper()} (b)"
        self.query_one("#kpi-model", Static).update(f"MODEL: {next_backend.upper()}")
        self._safe_write(f"[yellow]Switched LLM backend: {before} -> [bold]{next_backend}[/][/]")

    @guarded_action
    def action_toggle_headless(self) -> None:
        if self._suite_running:
            self._safe_write("[yellow]Browser mode is locked while a suite is running.[/]")
            return
        before = self.cfg.headless
        self.cfg.headless = not before
        state_str = "Headless" if self.cfg.headless else "Headed"
        self.query_one("#btn-headless", Button).label = f"Browser: {state_str} (h)"
        self._safe_write(f"[yellow]Browser mode: {before} -> [bold]{state_str}[/][/]")

    @guarded_action
    def action_toggle_mode(self) -> None:
        if self._suite_running:
            self._safe_write("[yellow]Execution mode is locked while a suite is running.[/]")
            return
        modes = ["data_driven", "autonomous_goal", "crawl_smoke"]
        current_idx = modes.index(self.cfg.suite.mode) if self.cfg.suite.mode in modes else 0
        before = self.cfg.suite.mode
        next_mode = modes[(current_idx + 1) % len(modes)]
        self.cfg.suite.mode = next_mode
        self.query_one("#btn-mode", Button).label = f"Mode: {next_mode} (Shift+M)"
        self._safe_write(f"[yellow]Execution mode: {before} -> [bold]{next_mode}[/][/]")

    @guarded_action
    def action_clear_cache(self) -> None:
        if self._suite_running:
            self._safe_write("[yellow]Cache clear is locked while a suite is running.[/]")
            return
        from cache import RecipeCache

        RecipeCache(self.cfg.recipe_path).clear()
        self._safe_write(f"[yellow]Cleared recipe cache ({self.cfg.recipe_path}).[/]")

    @guarded_action
    def action_save_config(self) -> None:
        self._sync_inputs_to_config()
        cfg_path = Path("harness_config.json")
        self.cfg.save(cfg_path)
        self._safe_write(f"[bold green]Saved current configuration to {cfg_path}[/]")

    @guarded_action
    def action_ensure_mock_site(self) -> None:
        if is_url_reachable(self.cfg.start_url):
            self._safe_write(f"[green]Target portal is active at {self.cfg.start_url}[/]")
            return
        self._safe_write("[yellow]Starting local mock portal on port 8000...[/]")
        httpd, _ = start_server_in_thread(host="127.0.0.1", port=8000)
        if not is_url_reachable("http://127.0.0.1:8000/step1"):
            raise RuntimeError(
                f"Mock server bound to {httpd.server_address} but /step1 is unreachable"
            )
        self._safe_write("[bold green]Mock portal active on http://127.0.0.1:8000/step1![/]")

    @guarded_action
    def action_stop_run(self) -> None:
        if self.engine is None:
            raise RuntimeError("No QA run is active; there is nothing to stop")
        self.engine.stop_requested = True
        self._safe_write("[bold yellow]Stop requested. Halting after current case completes...[/]")
        self.query_one("#btn-stop", Button).disabled = True

    @guarded_action
    def action_show_trajectory(self) -> None:
        report = self.trajectory.render()
        self._safe_write(report)
        logger.info("Trajectory requested; %d entries in recorder", len(self.trajectory.entries))

    @guarded_action
    def action_start_run(self) -> None:
        if self._suite_running:
            self._safe_write("[yellow]A suite is already running; press 's' to stop it first.[/]")
            return

        self._sync_inputs_to_config()
        resolved_cases = self.cfg.get_resolved_cases_path()
        if not resolved_cases.exists():
            raise FileNotFoundError(
                f"Test suite not found: {self.cfg.suite.cases_path} (resolved: {resolved_cases})"
            )

        self._suite_running = True
        self.query_one("#btn-run", Button).disabled = True
        self.query_one("#btn-stop", Button).disabled = False
        self.query_one("#btn-engine", Button).disabled = True
        self.query_one("#btn-backend", Button).disabled = True
        self.query_one("#btn-headless", Button).disabled = True
        self.query_one("#btn-mode", Button).disabled = True
        self.query_one("#btn-cache", Button).disabled = True
        self.query_one("#results-table", DataTable).clear()

        self.query_one("#kpi-pass", Static).update("PASS: 0")
        self.query_one("#kpi-fail", Static).update("FAIL: 0")
        self.query_one("#kpi-err", Static).update("ERR: 0")
        self.query_one("#kpi-cache", Static).update("CACHE HITS: 0")
        self.query_one("#kpi-heals", Static).update("HEALS: 0")

        self._run_worker()

    @work(thread=True)
    def _run_worker(self) -> None:
        self.trajectory.record(
            "worker", "run_worker", "request",
            f"target={self.cfg.start_url} mode={self.cfg.suite.mode} engine={self.cfg.engine}",
        )
        logger.info("QA worker starting | target=%s mode=%s engine=%s", self.cfg.start_url, self.cfg.suite.mode, self.cfg.engine)
        try:
            # Auto-start mock site ONLY if localhost:8000 is explicitly targeted
            if ("8000" in self.cfg.start_url or "localhost:8000" in self.cfg.start_url) and not is_url_reachable(self.cfg.start_url):
                self.call_from_thread(
                    self._safe_write,
                    "[yellow]Target is localhost:8000 and unreachable. Auto-launching local mock server...[/]",
                )
                start_server_in_thread(host="127.0.0.1", port=8000)
                time.sleep(0.6)

            self.call_from_thread(
                self._safe_write,
                f"[bold cyan]Launching QA Run...[/]\n"
                f"  Target:  [bold]{self.cfg.start_url}[/]\n"
                f"  Engine:  [bold]{self.cfg.engine}[/] | Headless: [bold]{self.cfg.headless}[/]\n"
                f"  Backend: [bold]{self.cfg.llm_backend}[/] ({self.cfg.local_model})\n"
                f"  Context: [italic]{self.cfg.flow.app_context or 'Universal multi-step workflow'}[/]",
            )
            self.engine = QAEngine(self.cfg, on_event=lambda ev: self.call_from_thread(self._handle_event, ev))
            self.engine.run_all()
        except Exception as exc:
            logger.exception("Fatal exception in QA worker thread | %s", exc)
            self.trajectory.record("worker", "run_worker", "error", str(exc))
            self.call_from_thread(self._log_error, str(exc))
        else:
            stats = self.engine.stats if self.engine else None
            self.trajectory.record("worker", "run_worker", "ok", f"stats={stats}" if stats else "no stats")
            logger.info("QA worker finished | stats=%s", stats)
        finally:
            self.call_from_thread(self._on_finished)

    def _action_state(self) -> str:
        return (
            f"engine={self.cfg.engine}, backend={self.cfg.llm_backend}, "
            f"headless={self.cfg.headless}, mode={self.cfg.suite.mode}"
        )

    def _notify_action_error(self, error: TUIActionError) -> None:
        logger.error(
            "TUI action error | source=%s action=%s cause=%s: %s",
            error.source,
            error.action,
            type(error.cause).__name__,
            error.cause,
        )
        detail = escape(f"{error.source} -> {error.action}: {type(error.cause).__name__}: {error.cause}")
        self._safe_write(f"[bold red]Action failed:[/] {detail}")
        self._safe_write("[dim]Full traceback is in vettor.log; press 't' for the action trajectory.[/]")

    def _log_error(self, msg: str) -> None:
        logger.error("QA worker failed: %s", msg)
        self._safe_write(f"[bold red]Fatal Engine Error:[/] {escape(msg)}")

    def _on_finished(self) -> None:
        self._suite_running = False
        self.query_one("#btn-run", Button).disabled = False
        self.query_one("#btn-stop", Button).disabled = True
        self.query_one("#btn-engine", Button).disabled = False
        self.query_one("#btn-backend", Button).disabled = False
        self.query_one("#btn-headless", Button).disabled = False
        self.query_one("#btn-mode", Button).disabled = False
        self.query_one("#btn-cache", Button).disabled = False

    def _handle_event(self, ev: dict) -> None:
        logger.debug("Engine event received: %s", ev)
        try:
            table = self.query_one("#results-table", DataTable)
            bar = self.query_one("#progress", ProgressBar)

            etype = ev.get("type", "")
            if etype == "log":
                self._safe_write(str(ev.get("msg", "")))

            elif etype == "init":
                rows = ev.get("rows", [])
                bar.update(total=max(len(rows), 1), progress=0)

            elif etype == "row_start":
                self._safe_write(f"[dim]Running case {ev.get('case_id')}...[/]")

            elif etype == "row_done":
                res = ev.get("result", {})
                st: RunStats | None = ev.get("stats")
                status = res.get("status", "UNKNOWN")
                color = {"PASS": "green", "FAIL": "red", "ERROR": "yellow"}.get(status, "white")

                act_val = res.get("actual", "")
                exp_val = res.get("expected", "")
                act_str = f"${float(act_val):.2f}" if isinstance(act_val, (int, float)) and act_val != "" else str(act_val)
                exp_str = f"${float(exp_val):.2f}" if isinstance(exp_val, (int, float)) and exp_val != "" else str(exp_val)
                dur_str = f"{float(res.get('duration_s', 0.0)):.2f}s"

                table.add_row(
                    str(res.get("case_id", "")),
                    f"[{color}]{status}[/]",
                    act_str,
                    exp_str,
                    dur_str,
                    escape(str(res.get("note", ""))),
                )
                bar.advance(1)

                if st:
                    self.query_one("#kpi-pass", Static).update(f"[green]PASS: {st.passed}[/]")
                    self.query_one("#kpi-fail", Static).update(f"[red]FAIL: {st.failed}[/]")
                    self.query_one("#kpi-err", Static).update(f"[yellow]ERR: {st.errors}[/]")
                    self.query_one("#kpi-cache", Static).update(f"CACHE HITS: {st.cache_hits}")
                    self.query_one("#kpi-heals", Static).update(f"HEALS: {st.heals}")

            elif etype == "done":
                st = ev.get("stats")
                summary = (
                    f"[bold green]Finished suite![/] Total: {st.total} | Passed: {st.passed} | Failed: {st.failed} | Errors: {st.errors}"
                    if st
                    else "[bold green]Finished suite![/]"
                )
                self._safe_write(summary)
                self._safe_write(f"Results exported to [bold]{self.cfg.results_path}[/]")

        except Exception as exc:
            logger.exception("Error processing TUI event: %s", exc)
            self.trajectory.record("engine_event", str(ev.get("type", "unknown")), "error", str(exc))
