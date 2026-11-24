from __future__ import annotations

import asyncio
import re
import threading
from typing import Any

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.events import Key
from textual.widgets import Input

from vettor.config import VettorConfig
from vettor.core.events import (
    ActionTaken,
    CaseFinished,
    CaseStarted,
    Event,
    EventBus,
    FindingLogged,
    LogMessage,
    RunFinished,
    RunStarted,
)
from vettor.core.schema import CaseResult, Finding
from vettor.orchestrator import Orchestrator
from vettor.tui.commands import parse_and_dispatch
from vettor.tui.widgets import HITLBar, TelemetryBar, TranscriptLog


class VettorApp(App):
    """Developer-first terminal QA interface for autonomous web testing."""

    CSS = """
    Screen {
        layout: vertical;
        background: #020617;
        color: #e2e8f0;
    }

    #main-container {
        height: 1fr;
        layout: vertical;
    }

    #command-input {
        dock: bottom;
        height: 3;
        background: #0f172a;
        color: #f8fafc;
        border: solid #334155;
        padding: 0 1;
    }

    #command-input:focus {
        border: double #38bdf8;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("ctrl+q", "quit", "Quit", show=False),
        Binding("escape", "cancel_hitl", "Cancel HITL", show=False),
    ]

    def __init__(
        self,
        cfg: VettorConfig | None = None,
        bus: EventBus | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.cfg = cfg or VettorConfig()
        self.bus = bus or EventBus()
        self.orchestrator: Orchestrator | None = None
        self._hitl_future: asyncio.Future[dict[str, Any] | None] | None = None
        self._history: list[str] = []
        self._history_idx: int = -1

        # Register event bus listener
        self.bus.subscribe(self._on_bus_event)

    def compose(self) -> ComposeResult:
        self.telemetry = TelemetryBar(
            url=self.cfg.target.url or "(no target)",
            model=self.cfg.model.active,
            profile=self.cfg.model.profile,
            id="telemetry-bar",
        )
        self.transcript = TranscriptLog(id="transcript-log")
        self.hitl_bar = HITLBar(id="hitl-bar")
        self.command_input = Input(
            id="command-input",
            placeholder="> /run, /plan <url>, or type a QA goal...  [/help]",
        )

        yield self.telemetry
        with Vertical(id="main-container"):
            yield self.transcript
        yield self.hitl_bar
        yield self.command_input

    def on_mount(self) -> None:
        self.command_input.focus()
        target_display = self.cfg.target.url or "(no target)"
        self.transcript.write(
            "[bold cyan]Vettor Terminal Agent Engine v0.3.0[/bold cyan]\n"
            f"[dim]Connected to {target_display} using {self.cfg.model.active} ({self.cfg.model.profile})[/dim]\n"
            "[dim]Type [/dim][bold green]/help[/bold green][dim] for commands, or type any goal to execute.[/dim]\n"
        )

    def action_cancel_hitl(self) -> None:
        """Handles Esc key to cancel or skip an active human-in-the-loop prompt."""
        if self._hitl_future and not self._hitl_future.done():
            self._hitl_future.set_result(None)
            self.hitl_bar.deactivate()
            self.transcript.log_system("HITL prompt skipped by user.")

    def on_key(self, event: Key) -> None:
        """Navigates command history using Up and Down arrows when input is focused."""
        if not self.command_input.has_focus:
            return

        if event.key == "up":
            if self._history and self._history_idx < len(self._history) - 1:
                self._history_idx += 1
                cmd = self._history[len(self._history) - 1 - self._history_idx]
                self.command_input.value = cmd
                self.command_input.cursor_position = len(cmd)
                event.prevent_default()
        elif event.key == "down":
            if self._history_idx > 0:
                self._history_idx -= 1
                cmd = self._history[len(self._history) - 1 - self._history_idx]
                self.command_input.value = cmd
                self.command_input.cursor_position = len(cmd)
                event.prevent_default()
            elif self._history_idx == 0:
                self._history_idx = -1
                self.command_input.value = ""
                event.prevent_default()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Dispatches command input or resolves active HITL request."""
        val = event.value.strip()
        self.command_input.value = ""

        # 1. Active HITL Input Resolution
        if self._hitl_future and not self._hitl_future.done():
            res: dict[str, Any] = {}
            if re.match(r"^vid:\d+=", val, re.IGNORECASE):
                parts = val.split("=", 1)
                vid_num = int(re.sub(r"[^\d]", "", parts[0]))
                res = {"vid": vid_num, "value": parts[1].strip()}
            elif "=" in val and (val.startswith("#") or val.startswith(".") or val.startswith("[")):
                parts = val.split("=", 1)
                res = {"selector": parts[0].strip(), "value": parts[1].strip()}
            else:
                res = {"hint": val}

            self.transcript.log_system(f"Human hint provided: {val}")
            self.hitl_bar.deactivate()
            self._hitl_future.set_result(res)
            return

        if not val:
            return

        # 2. Append to history
        self._history.append(val)
        self._history_idx = -1

        # 3. Echo command to transcript
        self.transcript.write(f"\n[bold white]>[/bold white] [bold cyan]{val}[/bold cyan]")

        # 4. Parse & dispatch
        cmd_result = parse_and_dispatch(val, self.cfg)
        if cmd_result.output:
            self.transcript.write(cmd_result.output)

        # Refresh telemetry display in case model/target changed
        self.telemetry.update_stats(
            url=self.cfg.target.url or "(no target)",
            model=self.cfg.model.active,
            profile=self.cfg.model.profile,
        )

        # 5. Handle action
        if cmd_result.action == "quit":
            self.exit()
        elif cmd_result.action == "clear_log":
            self.transcript.clear()
        elif cmd_result.action == "stop":
            if self.orchestrator:
                self.orchestrator.stop_requested = True
                self.transcript.log_system("Cancellation requested.")
        elif cmd_result.action == "run":
            if not self.cfg.target.url:
                self.transcript.log_system("No target set. Use /target <url> first.")
            else:
                self.run_orchestrator()
        elif cmd_result.action == "plan":
            target_url = cmd_result.args.get("url", self.cfg.target.url)
            instruction = cmd_result.args.get("instruction", "")
            self.run_plan(target_url, instruction)

    async def _hitl_bridge(self, context: dict[str, Any]) -> dict[str, Any] | None:
        """Async bridge providing non-modal, inline HITL prompts directly above the command line."""
        case_id = str(context.get("case_id", ""))
        obs = context.get("obs")
        url = obs.url if obs else ""
        reason = str(context.get("reason", "workflow stalled"))

        self._hitl_future = asyncio.get_running_loop().create_future()
        self.hitl_bar.activate(case_id, url=url, reason=reason)
        self.command_input.focus()

        try:
            return await self._hitl_future
        finally:
            self._hitl_future = None
            self.hitl_bar.deactivate()

    @work(exclusive=True)
    async def run_orchestrator(self) -> None:
        """Executes the test suite in a non-blocking background async Textual worker."""
        self.orchestrator = Orchestrator(
            cfg=self.cfg,
            bus=self.bus,
            hitl_callback=self._hitl_bridge,
        )
        try:
            summary = await self.orchestrator.run()
            self.transcript.log_system(
                f"Suite completed: {summary.total} cases ({summary.passed} passed, "
                f"{summary.failed} failed, {summary.errors} errors) in {summary.duration_s:.2f}s"
            )
        except Exception as exc:
            self.transcript.log_error(f"Orchestration failure: {exc}")

    @work(exclusive=True)
    async def run_plan(self, url: str, instruction: str) -> None:
        """Plans a test suite in a non-blocking background async Textual worker."""
        self.orchestrator = Orchestrator(cfg=self.cfg, bus=self.bus)
        try:
            cases = await self.orchestrator.plan_suite(url, instruction=instruction)
            self.transcript.log_system(f"Successfully generated {len(cases)} test cases.")
        except Exception as exc:
            self.transcript.log_error(f"Planning failure: {exc}")

    def _on_bus_event(self, event: Event) -> None:
        """Handles incoming async events from the EventBus and routes to widgets."""
        if self._thread_id == threading.get_ident():
            self._handle_event(event)
        else:
            self.call_from_thread(self._handle_event, event)

    def _handle_event(self, event: Event) -> None:
        if isinstance(event, RunStarted):
            self.telemetry.update_stats(
                total_cases=event.total_cases,
                passed=0,
                failed=0,
                errors=0,
                findings=0,
                cache_hits=0,
            )
            self.transcript.log_system(f"Run started: {event.total_cases} total test cases queued.")

        elif isinstance(event, CaseStarted):
            self.transcript.log_case_start(event.case_id, name=event.name, goal=event.goal)

        elif isinstance(event, ActionTaken):
            action_obj = event.action
            action_type = getattr(action_obj, "action", "") if not isinstance(action_obj, dict) else action_obj.get("action", "")
            action_val = getattr(action_obj, "value", "") if not isinstance(action_obj, dict) else action_obj.get("value", "")
            action_vid = getattr(action_obj, "vid", None) if not isinstance(action_obj, dict) else action_obj.get("vid")
            vid_str = f"vid={action_vid} " if action_vid is not None else ""
            val_str = f"'{action_val}'" if action_val else ""
            detail = f"{vid_str}{val_str}".strip()
            self.transcript.log_action(event.case_id, mode="LLM", detail=f"{action_type} {detail}")

        elif isinstance(event, FindingLogged):
            finding_obj = event.finding
            if isinstance(finding_obj, dict):
                finding_obj = Finding.model_validate(finding_obj)
            self.transcript.log_finding(finding_obj)
            self.telemetry.findings += 1
            self.telemetry.refresh_display()

        elif isinstance(event, CaseFinished):
            res_obj = event.result
            if isinstance(res_obj, dict):
                res_obj = CaseResult.model_validate(res_obj)

            if res_obj.status == "PASS":
                self.telemetry.passed += 1
            elif res_obj.status == "FAIL":
                self.telemetry.failed += 1
            else:
                self.telemetry.errors += 1
            self.telemetry.cache_hits += res_obj.cache_hits
            self.telemetry.refresh_display()
            self.transcript.log_case_done(res_obj)

        elif isinstance(event, RunFinished):
            self.transcript.log_system("Execution run completed.")

        elif isinstance(event, LogMessage):
            if event.level == "ERROR":
                self.transcript.log_error(event.message)
            else:
                self.transcript.log_system(event.message)
