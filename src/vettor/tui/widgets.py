from __future__ import annotations

from typing import Any

from rich.markup import escape
from textual.widgets import RichLog, Static

from vettor.core.schema import CaseResult, Finding


class TelemetryBar(Static):
    """Single-line top status bar displaying execution telemetry and target/model context."""

    DEFAULT_CSS = """
    TelemetryBar {
        dock: top;
        height: 1;
        background: #0f172a;
        color: #94a3b8;
        padding: 0 1;
    }
    """

    def __init__(
        self,
        url: str = "",
        model: str = "",
        profile: str = "auto",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.url = url or "(no target)"
        self.model = model or "openai/gpt-4o-mini"
        self.profile = profile or "auto"
        self.passed: int = 0
        self.failed: int = 0
        self.errors: int = 0
        self.findings: int = 0
        self.cache_hits: int = 0
        self.total_cases: int = 0

    def on_mount(self) -> None:
        self.refresh_display()

    def update_stats(
        self,
        url: str | None = None,
        model: str | None = None,
        profile: str | None = None,
        passed: int | None = None,
        failed: int | None = None,
        errors: int | None = None,
        findings: int | None = None,
        cache_hits: int | None = None,
        total_cases: int | None = None,
    ) -> None:
        """Updates internal telemetry counters and re-renders the markup string."""
        if url is not None:
            self.url = url or "(no target)"
        if model is not None:
            self.model = model
        if profile is not None:
            self.profile = profile
        if passed is not None:
            self.passed = passed
        if failed is not None:
            self.failed = failed
        if errors is not None:
            self.errors = errors
        if findings is not None:
            self.findings = findings
        if cache_hits is not None:
            self.cache_hits = cache_hits
        if total_cases is not None:
            self.total_cases = total_cases
        self.refresh_display()

    def refresh_display(self) -> None:
        """Renders the single-line telemetry string."""
        markup = (
            f"[bold cyan]vettor[/bold cyan] │ "
            f"[bold white]{escape(self.url)}[/bold white] │ "
            f"[cyan]{escape(self.model)}[/cyan] ([yellow]{escape(self.profile)}[/yellow]) │ "
            f"[green]PASS {self.passed}[/green] "
            f"[red]FAIL {self.failed}[/red] "
            f"[yellow]ERR {self.errors}[/yellow] "
            f"[magenta]BUGS {self.findings}[/magenta] │ "
            f"[blue]CACHE {self.cache_hits}[/blue]"
        )
        self.update(markup)


class TranscriptLog(RichLog):
    """Scrollable streaming terminal log showing live, indented agent trajectories."""

    DEFAULT_CSS = """
    TranscriptLog {
        height: 1fr;
        background: #020617;
        color: #e2e8f0;
        border: none;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("highlight", True)
        kwargs.setdefault("markup", True)
        kwargs.setdefault("wrap", True)
        super().__init__(**kwargs)

    def log_case_start(self, case_id: str, name: str = "", goal: str = "") -> None:
        """Logs the start of a test case or autonomous goal."""
        desc = name or goal or "Autonomous Workflow"
        self.write(f"\n[bold cyan]▶ Case #{escape(case_id)}[/bold cyan] [dim]{escape(desc)}[/dim]")

    def log_action(
        self,
        case_id: str,
        agent: str = "FunctionalityAgent",
        mode: str = "LLM",
        detail: str = "",
        latency_ms: float = 0.0,
    ) -> None:
        """Logs an action taken by an agent with color coding and tree indentation."""
        color = "cyan"
        if mode == "REPLAY":
            color = "blue"
        elif mode == "HEAL":
            color = "magenta"
        elif mode in ("DONE", "VERIFIED"):
            color = "green"
        elif mode in ("STUCK", "ERROR"):
            color = "red"

        sec = latency_ms / 1000.0 if latency_ms > 10 else latency_ms
        self.write(f"  ├─ [dim][{sec:.2f}s][/dim] [bold {color}]{mode}[/bold {color}] {escape(detail)}")

    def log_finding(self, finding: Finding) -> None:
        """Logs a bug, accessibility, or visual finding."""
        sev_color = "red" if finding.severity in ("P0", "P1") else "yellow"
        page_info = f" on [dim]{escape(finding.page)}[/dim]" if finding.page else ""
        self.write(
            f"  ├─ [magenta]UIAgent[/magenta] ⚠ [bold {sev_color}][{finding.severity}][/bold {sev_color}] "
            f"[yellow]{escape(finding.kind)}[/yellow]{page_info}: {escape(finding.detail)}"
        )

    def log_case_done(self, result: CaseResult) -> None:
        """Logs the conclusion of a test case with final status."""
        color = "green" if result.status == "PASS" else ("red" if result.status == "FAIL" else "yellow")
        note = f" ({result.note})" if result.note else ""
        self.write(f"  └─ [bold {color}]{result.status}[/bold {color}] in {result.duration_s:.2f}s{escape(note)}")

    def log_system(self, message: str) -> None:
        """Logs a general informational message."""
        self.write(f"[dim]• {escape(message)}[/dim]")

    def log_error(self, message: str) -> None:
        """Logs an error message."""
        self.write(f"[bold red]✖ {escape(message)}[/bold red]")


class HITLBar(Static):
    """Inline Human-in-the-Loop alert banner docked directly above the input prompt."""

    DEFAULT_CSS = """
    HITLBar {
        dock: bottom;
        height: auto;
        background: #451a03;
        color: #fde68a;
        padding: 0 1;
        border-top: solid #b45309;
        display: none;
    }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.case_id: str = ""
        self.url: str = ""
        self.reason: str = ""

    def activate(self, case_id: str, url: str = "", reason: str = "") -> None:
        """Displays the inline HITL alert."""
        self.case_id = case_id
        self.url = url or "page"
        self.reason = reason or "workflow stalled"
        markup = (
            f"[bold yellow]⏸ Case {escape(self.case_id)} stalled on {escape(self.url)}[/bold yellow] "
            f"([italic]{escape(self.reason)}[/italic]). "
            f"Type hint or selector=val below to resume (or press Esc to skip):"
        )
        self.update(markup)
        self.display = True

    def deactivate(self) -> None:
        """Hides the inline HITL alert."""
        self.display = False
        self.update("")
