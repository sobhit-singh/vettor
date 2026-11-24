from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from vettor.core.schema import CaseResult, Finding

logger = logging.getLogger("vettor.reporting.handoff")

HANDOFF_TEMPLATE = """# Development Handoff

Capture product issues, usability friction, and testability opportunities discovered during QA runs.

---

## Open Items

"""


class HandoffManager:
    """Manages the DEVELOPMENT_HANDOFF.md tracking file directly within the repository."""

    def __init__(self, handoff_path: str | Path = "DEVELOPMENT_HANDOFF.md"):
        self.handoff_path = Path(handoff_path)
        self._ensure_file_exists()

    def _ensure_file_exists(self) -> None:
        if not self.handoff_path.exists():
            self.handoff_path.parent.mkdir(parents=True, exist_ok=True)
            self.handoff_path.write_text(HANDOFF_TEMPLATE, encoding="utf-8")

    def record_finding(
        self,
        finding: Finding,
        case_id: str = "",
        url: str = "",
    ) -> bool:
        """Appends a deduplicated finding entry to DEVELOPMENT_HANDOFF.md. Returns True if appended."""
        self._ensure_file_exists()
        content = self.handoff_path.read_text(encoding="utf-8")

        date_str = datetime.now(UTC).strftime("%Y-%m-%d")
        title = f"{finding.kind.replace('_', ' ').title()}: {finding.detail[:60]}"
        header = f"### {date_str} - {title}"

        # Deduplicate on date + title or exact detail
        if header in content or finding.detail in content:
            return False

        fix_suggestion = finding.suggested_fix or "Investigate and resolve in target application."
        case_tag = f"`{case_id}`" if case_id else "General"
        page_ref = url or finding.page or "N/A"

        entry = (
            f"\n{header}\n\n"
            f"- **Severity:** {finding.severity}\n"
            f"- **Kind:** {finding.kind}\n"
            f"- **Related Test:** {case_tag}\n"
            f"- **Page / URL:** {page_ref}\n"
            f"- **Detail:** {finding.detail}\n"
            f"- **Suggested Fix:** {fix_suggestion}\n"
            f"- **Source:** {finding.source}\n"
            f"- **Status:** Open\n"
        )

        with self.handoff_path.open("a", encoding="utf-8") as f:
            f.write(entry)
        return True

    def list_open_items(self) -> list[dict[str, str]]:
        """Parses and returns open items from the handoff file."""
        if not self.handoff_path.exists():
            return []

        content = self.handoff_path.read_text(encoding="utf-8")
        items: list[dict[str, str]] = []

        raw_sections = content.split("### ")
        for sec in raw_sections[1:]:
            lines = sec.strip().splitlines()
            title_line = lines[0].strip() if lines else "Unknown"

            severity = "P2"
            detail = ""
            status = "Open"

            for line in lines:
                line_str = line.strip()
                if line_str.startswith("- **Severity:**"):
                    severity = line_str.split(":**", 1)[1].strip()
                elif line_str.startswith("- **Detail:**"):
                    detail = line_str.split(":**", 1)[1].strip()
                elif line_str.startswith("- **Status:**"):
                    status = line_str.split(":**", 1)[1].strip()

            if status.lower() == "open":
                items.append({
                    "title": title_line,
                    "severity": severity,
                    "detail": detail,
                    "status": status,
                })

        return items

    # Compatibility alias
    get_open_items = list_open_items


def stream_result_jsonl(output_dir: str | Path, result: CaseResult) -> Path:
    """Appends a single CaseResult JSON line to the daily test_results_YYYY-MM-DD.jsonl log."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(UTC).strftime("%Y-%m-%d")
    log_file = out_path / f"test_results_{date_str}.jsonl"

    line = result.model_dump_json() + "\n"
    with log_file.open("a", encoding="utf-8") as f:
        f.write(line)

    return log_file
