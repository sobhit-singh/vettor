from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("vettor.handoff")

HANDOFF_HEADER = """# Development Handoff

Capture product issues discovered while running the Agent QA Harness.

This file is separate from dated test reports. Reports answer "what happened in this run." This file answers "what should be fixed or improved next?"

Capture usability friction here even when the underlying test passes. If a workflow is technically correct but feels confusing, hidden, slow, overly risky, or hard to recover from, add a `UX` entry so the team can review it.

---

## Entry Template

```markdown
## YYYY-MM-DD - Short Title

- **Type:** Bug | Enhancement | UX | Testability | Docs
- **Platform:** [target platform]
- **Related tests:** TC-XXX, TC-YYY
- **Severity:** P0 | P1 | P2 | P3
- **Observed:** What happened.
- **Expected:** What should happen.
- **Evidence:** Result ID, report section, or short reproduction note.
- **Suggested fix:** Concrete implementation or product suggestion.
- **Status:** Open
```

---

## Severity Guide

- **P0:** Blocks core use or risks data loss/security
- **P1:** Breaks an important workflow on a supported platform
- **P2:** Noticeable UX or reliability issue with a workaround
- **P3:** Polish, copy, docs, or testability improvement

---

## Open Items

"""


class HandoffManager:
    """Manages the DEVELOPMENT_HANDOFF.md file and streaming JSONL run logs.

    Extracts functional bugs, UX friction, and testability opportunities so
    developers receive actionable product feedback directly in their repo.
    """

    def __init__(self, handoff_path: str | Path = "DEVELOPMENT_HANDOFF.md"):
        self.handoff_path = Path(handoff_path)
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        if not self.handoff_path.exists():
            self.handoff_path.parent.mkdir(parents=True, exist_ok=True)
            self.handoff_path.write_text(HANDOFF_HEADER, encoding="utf-8")

    def record_issue(
        self,
        title: str,
        issue_type: str = "Bug",  # "Bug" | "UX" | "Enhancement" | "Testability" | "Docs"
        severity: str = "P1",      # "P0" | "P1" | "P2" | "P3"
        observed: str = "",
        expected: str = "",
        evidence: str = "",
        suggested_fix: str = "",
        related_tests: Optional[list[str]] = None,
        platform: str = "Web",
    ) -> bool:
        """Appends an issue entry to DEVELOPMENT_HANDOFF.md if not already recorded."""
        self._ensure_file_exists()
        content = self.handoff_path.read_text(encoding="utf-8")

        date_str = datetime.now().strftime("%Y-%m-%d")
        entry_title = f"{date_str} - {title}"
        tests_str = ", ".join(related_tests) if related_tests else "N/A"

        heading = f"## {entry_title}\n"
        if heading in content:
            return False

        entry_text = f"""
## {entry_title}

- **Type:** {issue_type}
- **Platform:** {platform}
- **Related tests:** {tests_str}
- **Severity:** {severity}
- **Observed:** {observed}
- **Expected:** {expected}
- **Evidence:** {evidence}
- **Suggested fix:** {suggested_fix}
- **Status:** Open
"""
        with open(self.handoff_path, "a", encoding="utf-8") as f:
            f.write(entry_text)

        logger.info(f"Recorded handoff issue [{severity}] {entry_title}")
        return True

    def list_open_items(self) -> list[dict[str, str]]:
        """Parses the current open items in DEVELOPMENT_HANDOFF.md."""
        if not self.handoff_path.exists():
            return []

        text = self.handoff_path.read_text(encoding="utf-8")
        if "## Open Items" not in text:
            return []

        items_part = text.split("## Open Items")[-1]
        sections = items_part.split("\n## ")
        parsed = []
        for sec in sections:
            sec = sec.strip()
            if not sec or sec.startswith("(_(Entries added"):
                continue
            lines = sec.splitlines()
            title = lines[0].strip("# ").strip()
            data = {"title": title}
            for line in lines[1:]:
                if line.startswith("- **") and ":**" in line:
                    key = line.split(":**")[0].replace("- **", "").strip().lower()
                    val = line.split(":**")[1].strip()
                    data[key] = val
            if data.get("status") == "Open" or "severity" in data:
                parsed.append(data)
        return parsed


def stream_test_result_jsonl(output_dir: str | Path, result: dict[str, Any], date_str: Optional[str] = None) -> Path:
    """Appends an individual test case execution result to daily streaming JSONL log."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    d_str = date_str or datetime.now().strftime("%Y-%m-%d")
    jsonl_path = out_dir / f"test_results_{d_str}.jsonl"

    entry = {
        "timestamp": datetime.now().isoformat(),
        **result,
    }

    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return jsonl_path
