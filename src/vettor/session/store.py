from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("vettor.session.store")


class SessionTrace:
    """Append-only JSONL session trace enabling resume, fork, deterministic replay, and diff."""

    def __init__(self, session_id: str, base_dir: str | Path = "artifacts/sessions"):
        self.session_id = session_id
        self.base_dir = Path(base_dir)
        self.path = self.base_dir / f"{session_id}.jsonl"

    def append(self, event_type: str, payload: dict[str, Any]) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(UTC).isoformat(),
            "session_id": self.session_id,
            "type": event_type,
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def read_events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        events: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except Exception:
                    pass
        return events

    def fork(self, new_session_id: str, up_to_step: int | None = None) -> SessionTrace:
        """Creates a branched session trace from this session up to `up_to_step`."""
        child = SessionTrace(new_session_id, self.base_dir)
        events = self.read_events()
        for ev in events:
            if up_to_step is not None and ev.get("type") == "step":
                step_num = ev.get("payload", {}).get("step", 0)
                if step_num > up_to_step:
                    break
            child.append(ev.get("type", "event"), ev.get("payload", {}))
        return child


def diff_sessions(trace_a: SessionTrace, trace_b: SessionTrace) -> dict[str, Any]:
    """Compares two session traces by step count, outcomes, and findings."""
    ev_a = trace_a.read_events()
    ev_b = trace_b.read_events()
    steps_a = [e for e in ev_a if e.get("type") == "step"]
    steps_b = [e for e in ev_b if e.get("type") == "step"]
    fin_a = [e for e in ev_a if e.get("type") == "case_finished"]
    fin_b = [e for e in ev_b if e.get("type") == "case_finished"]
    return {
        "session_a": trace_a.session_id,
        "session_b": trace_b.session_id,
        "steps_a": len(steps_a),
        "steps_b": len(steps_b),
        "outcomes_a": [e.get("payload", {}).get("status") for e in fin_a],
        "outcomes_b": [e.get("payload", {}).get("status") for e in fin_b],
    }


def load_project_memory(root: str | Path = ".") -> str:
    """Loads VETTOR.md project memory file (app description, quirks, forbidden actions) if present."""
    p = Path(root) / "VETTOR.md"
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8").strip()[:4000]
    except Exception as exc:
        logger.debug("Could not read VETTOR.md: %s", exc)
        return ""
