import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

logger = logging.getLogger("vettor.memory")


class MemoryRecord:
    """A procedural or factual knowledge entry learned by the QA Agent."""

    def __init__(
        self,
        id: str,
        kind: str,  # "fact" | "procedure" | "caution" | "policy"
        domain: str,
        path: str,
        summary: str,
        data: dict[str, Any],
        trust: float = 1.0,
        confirmed: int = 1,
        contradicted: int = 0,
        helpful: int = 1,
        harmful: int = 0,
        status: str = "active",  # "candidate" | "active" | "superseded" | "deprecated"
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
    ):
        self.id = id
        self.kind = kind
        self.domain = domain
        self.path = path
        self.summary = summary
        self.data = data
        self.trust = max(0.0, min(1.0, trust))
        self.confirmed = confirmed
        self.contradicted = contradicted
        self.helpful = helpful
        self.harmful = harmful
        self.status = status
        now_iso = datetime.now().isoformat()
        self.created_at = created_at or now_iso
        self.updated_at = updated_at or now_iso

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "domain": self.domain,
            "path": self.path,
            "summary": self.summary,
            "data": self.data,
            "trust": round(self.trust, 2),
            "confirmed": self.confirmed,
            "contradicted": self.contradicted,
            "helpful": self.helpful,
            "harmful": self.harmful,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MemoryRecord":
        return cls(
            id=d["id"],
            kind=d.get("kind", "fact"),
            domain=d.get("domain", "*"),
            path=d.get("path", "*"),
            summary=d.get("summary", ""),
            data=d.get("data", {}),
            trust=float(d.get("trust", 1.0)),
            confirmed=int(d.get("confirmed", 1)),
            contradicted=int(d.get("contradicted", 0)),
            helpful=int(d.get("helpful", 1)),
            harmful=int(d.get("harmful", 0)),
            status=d.get("status", "active"),
            created_at=d.get("created_at"),
            updated_at=d.get("updated_at"),
        )

    def recalculate_trust(self):
        total = self.confirmed + self.contradicted
        if total == 0:
            self.trust = 0.5
        else:
            self.trust = round(self.confirmed / total, 2)


class QAMemory:
    """Persistent procedural memory repository for Web QA agents.

    Records facts (element selectors, forms), cautions (quirks, hidden inputs, popups),
    and procedures learned during test runs so the agent improves across executions.
    """

    def __init__(self, storage_path: str | Path = "qa_memory.json"):
        self.storage_path = Path(storage_path)
        self.records: dict[str, MemoryRecord] = {}
        self._seq = 0
        self.load()

    def _normalize_domain(self, target_url: str) -> str:
        if not target_url or target_url == "*":
            return "*"
        try:
            parsed = urlparse(target_url if "://" in target_url else f"http://{target_url}")
            return parsed.netloc or parsed.path or "*"
        except Exception:
            return "*"

    def _normalize_path(self, target_url: str) -> str:
        if not target_url or target_url == "*":
            return "*"
        try:
            parsed = urlparse(target_url if "://" in target_url else f"http://{target_url}")
            return parsed.path or "/"
        except Exception:
            return "/"

    def _generate_id(self, prefix: str = "rec") -> str:
        self._seq += 1
        return f"{prefix}-{datetime.now().strftime('%y%m%d')}-{self._seq:04d}"

    def record_fact(
        self,
        target_url: str,
        summary: str,
        data: dict[str, Any],
        trust: float = 1.0,
    ) -> MemoryRecord:
        """Records a verified fact about a page or form."""
        domain = self._normalize_domain(target_url)
        path = self._normalize_path(target_url)

        # Check for existing duplicate fact
        for rec in self.records.values():
            if rec.kind == "fact" and rec.domain == domain and rec.path == path:
                next_button = data.get("next_button")
                same_button = bool(next_button) and rec.data.get("next_button") == next_button
                if rec.summary == summary or rec.data == data or same_button:
                    rec.confirmed += 1
                    rec.data.update(data)
                    rec.updated_at = datetime.now().isoformat()
                    rec.recalculate_trust()
                    self.save()
                    return rec

        rec = MemoryRecord(
            id=self._generate_id("fact"),
            kind="fact",
            domain=domain,
            path=path,
            summary=summary,
            data=data,
            trust=trust,
        )
        self.records[rec.id] = rec
        self.save()
        logger.info(f"Learned fact [{rec.id}] for {domain}{path}: {summary}")
        return rec

    def record_caution(
        self,
        target_url: str,
        summary: str,
        workaround: dict[str, Any],
        trust: float = 1.0,
    ) -> MemoryRecord:
        """Records a trap, quirk, or required recovery action (e.g. required terms checkbox)."""
        domain = self._normalize_domain(target_url)
        path = self._normalize_path(target_url)

        for rec in self.records.values():
            if rec.kind == "caution" and rec.domain == domain and rec.path == path:
                if rec.data.get("fix_selector") == workaround.get("fix_selector"):
                    rec.confirmed += 1
                    rec.helpful += 1
                    rec.updated_at = datetime.now().isoformat()
                    rec.recalculate_trust()
                    self.save()
                    return rec

        rec = MemoryRecord(
            id=self._generate_id("caution"),
            kind="caution",
            domain=domain,
            path=path,
            summary=summary,
            data=workaround,
            trust=trust,
        )
        self.records[rec.id] = rec
        self.save()
        logger.info(f"Learned caution [{rec.id}] for {domain}{path}: {summary}")
        return rec

    def lookup(
        self,
        target_url: str,
        kind: Optional[str] = None,
        min_trust: float = 0.5,
    ) -> list[MemoryRecord]:
        """Finds active knowledge records matching target URL and minimum trust score."""
        domain = self._normalize_domain(target_url)
        path = self._normalize_path(target_url)

        matched = []
        for rec in self.records.values():
            if rec.status != "active":
                continue
            if rec.trust < min_trust:
                continue
            if kind and rec.kind != kind:
                continue

            # Domain match
            if rec.domain != "*" and domain != "*" and rec.domain.lower() != domain.lower():
                continue

            # Path match
            if rec.path != "*" and path != "*" and rec.path.rstrip("/") != path.rstrip("/"):
                continue

            matched.append(rec)

        matched.sort(key=lambda r: (r.trust, r.confirmed), reverse=True)
        return matched

    def confirm(self, record_id: str):
        """Reinforce confidence in a record when it leads to a successful test outcome."""
        rec = self.records.get(record_id)
        if rec:
            rec.confirmed += 1
            rec.helpful += 1
            rec.recalculate_trust()
            rec.updated_at = datetime.now().isoformat()
            self.save()

    def contradict(self, record_id: str):
        """Decrease confidence in a record when it leads to failure or obsolete behavior."""
        rec = self.records.get(record_id)
        if rec:
            rec.contradicted += 1
            rec.harmful += 1
            rec.recalculate_trust()
            if rec.trust < 0.2:
                rec.status = "deprecated"
            rec.updated_at = datetime.now().isoformat()
            self.save()

    def forget(self, record_id: str) -> bool:
        """Removes a record from active memory."""
        if record_id in self.records:
            del self.records[record_id]
            self.save()
            return True
        return False

    def clear(self):
        """Clears all stored memory."""
        self.records.clear()
        self._seq = 0
        self.save()

    def summary_stats(self) -> dict[str, Any]:
        facts = sum(1 for r in self.records.values() if r.kind == "fact" and r.status == "active")
        cautions = sum(1 for r in self.records.values() if r.kind == "caution" and r.status == "active")
        procedures = sum(1 for r in self.records.values() if r.kind == "procedure" and r.status == "active")
        avg_trust = (
            round(sum(r.trust for r in self.records.values()) / len(self.records), 2)
            if self.records
            else 1.0
        )
        return {
            "total_records": len(self.records),
            "facts": facts,
            "cautions": cautions,
            "procedures": procedures,
            "avg_trust": avg_trust,
            "storage_path": str(self.storage_path),
        }

    def load(self):
        if not self.storage_path.exists():
            return
        try:
            content = self.storage_path.read_text(encoding="utf-8")
            if not content.strip():
                return
            data = json.loads(content)
            self.records = {
                r["id"]: MemoryRecord.from_dict(r)
                for r in data.get("records", [])
                if "id" in r
            }
            self._seq = data.get("seq", len(self.records))
        except Exception as exc:
            logger.warning(f"Failed to read memory file {self.storage_path}: {exc}")

    def save(self):
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": "1.0",
                "seq": self._seq,
                "updated_at": datetime.now().isoformat(),
                "records": [r.to_dict() for r in self.records.values()],
            }
            self.storage_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning(f"Failed to save memory file {self.storage_path}: {exc}")
