from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, Field

from vettor.core.recipe import normalize_page_key

logger = logging.getLogger("vettor.core.memory")


class MemoryRecord(BaseModel):
    """Persistent procedural or factual knowledge entry with trust calibration."""

    id: str
    kind: Literal["fact", "caution", "procedure"]
    domain: str
    page_key: str
    summary: str
    data: dict[str, Any] = Field(default_factory=dict)
    trust: float = 1.0
    confirmed: int = 1
    contradicted: int = 0
    status: Literal["active", "deprecated"] = "active"


class QAMemory:
    """Procedural memory repository with trust scoring and deprecation logic."""

    def __init__(self, storage_path: str | Path = "artifacts/qa_memory.json"):
        self.storage_path = Path(storage_path)
        self._lock = Lock()
        self._seq = 0
        self.records: dict[str, MemoryRecord] = self._load()

    def _normalize_domain(self, url: str) -> str:
        if not url or url == "*":
            return "*"
        try:
            parsed = urlparse(url if "://" in url else f"http://{url}")
            return parsed.netloc or parsed.path or "*"
        except Exception:
            return "*"

    def _generate_id(self, prefix: str = "rec") -> str:
        self._seq += 1
        now_str = datetime.now(UTC).strftime("%y%m%d")
        return f"{prefix}-{now_str}-{self._seq:04d}"

    def _load(self) -> dict[str, MemoryRecord]:
        if self.storage_path.exists():
            try:
                raw = json.loads(self.storage_path.read_text(encoding="utf-8"))
                records: dict[str, MemoryRecord] = {}
                for k, v in raw.items():
                    if isinstance(v, dict):
                        records[k] = MemoryRecord.model_validate(v)
                return records
            except Exception as exc:
                logger.warning("Could not load memory file %s: %s", self.storage_path, exc)
                return {}
        return {}

    def _save(self) -> None:
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        dump = {k: v.model_dump() for k, v in self.records.items()}
        self.storage_path.write_text(json.dumps(dump, indent=2), encoding="utf-8")

    def save(self) -> None:
        """Persists memory state to disk."""
        with self._lock:
            self._save()

    def get_all(self) -> list[MemoryRecord]:
        """Returns all records in memory."""
        with self._lock:
            return list(self.records.values())

    def record_fact(
        self,
        url: str,
        summary: str,
        data: dict[str, Any],
    ) -> MemoryRecord:
        """Records a verified fact about a page or form, deduplicating existing entries."""
        domain = self._normalize_domain(url)
        page_key = normalize_page_key(url)

        with self._lock:
            for rec in self.records.values():
                if (
                    rec.kind == "fact"
                    and rec.domain == domain
                    and rec.page_key == page_key
                    and (rec.summary == summary or rec.data == data)
                ):
                    rec.confirmed += 1
                    rec.data.update(data)
                    rec.trust = round(rec.confirmed / (rec.confirmed + rec.contradicted), 2)
                    self._save()
                    return rec

            rec_id = self._generate_id("fact")
            record = MemoryRecord(
                id=rec_id,
                kind="fact",
                domain=domain,
                page_key=page_key,
                summary=summary,
                data=data,
                trust=1.0,
                confirmed=1,
                contradicted=0,
                status="active",
            )
            self.records[rec_id] = record
            self._save()
            return record

    def record_caution(
        self,
        url: str,
        summary: str,
        workaround: dict[str, Any] | str,
    ) -> MemoryRecord:
        """Records a quirk, blocker, or self-healing workaround for a page."""
        domain = self._normalize_domain(url)
        page_key = normalize_page_key(url)
        workaround_dict = workaround if isinstance(workaround, dict) else {"workaround": workaround}

        with self._lock:
            for rec in self.records.values():
                if (
                    rec.kind == "caution"
                    and rec.domain == domain
                    and rec.page_key == page_key
                    and (rec.summary == summary or rec.data == workaround_dict)
                ):
                    rec.confirmed += 1
                    rec.data.update(workaround_dict)
                    rec.trust = round(rec.confirmed / (rec.confirmed + rec.contradicted), 2)
                    self._save()
                    return rec

            rec_id = self._generate_id("caut")
            record = MemoryRecord(
                id=rec_id,
                kind="caution",
                domain=domain,
                page_key=page_key,
                summary=summary,
                data=workaround_dict,
                trust=1.0,
                confirmed=1,
                contradicted=0,
                status="active",
            )
            self.records[rec_id] = record
            self._save()
            return record

    def lookup(
        self,
        url: str,
        kind: Literal["fact", "caution", "procedure"] | None = None,
        min_trust: float = 0.6,
    ) -> list[MemoryRecord]:
        """Returns matching active memory records sorted by trust and confirmed count descending."""
        target_page_key = normalize_page_key(url)
        target_domain = self._normalize_domain(url)

        with self._lock:
            results: list[MemoryRecord] = []
            for rec in self.records.values():
                if rec.status != "active":
                    continue
                if rec.trust < min_trust:
                    continue
                if kind is not None and rec.kind != kind:
                    continue
                # Match by exact page key or wildcard
                if (
                    rec.page_key == target_page_key
                    or rec.page_key == "*"
                    or (rec.domain == target_domain and rec.page_key == "*")
                ):
                    results.append(rec)

            results.sort(key=lambda r: (r.trust, r.confirmed), reverse=True)
            return results

    def confirm(self, record_id: str) -> None:
        """Increments confirmation count and recalculates trust."""
        with self._lock:
            rec = self.records.get(record_id)
            if rec:
                rec.confirmed += 1
                rec.trust = round(rec.confirmed / (rec.confirmed + rec.contradicted), 2)
                self._save()

    def contradict(self, record_id: str) -> None:
        """Increments contradiction count, recalculates trust, and deprecates if below 0.25."""
        with self._lock:
            rec = self.records.get(record_id)
            if rec:
                rec.contradicted += 1
                rec.trust = round(rec.confirmed / (rec.confirmed + rec.contradicted), 2)
                if rec.trust < 0.25:
                    rec.status = "deprecated"
                self._save()

    def summary_stats(self) -> dict[str, Any]:
        """Returns aggregated telemetry on stored memories."""
        with self._lock:
            active = [r for r in self.records.values() if r.status == "active"]
            return {
                "total": len(self.records),
                "active": len(active),
                "deprecated": len(self.records) - len(active),
                "facts": len([r for r in active if r.kind == "fact"]),
                "cautions": len([r for r in active if r.kind == "caution"]),
                "procedures": len([r for r in active if r.kind == "procedure"]),
            }

    def clear(self) -> None:
        """Clears in-memory records and deletes storage file."""
        with self._lock:
            self.records.clear()
            if self.storage_path.exists():
                try:
                    self.storage_path.unlink()
                except Exception:
                    pass
