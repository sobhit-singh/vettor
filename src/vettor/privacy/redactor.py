from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REDACTION_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{4,}\b"), "[REDACTED_JWT]"),
    ("bearer", re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]{12,}"), "Bearer [REDACTED_TOKEN]"),
    ("api_key", re.compile(r"\b(?:sk|pk|ghp|gho|xoxb|AIza)[A-Za-z0-9_-]{16,}\b"), "[REDACTED_KEY]"),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
    ("card", re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "[REDACTED_CARD]"),
    ("ssn", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
]


class Redactor:
    """Scrubs PII, credentials, tokens, and custom patterns before egress or persistence."""

    def __init__(self, extra_patterns: list[str] | None = None):
        self.patterns = list(REDACTION_PATTERNS)
        for idx, pat in enumerate(extra_patterns or []):
            try:
                self.patterns.append((f"custom_{idx}", re.compile(pat), "[REDACTED]"))
            except re.error:
                pass

    def redact_text(self, text: str) -> tuple[str, int]:
        """Returns (redacted_text, replacement_count)."""
        if not text:
            return text, 0
        count = 0
        out = text
        for _, rx, repl in self.patterns:
            out, n = rx.subn(repl, out)
            count += n
        return out, count

    def redact_messages(self, messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
        """Deep-redacts chat completion messages without mutating caller objects."""
        total = 0
        cleaned: list[dict[str, Any]] = []
        for msg in messages:
            m_copy = dict(msg)
            content = m_copy.get("content")
            if isinstance(content, str):
                m_copy["content"], n = self.redact_text(content)
                total += n
            elif isinstance(content, list):
                new_parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        p_copy = dict(part)
                        p_copy["text"], n = self.redact_text(str(part.get("text", "")))
                        total += n
                        new_parts.append(p_copy)
                    else:
                        new_parts.append(part)
                m_copy["content"] = new_parts
            cleaned.append(m_copy)
        return cleaned, total


class PrivacyAuditLog:
    """Appends structured audit entries for every model call and egress check."""

    def __init__(self, path: str | Path = "artifacts/privacy_audit.jsonl"):
        self.path = Path(path)

    def record(
        self,
        *,
        provider: str,
        model: str,
        endpoint: str,
        tier: str = "",
        redactions: int = 0,
        blocked: bool = False,
        reason: str = "",
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entry = {
            "ts": datetime.now(UTC).isoformat(),
            "provider": provider,
            "model": model,
            "endpoint": endpoint,
            "tier": tier,
            "redactions": redactions,
            "blocked": blocked,
            "reason": reason,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def read_entries(self, limit: int = 100) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        out: list[dict[str, Any]] = []
        for line in lines[-limit:]:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
        return out
