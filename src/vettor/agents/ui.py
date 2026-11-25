from __future__ import annotations

import base64
import logging
from typing import Any

from pydantic import BaseModel, Field

from vettor.agents.base import BaseQAAgent
from vettor.core.events import FindingLogged
from vettor.core.schema import Finding, PageObservation
from vettor.env.browser import PageSession
from vettor.env.observer import capture_som_screenshot
from vettor.env.probes import run_ui_probes

logger = logging.getLogger("vettor.agents.ui")


class VisualAuditResponse(BaseModel):
    """Structured response from VLM containing detected UI/UX or slop findings."""

    findings: list[Finding] = Field(default_factory=list)


class UIAgent(BaseQAAgent):
    """Specialized agent running deterministic probes and optional VLM visual/slop audits."""

    name: str = "UIAgent"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._audited_signatures: dict[int, set[tuple[Any, ...]]] = {}

    async def audit_page(
        self,
        session: PageSession,
        obs: PageObservation,
        case_id: str,
    ) -> list[Finding]:
        """Audits current page state with zero-LLM probes and optional visual audit, deduplicating per session."""
        session_key = id(session)
        seen = self._audited_signatures.setdefault(session_key, set())
        sig = obs.signature()
        if sig in seen:
            return []
        seen.add(sig)

        findings: list[Finding] = []

        # Drain runtime error buffers so findings do not duplicate across steps (B-08)
        net_errs = list(getattr(session, "network_errors", []))
        if hasattr(session, "network_errors") and isinstance(session.network_errors, list):
            session.network_errors.clear()
        pg_errs = list(getattr(session, "page_errors", []))
        if hasattr(session, "page_errors") and isinstance(session.page_errors, list):
            session.page_errors.clear()

        # 1. Deterministic probes (layout overflow, broken images, overlap, network & JS errors)
        probe_findings = await run_ui_probes(
            session.page,
            page_path=obs.pathname,
            network_errors=net_errs,
            page_errors=pg_errs,
        )
        findings.extend(probe_findings)

        # 2. Optional VLM visual/slop audit
        if self.cfg.model.enable_visual_audit:
            try:
                screenshot_bytes = await capture_som_screenshot(session.page)
                b64_image = base64.b64encode(screenshot_bytes).decode("utf-8")

                probe_summary = "\n".join(f"- [{f.severity}] {f.kind}: {f.detail}" for f in probe_findings)
                prompt = (
                    "Inspect the attached Set-of-Mark screenshot and current DOM state.\n"
                    "Identify visual bugs, text clipping, broken alignment, AI-slop placeholder text, "
                    "or contrast defects. Do NOT duplicate the following existing findings:\n"
                    f"{probe_summary or 'None'}\n\n"
                    "Output a VisualAuditResponse JSON object with any new findings."
                )

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                            },
                        ],
                    }
                ]

                vlm_resp, _ = await self.router.complete_structured(messages, VisualAuditResponse)
                for f in vlm_resp.findings:
                    f.page = obs.pathname
                    f.source = "vlm"
                    findings.append(f)
            except Exception as exc:
                logger.warning("VLM visual audit failed: %s", exc)

        # Emit FindingLogged event for each finding
        for finding in findings:
            await self.bus.emit(FindingLogged(case_id=case_id, finding=finding))

        return findings
