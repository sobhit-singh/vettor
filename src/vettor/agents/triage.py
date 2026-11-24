from __future__ import annotations

import logging
from typing import Literal

from pydantic import BaseModel

from vettor.agents.base import BaseQAAgent
from vettor.core.schema import ActionType, PageObservation, TestCase

logger = logging.getLogger("vettor.agents.triage")


class TriageDecision(BaseModel):
    """Diagnosis of a workflow failure or blocker, determining whether it is a healable drift or real bug."""

    diagnosis: Literal["heal", "app_bug", "escalate"]
    reason: str
    action: ActionType | None = None
    vid: int | None = None
    value: str = ""
    severity: Literal["P0", "P1", "P2", "P3"] = "P1"


class TriageAgent(BaseQAAgent):
    """Diagnoses workflow blockers, selector drift, validation errors, and modal dialogs."""

    name: str = "TriageAgent"

    async def diagnose(
        self,
        obs: PageObservation,
        test_case: TestCase,
        errors: list[str],
        user_hint: str = "",
    ) -> TriageDecision:
        """Diagnoses blocker and determines heal action, product bug, or human escalation."""
        prompt = (
            f"The QA workflow is blocked on page: {obs.pathname or obs.url}\n\n"
            f"Page State:\n{obs.to_compact_text()}\n\n"
            f"Active Errors:\n{'; '.join(errors) if errors else 'None'}\n\n"
            f"Test Case Goal: {test_case.goal}\n"
            f"Test Case Inputs: {test_case.inputs}\n"
        )
        if user_hint:
            prompt += f"\nHuman Guidance Hint:\n{user_hint}\n"

        prompt += (
            "\nAnalyze the blocker and decide:\n"
            "- 'heal': A cookie banner, modal dialog, or self-healing fix (e.g. click close button, accept terms, fill missing required field).\n"
            "- 'app_bug': A genuine application defect, 500 error, or broken workflow.\n"
            "- 'escalate': Unresolvable without human input.\n"
            "If 'heal', provide the integer 'vid', 'action', and optional 'value'."
        )

        messages = [
            {
                "role": "system",
                "content": "You are an elite QA Triage Agent. Distinguish between external blockers (cookie banners, popups, missing required inputs) and real application bugs.",
            },
            {"role": "user", "content": prompt},
        ]

        decision, _ = await self.router.complete_structured(messages, TriageDecision)
        return decision
