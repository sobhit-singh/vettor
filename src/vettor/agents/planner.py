from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from vettor.agents.base import BaseQAAgent
from vettor.core.schema import TestCase
from vettor.env.browser import PageSession

logger = logging.getLogger("vettor.agents.planner")


class GeneratedSuite(BaseModel):
    """Collection of generated test cases derived from page exploration or user specification."""

    cases: list[TestCase] = Field(default_factory=list)


class TestCaseAgent(BaseQAAgent):
    """Explores web pages or parses user prompts to generate comprehensive TestCase suites."""

    __test__ = False

    name: str = "TestCaseAgent"

    async def plan_from_page(
        self,
        session: PageSession,
        url: str,
        instruction: str = "",
    ) -> list[TestCase]:
        """Navigates to the specified URL, inspects the interactive DOM, and generates test cases."""
        await session.goto(url)
        obs = await session.observe()

        prompt = (
            f"Analyze the following page elements and interactive controls:\n\n"
            f"{obs.to_compact_text(max_elements=60)}\n\n"
        )
        if instruction:
            prompt += f"User Guidance: {instruction}\n\n"

        prompt += (
            "Generate a comprehensive test suite (happy-path, edge-case, and validation/negative cases) "
            "matching the GeneratedSuite schema. Each TestCase must have 'case_id', 'name', 'goal', "
            "and realistic 'inputs' matching the form fields."
        )

        messages = [
            {
                "role": "system",
                "content": "You are a senior QA Architect. Generate precise, realistic, and edge-case-rich test suites from web UI forms.",
            },
            {"role": "user", "content": prompt},
        ]

        suite, _ = await self.router.complete_structured(messages, GeneratedSuite)
        return suite.cases
