from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from vettor.agents.base import BaseQAAgent
from vettor.core.schema import PageObservation, TestCase
from vettor.env.browser import PageSession

logger = logging.getLogger("vettor.agents.planner")


class GeneratedSuite(BaseModel):
    """Collection of generated test cases derived from page exploration or user specification."""

    cases: list[TestCase] = Field(default_factory=list)


def generate_deterministic_cases(obs: PageObservation, url: str = "") -> list[TestCase]:
    """Deterministically generates happy-path, boundary, and negative test cases from DOM elements (B-27).

    Works with zero LLM calls, making test-case generation reliable even for sub-1B / heuristic setups.
    """
    cases: list[TestCase] = []
    start = url or obs.url
    input_els = [
        el for el in obs.elements
        if el.tag in ("input", "select", "textarea") and not el.disabled
    ]

    if input_els:
        happy_inputs: dict[str, str] = {}
        for el in input_els:
            key = el.name or el.id or el.label or f"field_{el.vid}"
            itype = el.type.lower()
            if el.tag == "select" and el.options:
                happy_inputs[key] = el.options[0]
            elif itype == "checkbox":
                happy_inputs[key] = "true"
            elif itype == "email":
                happy_inputs[key] = "qa.tester@example.com"
            elif itype == "number":
                happy_inputs[key] = "42"
            elif itype == "password":
                happy_inputs[key] = "ValidPass!123"
            else:
                happy_inputs[key] = "Test Value"

        cases.append(
            TestCase(
                case_id="TC-HAPPY-1",
                name="Happy Path Form Submission",
                start_url=start,
                goal="Complete primary workflow with valid inputs",
                inputs=happy_inputs,
                sector="functional",
                tags=["smoke", "happy-path"],
            )
        )

        # Required-field empty negative test
        req_els = [el for el in input_els if el.required]
        if req_els:
            cases.append(
                TestCase(
                    case_id="TC-NEG-EMPTY",
                    name="Empty Required Fields Validation",
                    start_url=start,
                    goal="Verify required field validation triggers when submitted empty",
                    inputs={el.name or el.id or f"field_{el.vid}": "" for el in req_els},
                    negative=True,
                    sector="functional",
                    tags=["negative", "validation"],
                )
            )

        # Boundary / XSS reflection safe test on first text input
        text_els = [el for el in input_els if el.tag == "input" and el.type in ("text", "search", "")]
        if text_els:
            target_el = text_els[0]
            key = target_el.name or target_el.id or f"field_{target_el.vid}"
            boundary_inputs = dict(happy_inputs)
            boundary_inputs[key] = "<script>alert(1)</script>"
            cases.append(
                TestCase(
                    case_id="TC-SEC-INPUT",
                    name="Special Characters & XSS Hygiene Check",
                    start_url=start,
                    goal="Verify special characters do not cause uncaught errors or unsafe reflection",
                    inputs=boundary_inputs,
                    sector="security",
                    tags=["security", "boundary"],
                )
            )

    # Always include a multi-sector baseline audit case
    cases.append(
        TestCase(
            case_id="TC-SECTOR-AUDIT",
            name="Multi-Sector Web Quality Audit (UI, A11y, Perf, Security, SEO)",
            start_url=start,
            goal="Audit page across UI, accessibility, performance, network, console, security, and SEO sectors",
            sector="ui",
            tags=["audit", "a11y", "seo", "performance", "security"],
        )
    )
    return cases


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

        # In heuristic mode or small models, combine deterministic generator with optional LLM cases
        if self.cfg.model.backend == "heuristic" or not getattr(self.router, "providers", True):
            return generate_deterministic_cases(obs, url=url)

        prompt = (
            f"Analyze the following page elements and interactive controls:\n\n"
            f"{obs.to_compact_text(max_elements=40)}\n\n"
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

        try:
            suite, _ = await self.router.complete_structured(messages, GeneratedSuite)
            return suite.cases or generate_deterministic_cases(obs, url=url)
        except Exception as exc:
            logger.warning("LLM suite generation failed (%s); using deterministic suite generator", exc)
            return generate_deterministic_cases(obs, url=url)
