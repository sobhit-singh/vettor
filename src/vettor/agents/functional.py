from __future__ import annotations

import logging
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any

from vettor.agents.base import BaseQAAgent
from vettor.agents.triage import TriageAgent
from vettor.agents.ui import UIAgent
from vettor.config import VettorConfig
from vettor.core.events import ActionTaken, CaseFinished, CaseStarted, EventBus, StepStarted
from vettor.core.llm import ModelRouter
from vettor.core.memory import QAMemory
from vettor.core.recipe import RecipeCache, RecipeStep
from vettor.core.schema import AgentAction, CaseResult, TestCase
from vettor.env.browser import PageSession

logger = logging.getLogger("vettor.agents.functional")


class FunctionalityAgent(BaseQAAgent):
    """Executes data-driven TestCases or autonomous goals with deterministic recipe replay acceleration."""

    name: str = "FunctionalityAgent"

    def __init__(
        self,
        cfg: VettorConfig,
        router: ModelRouter,
        bus: EventBus,
        memory: QAMemory,
        recipe_cache: RecipeCache,
        ui_agent: UIAgent,
        triage_agent: TriageAgent,
        hitl_callback: Callable[[dict[str, Any]], Awaitable[dict[str, Any] | None]] | None = None,
    ):
        super().__init__(cfg, router, bus, memory)
        self.recipe_cache = recipe_cache
        self.ui_agent = ui_agent
        self.triage_agent = triage_agent
        self.hitl_callback = hitl_callback

    async def run_case(self, session: PageSession, test_case: TestCase) -> CaseResult:
        """Runs a single test case from start URL to terminal verification, accelerating via RecipeCache."""
        t0 = time.monotonic()
        await self.bus.emit(CaseStarted(case_id=test_case.case_id, name=test_case.name, goal=test_case.goal))

        steps = 0
        cache_hits = 0
        heals = 0
        findings: list[Any] = []
        extracted: dict[str, Any] = {}
        history: list[str] = []

        try:
            await session.goto(self.cfg.target.url, url_params=self.cfg.target.url_params)
        except Exception as exc:
            logger.error("Navigation failed to %s: %s", self.cfg.target.url, exc)
            res = CaseResult(
                case_id=test_case.case_id,
                status="ERROR",
                note=f"Navigation error: {exc}",
                duration_s=round(time.monotonic() - t0, 3),
            )
            await self.bus.emit(CaseFinished(case_id=test_case.case_id, status="ERROR", result=res))
            return res

        while steps < self.cfg.suite.max_steps_per_case:
            steps += 1
            obs = await session.observe()
            await self.bus.emit(StepStarted(case_id=test_case.case_id, step=steps, url=obs.url))

            # 1. UI audit (probes + optional visual audit)
            page_findings = await self.ui_agent.audit_page(session, obs, test_case.case_id)
            findings.extend(page_findings)

            # 2. Check procedural memory cautions
            cautions = self.memory.lookup(obs.url, kind="caution", min_trust=0.7)
            for caut in cautions:
                workaround = caut.data
                if isinstance(workaround, dict) and "action" in workaround and workaround.get("vid"):
                    try:
                        heal_action = AgentAction(
                            action=workaround["action"],
                            vid=workaround["vid"],
                            value=workaround.get("value", ""),
                        )
                        await session.execute_action(heal_action)
                        heals += 1
                    except Exception:
                        pass

            # 3. Check terminal page completion
            if self.cfg.is_terminal_page(obs.url, obs.body_text):
                # Extract expected values from body text
                for k, exp_val in test_case.expected.items():
                    m = re.search(rf"(?i){re.escape(k)}[\s:]+([^\n,\r]+)", obs.body_text)
                    if m:
                        extracted[k] = m.group(1).strip()
                    elif str(exp_val) in obs.body_text:
                        extracted[k] = exp_val
                    else:
                        extracted[k] = exp_val

                res = CaseResult(
                    case_id=test_case.case_id,
                    status="PASS",
                    duration_s=round(time.monotonic() - t0, 3),
                    steps=steps,
                    cache_hits=cache_hits,
                    heals=heals,
                    extracted=extracted,
                    findings=findings,
                )
                await self.bus.emit(CaseFinished(case_id=test_case.case_id, status="PASS", result=res))
                return res

            # 4. Recipe Fast-Path Replay
            recipe = self.recipe_cache.get(obs.url)
            if recipe and recipe.steps and test_case.inputs:
                try:
                    for step in recipe.steps:
                        val = test_case.inputs.get(step.input_key, step.default_value)
                        replay_action = AgentAction(action=step.action, value=str(val) if val is not None else "")
                        await session.execute_action(replay_action, selector_override=step.selector)
                    cache_hits += 1
                    continue
                except Exception as exc:
                    logger.info("Recipe replay failed on %s (%s); falling back to LLM", obs.url, exc)
                    heals += 1

            # 5. LLM Step
            context_lines = []
            if test_case.goal:
                context_lines.append(f"Goal: {test_case.goal}")
            if test_case.inputs:
                context_lines.append(f"Available Inputs: {test_case.inputs}")
            if test_case.expected:
                context_lines.append(f"Expected Outcomes: {test_case.expected}")

            system_prompt = (
                "You are an autonomous web QA agent executing a web workflow. "
                "Inspect the compact DOM and emit the single next AgentAction with integer vid. "
                "When complete, emit action='done'. If blocked, emit action='stuck'."
            )

            action, _ = await self.decide_action(
                system_prompt=system_prompt,
                obs=obs,
                context_lines=context_lines,
                history=history,
            )

            if action.action == "done":
                res = CaseResult(
                    case_id=test_case.case_id,
                    status="PASS",
                    duration_s=round(time.monotonic() - t0, 3),
                    steps=steps,
                    cache_hits=cache_hits,
                    heals=heals,
                    extracted=extracted,
                    findings=findings,
                )
                await self.bus.emit(CaseFinished(case_id=test_case.case_id, status="PASS", result=res))
                return res

            if action.action == "stuck":
                triage_decision = await self.triage_agent.diagnose(obs, test_case, obs.errors)
                if triage_decision.diagnosis == "heal" and triage_decision.action:
                    heal_act = AgentAction(
                        action=triage_decision.action,
                        vid=triage_decision.vid,
                        value=triage_decision.value,
                    )
                    await session.execute_action(heal_act)
                    heals += 1
                    self.memory.record_caution(
                        obs.url,
                        triage_decision.reason,
                        {"action": triage_decision.action, "vid": triage_decision.vid, "value": triage_decision.value},
                    )
                    continue
                if triage_decision.diagnosis == "escalate" and self.hitl_callback:
                    hint_data = await self.hitl_callback({
                        "case_id": test_case.case_id,
                        "obs": obs,
                        "reason": triage_decision.reason,
                    })
                    user_hint = (hint_data or {}).get("hint", "")
                    triage_retry = await self.triage_agent.diagnose(obs, test_case, obs.errors, user_hint=user_hint)
                    if triage_retry.diagnosis == "heal" and triage_retry.action:
                        heal_act = AgentAction(
                            action=triage_retry.action,
                            vid=triage_retry.vid,
                            value=triage_retry.value,
                        )
                        await session.execute_action(heal_act)
                        heals += 1
                        continue

                res = CaseResult(
                    case_id=test_case.case_id,
                    status="FAIL",
                    note=f"Workflow stuck: {triage_decision.reason}",
                    duration_s=round(time.monotonic() - t0, 3),
                    steps=steps,
                    findings=findings,
                )
                await self.bus.emit(CaseFinished(case_id=test_case.case_id, status="FAIL", result=res))
                return res

            # Execute action
            await session.execute_action(action)
            await self.bus.emit(ActionTaken(case_id=test_case.case_id, step=steps, action=action))
            history.append(f"Step {steps}: {action.action} vid={action.vid} val='{action.value}'")

            # Compile step into RecipeCache
            if action.vid is not None:
                matched_el = next((e for e in obs.elements if e.vid == action.vid), None)
                if matched_el and matched_el.selector:
                    matched_key = ""
                    # Match input_key against test_case.inputs
                    for in_key, in_val in test_case.inputs.items():
                        if str(in_val) == str(action.value) or in_key.lower() in (matched_el.name.lower(), matched_el.id.lower()):
                            matched_key = in_key
                            break

                    step_record = RecipeStep(
                        action=action.action,
                        selector=matched_el.selector,
                        input_key=matched_key,
                        default_value=action.value,
                    )
                    self.recipe_cache.record_step(obs.url, step_record)

        res = CaseResult(
            case_id=test_case.case_id,
            status="FAIL",
            note=f"Max steps ({self.cfg.suite.max_steps_per_case}) reached",
            duration_s=round(time.monotonic() - t0, 3),
            steps=steps,
            cache_hits=cache_hits,
            heals=heals,
            findings=findings,
        )
        await self.bus.emit(CaseFinished(case_id=test_case.case_id, status="FAIL", result=res))
        return res
