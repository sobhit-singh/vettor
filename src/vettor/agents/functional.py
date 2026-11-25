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
from vettor.core.llm import ModelError, ModelRouter
from vettor.core.memory import QAMemory
from vettor.core.recipe import LocatorFingerprint, RecipeCache, RecipeStep, resolve_fingerprint
from vettor.core.schema import AgentAction, CaseResult, Finding, PageObservation, TestCase
from vettor.env.browser import PageSession
from vettor.models.profiles import rank_elements_for_case
from vettor.plugins.loader import GLOBAL_PLUGINS
from vettor.reporting.assertions import evaluate_assertions
from vettor.sectors.auditor import run_sector_audits
from vettor.session.store import SessionTrace, load_project_memory
from vettor.tiers.heuristic import heuristic_next_action
from vettor.tools.registry import is_action_allowed

logger = logging.getLogger("vettor.agents.functional")


def _extract_expected_from_obs(obs: PageObservation, expected: dict[str, Any]) -> dict[str, Any]:
    """Extracts expected keys from page text without ever defaulting missing values (B-01)."""
    extracted: dict[str, Any] = {}
    search_text = obs.regions.get("main", "") or obs.body_text
    for k in expected:
        if k.lower() in ("url", "text", "message", "contains"):
            continue
        m = re.search(rf"(?i){re.escape(k)}[\s:=]+([^\n,\r|]+)", search_text)
        if m:
            extracted[k] = m.group(1).strip()
        else:
            # Also check full body_text if main region didn't contain the labeled key
            m2 = re.search(rf"(?i){re.escape(k)}[\s:=]+([^\n,\r|]+)", obs.body_text)
            if m2:
                extracted[k] = m2.group(1).strip()
    return extracted


class FunctionalityAgent(BaseQAAgent):
    """Executes data-driven TestCases or autonomous goals with tiered replay, heuristic, and LLM execution."""

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
        self._project_memory = load_project_memory()

    async def _capture_screenshot(self, session: PageSession, case_id: str) -> str | None:
        """Captures a failure screenshot to the artifacts directory."""
        try:
            from pathlib import Path
            out_dir = Path(self.cfg.reporting.output_dir) / "screenshots"
            out_dir.mkdir(parents=True, exist_ok=True)
            img_path = out_dir / f"{case_id}.png"
            await session.page.screenshot(path=str(img_path))
            return str(img_path)
        except Exception as exc:
            logger.debug("Screenshot capture failed for %s: %s", case_id, exc)
            return None

    async def _finalize_verdict(
        self,
        *,
        test_case: TestCase,
        obs: PageObservation,
        extracted: dict[str, Any],
        findings: list[Finding],
        steps: int,
        cache_hits: int,
        heals: int,
        repro_steps: list[str],
        pending_traces: list[tuple[str, str, list[RecipeStep]]],
        t0: float,
        tier_used: str,
        model_calls: int,
        trace: SessionTrace | None = None,
        note: str = "",
    ) -> CaseResult:
        """Runs code-checked oracles on the captured observation and promotes recipes only on PASS."""
        # Merge extracted values from observation without defaulting missing keys (B-01)
        obs_extracted = _extract_expected_from_obs(obs, test_case.expected)
        merged_extracted = {**obs_extracted, **extracted}

        tol = test_case.tolerance if test_case.tolerance is not None else self.cfg.reporting.float_tolerance
        assertions = evaluate_assertions(test_case, obs, merged_extracted, float_tolerance=tol)
        all_passed = all(a.passed for a in assertions)
        status: Any = "PASS" if all_passed else "FAIL"

        if not all_passed and not note:
            failed_msgs = [a.message for a in assertions if not a.passed]
            note = "; ".join(failed_msgs[:3])

        # B-06: Promote pending recipe steps ONLY when case PASSes all oracles and is not negative
        if status == "PASS" and not test_case.negative:
            intent = test_case.goal or test_case.name
            for page_url, dom_sig, p_steps in pending_traces:
                self.recipe_cache.promote_pending(page_url, p_steps, intent=intent, dom_signature=dom_sig)

        res = CaseResult(
            case_id=test_case.case_id,
            status=status,
            duration_s=round(time.monotonic() - t0, 3),
            steps=steps,
            cache_hits=cache_hits,
            heals=heals,
            note=note,
            extracted=merged_extracted,
            assertions=assertions,
            findings=findings,
            final_observation=obs,
            repro_steps=repro_steps,
            tier_used=tier_used,
            model_calls=model_calls,
            sector=test_case.sector,
            trace_path=str(trace.path) if trace else "",
        )
        if trace:
            trace.append("case_finished", {"status": status, "steps": steps, "note": note})
        await self.bus.emit(CaseFinished(case_id=test_case.case_id, status=status, result=res))
        return res

    async def run_case(self, session: PageSession, test_case: TestCase) -> CaseResult:
        """Runs a single test case from start URL to oracle-verified completion."""
        t0 = time.monotonic()
        await self.bus.emit(CaseStarted(case_id=test_case.case_id, name=test_case.name, goal=test_case.goal))

        trace = SessionTrace(
            f"{test_case.case_id}_{int(t0)}",
            base_dir=f"{self.cfg.reporting.output_dir}/sessions",
        )

        steps = 0
        cache_hits = 0
        heals = 0
        model_calls = 0
        state_changing_actions = 0
        consecutive_errors = 0
        findings: list[Finding] = []
        extracted: dict[str, Any] = {}
        history: list[str] = []
        repro_steps: list[str] = []
        pending_traces: list[tuple[str, str, list[RecipeStep]]] = []
        applied_caution_urls: set[str] = set()
        audited_sector_urls: set[str] = set()
        attempted_heuristic_vids: set[int] = set()
        last_action_sig: tuple[str, int | None, str] | None = None
        last_pathname: str = ""
        repeat_count = 0
        tier_used = "heuristic" if self.cfg.model.backend == "heuristic" else "llm"

        target_url = test_case.start_url or self.cfg.target.url
        try:
            await session.goto(target_url, url_params=self.cfg.target.url_params)
            repro_steps.append(f"Navigate to {target_url}")
        except Exception as exc:
            logger.error("Navigation failed to %s: %s", target_url, exc)
            res = CaseResult(
                case_id=test_case.case_id,
                status="ERROR",
                note=f"Navigation error: {exc}",
                duration_s=round(time.monotonic() - t0, 3),
                repro_steps=repro_steps,
                sector=test_case.sector,
                trace_path=str(trace.path),
            )
            await self.bus.emit(CaseFinished(case_id=test_case.case_id, status="ERROR", result=res))
            return res

        while steps < self.cfg.suite.max_steps_per_case:
            steps += 1
            obs = await session.observe()
            if obs.pathname != last_pathname:
                attempted_heuristic_vids.clear()
                last_pathname = obs.pathname
            await self.bus.emit(StepStarted(case_id=test_case.case_id, step=steps, url=obs.url))
            trace.append("step", {"step": steps, "url": obs.url, "dom_hash": obs.dom_hash})

            # 1. UI & Multi-Sector audits (once per page signature per session)
            page_findings = await self.ui_agent.audit_page(session, obs, test_case.case_id)
            findings.extend(page_findings)

            sec_key = f"{obs.pathname}:{obs.dom_hash}"
            if sec_key not in audited_sector_urls:
                audited_sector_urls.add(sec_key)
                sec_findings, _ = await run_sector_audits(
                    session.page,
                    page_path=obs.pathname,
                    sectors=self.cfg.suite.sectors,
                )
                findings.extend(sec_findings)

            # 2. Apply procedural memory cautions once per page entry (B-07 / B-19)
            if obs.url not in applied_caution_urls:
                applied_caution_urls.add(obs.url)
                cautions = self.memory.lookup(obs.url, kind="caution", min_trust=0.7)
                for caut in cautions:
                    workaround = caut.data
                    if isinstance(workaround, dict) and "action" in workaround:
                        try:
                            heal_action = AgentAction(
                                action=workaround["action"],
                                vid=workaround.get("vid"),
                                value=workaround.get("value", ""),
                            )
                            sel_override = workaround.get("selector")
                            await session.execute_action(heal_action, selector_override=sel_override)
                            heals += 1
                        except Exception:
                            pass

            # 3. Check terminal page completion ONLY after >=1 state-changing action (unless case has no inputs) (B-02)
            can_terminate = state_changing_actions > 0 or not test_case.inputs
            if can_terminate and self.cfg.is_terminal_page(obs.url, obs.body_text):
                return await self._finalize_verdict(
                    test_case=test_case,
                    obs=obs,
                    extracted=extracted,
                    findings=findings,
                    steps=steps,
                    cache_hits=cache_hits,
                    heals=heals,
                    repro_steps=repro_steps,
                    pending_traces=pending_traces,
                    t0=t0,
                    tier_used=tier_used,
                    model_calls=model_calls,
                    trace=trace,
                )

            # 4. Tier 1: Recipe Fast-Path Replay (with deterministic fingerprint self-healing)
            intent = test_case.goal or test_case.name
            recipe = self.recipe_cache.get(obs.url, intent=intent, dom_signature=obs.dom_hash)
            if recipe and recipe.steps and test_case.inputs and not test_case.negative:
                try:
                    for step in recipe.steps:
                        val = test_case.inputs.get(step.input_key, step.default_value)
                        replay_action = AgentAction(action=step.action, value=str(val) if val is not None else "")
                        sel_to_use = step.selector
                        # Drift check + deterministic healing ladder
                        if step.fingerprint and obs.elements and not any(e.selector == sel_to_use for e in obs.elements):
                            healed_el, method = resolve_fingerprint(step.fingerprint, obs.elements)
                            if healed_el is not None:
                                sel_to_use = healed_el.selector
                                heals += 1
                                logger.info("Self-healed drifted step via %s -> %s", method, sel_to_use)
                        await session.execute_action(replay_action, selector_override=sel_to_use)
                        state_changing_actions += 1
                        repro_steps.append(f"Replay {step.action} on {sel_to_use}")
                    cache_hits += 1
                    tier_used = "replay"
                    continue
                except Exception as exc:
                    logger.info("Recipe replay failed on %s (%s); falling back", obs.url, exc)
                    heals += 1

            # 5. Tier 2: Zero-LLM Heuristic Tier (when heuristic backend or heuristic tier enabled)
            action: AgentAction | None = None
            use_heuristic = (
                self.cfg.model.backend == "heuristic"
                or self.cfg.model.use_heuristic_tier
            )
            if use_heuristic:
                action = heuristic_next_action(obs, test_case, attempted_vids=attempted_heuristic_vids)
                if action is not None:
                    tier_used = "heuristic"

            # 6. Tier 3+: LLM Step (if heuristic did not produce an action)
            if action is None:
                if self.cfg.model.backend == "heuristic":
                    # Heuristic-only mode: if no more heuristic actions, trigger oracle verification
                    return await self._finalize_verdict(
                        test_case=test_case,
                        obs=obs,
                        extracted=extracted,
                        findings=findings,
                        steps=steps,
                        cache_hits=cache_hits,
                        heals=heals,
                        repro_steps=repro_steps,
                        pending_traces=pending_traces,
                        t0=t0,
                        tier_used="heuristic",
                        model_calls=model_calls,
                        trace=trace,
                    )

                context_lines: list[str] = []
                if test_case.goal:
                    context_lines.append(f"Goal: {test_case.goal}")
                if test_case.inputs:
                    context_lines.append(f"Inputs: {test_case.inputs}")
                if test_case.expected:
                    context_lines.append(f"Expected: {test_case.expected}")
                if self._project_memory:
                    context_lines.append(f"Project Notes:\n{self._project_memory[:400]}")

                # Rank elements for small/sub1b profiles
                profile_obj = getattr(self.router, "detailed_profile", None)
                max_els = getattr(profile_obj, "max_elements", None)
                if isinstance(max_els, int) and len(obs.elements) > max_els:
                    ranked_obs = obs.model_copy(
                        update={"elements": rank_elements_for_case(obs.elements, test_case.inputs, max_els)}
                    )
                else:
                    ranked_obs = obs

                system_prompt = (
                    "Autonomous web QA agent. Inspect compact DOM and emit ONE next action with integer vid. "
                    "Emit action='done' when workflow finishes, or 'stuck' if blocked."
                )

                try:
                    action, _ = await self.decide_action(
                        system_prompt=system_prompt,
                        obs=ranked_obs,
                        context_lines=context_lines,
                        history=history,
                    )
                    model_calls += 1
                    consecutive_errors = 0
                except ModelError as exc:
                    consecutive_errors += 1
                    history.append(f"Step {steps} model error: {exc}")
                    if consecutive_errors >= 2:
                        res = CaseResult(
                            case_id=test_case.case_id,
                            status="ERROR",
                            note=f"Model inference failed: {exc}",
                            duration_s=round(time.monotonic() - t0, 3),
                            steps=steps,
                            findings=findings,
                            repro_steps=repro_steps,
                            tier_used=tier_used,
                            model_calls=model_calls,
                            sector=test_case.sector,
                            trace_path=str(trace.path),
                        )
                        await self.bus.emit(CaseFinished(case_id=test_case.case_id, status="ERROR", result=res))
                        return res
                    continue

            # Loop detection: same action + vid + unchanged DOM hash twice
            sig = (action.action, action.vid, obs.dom_hash)
            if sig == last_action_sig:
                repeat_count += 1
                if repeat_count >= 2:
                    action = AgentAction(action="stuck", reason="Loop detected: repeated action with unchanged DOM")
            else:
                last_action_sig = sig
                repeat_count = 0

            # 7. Handle terminal / special actions
            if action.action == "done":
                return await self._finalize_verdict(
                    test_case=test_case,
                    obs=obs,
                    extracted=extracted,
                    findings=findings,
                    steps=steps,
                    cache_hits=cache_hits,
                    heals=heals,
                    repro_steps=repro_steps,
                    pending_traces=pending_traces,
                    t0=t0,
                    tier_used=tier_used,
                    model_calls=model_calls,
                    trace=trace,
                )

            if action.action == "report_bug":
                findings.append(
                    Finding(
                        kind="agent_reported_bug",
                        sector=test_case.sector,
                        severity="P1",
                        page=obs.pathname,
                        detail=action.value or action.reason or "Agent reported application defect",
                        element_vid=action.vid,
                        source="agent",
                        repro_steps=list(repro_steps),
                    )
                )
                history.append(f"Step {steps}: reported bug '{action.value}'")
                continue

            if action.action == "stuck":
                triage_decision = await self.triage_agent.diagnose(obs, test_case, obs.errors)
                if triage_decision.diagnosis == "heal" and triage_decision.action:
                    heal_act = AgentAction(
                        action=triage_decision.action,
                        vid=triage_decision.vid,
                        value=triage_decision.value,
                    )
                    try:
                        await session.execute_action(heal_act)
                        heals += 1
                        matched_heal_el = next((e for e in obs.elements if e.vid == triage_decision.vid), None)
                        self.memory.record_caution(
                            obs.url,
                            triage_decision.reason,
                            {
                                "action": triage_decision.action,
                                "vid": triage_decision.vid,
                                "selector": matched_heal_el.selector if matched_heal_el else None,
                                "value": triage_decision.value,
                            },
                        )
                        continue
                    except Exception as exc:
                        history.append(f"Step {steps} heal failed: {exc}")

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

                return await self._finalize_verdict(
                    test_case=test_case,
                    obs=obs,
                    extracted=extracted,
                    findings=findings,
                    steps=steps,
                    cache_hits=cache_hits,
                    heals=heals,
                    repro_steps=repro_steps,
                    pending_traces=pending_traces,
                    t0=t0,
                    tier_used=tier_used,
                    model_calls=model_calls,
                    trace=trace,
                    note=f"Workflow stuck: {triage_decision.reason}",
                )

            # Permission mode guard
            matched_el = next((e for e in obs.elements if e.vid == action.vid), None) if action.vid is not None else None
            allowed, block_reason = is_action_allowed(
                action,
                permission_mode=self.cfg.suite.permission_mode,
                element_label=matched_el.label if matched_el else "",
            )
            if not allowed:
                return await self._finalize_verdict(
                    test_case=test_case,
                    obs=obs,
                    extracted=extracted,
                    findings=findings,
                    steps=steps,
                    cache_hits=cache_hits,
                    heals=heals,
                    repro_steps=repro_steps,
                    pending_traces=pending_traces,
                    t0=t0,
                    tier_used=tier_used,
                    model_calls=model_calls,
                    trace=trace,
                    note=block_reason,
                )

            await GLOBAL_PLUGINS.emit_hook("before_action", {"case_id": test_case.case_id, "step": steps, "action": action.model_dump()})

            # 8. Execute action with resilient error-to-observation recovery (B-03)
            try:
                await session.execute_action(action)
                if action.action in ("fill", "select", "check", "uncheck", "click", "press"):
                    state_changing_actions += 1
                consecutive_errors = 0
                if action.vid is not None and tier_used == "heuristic":
                    attempted_heuristic_vids.add(action.vid)
            except Exception as exc:
                consecutive_errors += 1
                if action.vid is not None:
                    attempted_heuristic_vids.add(action.vid)
                history.append(f"Step {steps} action failed ({action.action} vid={action.vid}): {exc}")
                repro_steps.append(f"Failed {action.action} vid={action.vid}: {exc}")
                if consecutive_errors >= 3:
                    return await self._finalize_verdict(
                        test_case=test_case,
                        obs=obs,
                        extracted=extracted,
                        findings=findings,
                        steps=steps,
                        cache_hits=cache_hits,
                        heals=heals,
                        repro_steps=repro_steps,
                        pending_traces=pending_traces,
                        t0=t0,
                        tier_used=tier_used,
                        model_calls=model_calls,
                        trace=trace,
                        note=f"Repeated execution failure: {exc}",
                    )
                continue

            await self.bus.emit(ActionTaken(case_id=test_case.case_id, step=steps, action=action))
            await GLOBAL_PLUGINS.emit_hook("after_action", {"case_id": test_case.case_id, "step": steps, "action": action.model_dump()})
            trace.append("action", action.model_dump())
            history.append(f"Step {steps}: {action.action} vid={action.vid} val='{action.value}'")
            repro_steps.append(f"{action.action} vid={action.vid} val='{action.value}'")

            # Record to pending trace (promoted to RecipeCache ONLY on PASS)
            if action.vid is not None and matched_el and matched_el.selector:
                matched_key = ""
                for in_key, in_val in test_case.inputs.items():
                    if str(in_val) == str(action.value) or in_key.lower() in (matched_el.name.lower(), matched_el.id.lower()):
                        matched_key = in_key
                        break

                step_record = RecipeStep(
                    action=action.action,
                    selector=matched_el.selector,
                    input_key=matched_key,
                    default_value=action.value,
                    fingerprint=LocatorFingerprint.from_element(matched_el),
                )
                if pending_traces and pending_traces[-1][0] == obs.url:
                    pending_traces[-1][2].append(step_record)
                else:
                    pending_traces.append((obs.url, obs.dom_hash, [step_record]))

        # Max steps reached -> run final oracle check on current observation
        final_obs = await session.observe()
        return await self._finalize_verdict(
            test_case=test_case,
            obs=final_obs,
            extracted=extracted,
            findings=findings,
            steps=steps,
            cache_hits=cache_hits,
            heals=heals,
            repro_steps=repro_steps,
            pending_traces=pending_traces,
            t0=t0,
            tier_used=tier_used,
            model_calls=model_calls,
            trace=trace,
            note=f"Max steps ({self.cfg.suite.max_steps_per_case}) reached",
        )
