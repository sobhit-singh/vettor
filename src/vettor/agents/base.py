from __future__ import annotations

import logging
from typing import Any

from vettor.config import VettorConfig
from vettor.core.events import EventBus, LogMessage
from vettor.core.llm import CompletionResult, ModelRouter
from vettor.core.memory import QAMemory
from vettor.core.schema import AgentAction, PageObservation, SmallAgentAction

logger = logging.getLogger("vettor.agents.base")


class BaseQAAgent:
    """Base class for specialized Vettor QA agents interacting via compact text and structured LLM calls."""

    name: str = "BaseQAAgent"

    def __init__(
        self,
        cfg: VettorConfig,
        router: ModelRouter,
        bus: EventBus,
        memory: QAMemory,
    ):
        self.cfg = cfg
        self.router = router
        self.bus = bus
        self.memory = memory

    async def decide_action(
        self,
        system_prompt: str,
        obs: PageObservation,
        context_lines: list[str] | None = None,
        history: list[str] | None = None,
    ) -> tuple[AgentAction, CompletionResult]:
        """Builds a token-minimal prompt using obs.to_compact_text() and requests a structured AgentAction."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
        ]

        if context_lines:
            ctx_text = "\n".join(context_lines)
            messages.append({"role": "user", "content": f"Task Context:\n{ctx_text}"})

        profile = self.router.profile
        window_size = 2 if profile == "small" else 8
        recent_history = (history or [])[-window_size:]
        if recent_history:
            messages.append({"role": "user", "content": "Recent Steps:\n" + "\n".join(recent_history)})

        compact_obs = obs.to_compact_text(
            max_elements=40 if profile == "small" else 60,
            body_chars=200 if profile == "small" else 400,
        )

        if profile == "small":
            prompt = (
                f"Current Page State:\n{compact_obs}\n\n"
                "Rules: Skip fields whose val already matches the input. "
                "Output ONLY valid JSON with keys: 'action' "
                "('fill','select','check','uncheck','click','press','scroll','goto','wait','extract','assert_oracle','report_bug','done','stuck'), "
                "integer 'vid' (or null), and 'value'."
            )
            messages.append({"role": "user", "content": prompt})
            small_act, result = await self.router.complete_structured(messages, SmallAgentAction)
            if isinstance(small_act, AgentAction):
                return small_act, result
            return small_act.to_agent_action(), result

        prompt = (
            f"Current Page State:\n{compact_obs}\n\n"
            "Select the next action. Output MUST be valid JSON with 'action' "
            "('fill','select','check','uncheck','click','press','scroll','goto','wait','extract','assert_oracle','report_bug','done','stuck'), "
            "integer 'vid', 'value', and brief 'reason'."
        )
        messages.append({"role": "user", "content": prompt})
        action, result = await self.router.complete_structured(messages, AgentAction)
        return action, result

    async def emit_log(self, msg: str, level: str = "INFO") -> None:
        """Pushes a LogMessage event to the system EventBus."""
        await self.bus.emit(LogMessage(level=level, message=msg))
