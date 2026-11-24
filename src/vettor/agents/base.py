from __future__ import annotations

import logging
from typing import Any

from vettor.config import VettorConfig
from vettor.core.events import EventBus, LogMessage
from vettor.core.llm import CompletionResult, ModelRouter
from vettor.core.memory import QAMemory
from vettor.core.schema import AgentAction, PageObservation

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

        # Context lines (e.g. test case inputs, active goals, cautions)
        if context_lines:
            ctx_text = "\n".join(context_lines)
            messages.append({"role": "user", "content": f"Task Context:\n{ctx_text}"})

        # History window tailored to model profile: 2 turns for small, 8 turns for frontier
        profile = self.router.profile
        window_size = 2 if profile == "small" else 8
        recent_history = (history or [])[-window_size:]
        if recent_history:
            messages.append({"role": "user", "content": "Recent Steps:\n" + "\n".join(recent_history)})

        # Compact observation
        compact_obs = obs.to_compact_text()
        prompt = (
            f"Current Page State:\n{compact_obs}\n\n"
            f"Select the next action. Output MUST be valid JSON with 'action' ('fill','select','check','uncheck','click','press','wait','done','stuck'), "
            f"integer 'vid', 'value', and brief 'reason'."
        )
        messages.append({"role": "user", "content": prompt})

        action, result = await self.router.complete_structured(messages, AgentAction)
        return action, result

    async def emit_log(self, msg: str, level: str = "INFO") -> None:
        """Pushes a LogMessage event to the system EventBus."""
        await self.bus.emit(LogMessage(level=level, message=msg))
