from __future__ import annotations

import inspect
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from vettor.core.schema import AgentAction, CaseResult, Finding

logger = logging.getLogger("vettor.events")


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


class Event(BaseModel):
    """Base event contract for the typed async EventBus."""

    type: str = "event"
    timestamp: str = Field(default_factory=_utc_now_iso)


class RunStarted(Event):
    type: str = "run_started"
    total_cases: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class CaseStarted(Event):
    type: str = "case_started"
    case_id: str
    name: str = ""
    goal: str = ""


class StepStarted(Event):
    type: str = "step_started"
    case_id: str = ""
    step: int = 0
    url: str = ""


class ActionTaken(Event):
    type: str = "action_taken"
    case_id: str = ""
    step: int = 0
    action: AgentAction | dict[str, Any] = Field(default_factory=dict)


class FindingLogged(Event):
    type: str = "finding_logged"
    case_id: str = ""
    finding: Finding | dict[str, Any] = Field(default_factory=dict)


class HITLRequested(Event):
    type: str = "hitl_requested"
    case_id: str = ""
    prompt: str = ""
    options: list[str] = Field(default_factory=list)


class CaseFinished(Event):
    type: str = "case_finished"
    case_id: str = ""
    status: str = ""
    result: CaseResult | dict[str, Any] = Field(default_factory=dict)


class RunFinished(Event):
    type: str = "run_finished"
    stats: dict[str, Any] = Field(default_factory=dict)


class LogMessage(Event):
    type: str = "log_message"
    level: str = "INFO"
    message: str = ""


class EventBus:
    """Lightweight, typed async EventBus isolating subscriber exceptions."""

    def __init__(self) -> None:
        self._handlers: list[Callable[[Event], Any]] = []

    def subscribe(self, handler: Callable[[Event], Any]) -> None:
        """Register a sync or async event handler."""
        if handler not in self._handlers:
            self._handlers.append(handler)

    def unsubscribe(self, handler: Callable[[Event], Any]) -> None:
        """Unregister a handler if previously registered."""
        if handler in self._handlers:
            self._handlers.remove(handler)

    async def emit(self, event: Event) -> None:
        """Broadcast an event to all subscribers, isolating subscriber exceptions."""
        for handler in list(self._handlers):
            try:
                if inspect.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    res = handler(event)
                    if inspect.isawaitable(res):
                        await res
            except Exception as ex:
                logger.warning(
                    "Event subscriber %r raised an unhandled exception for event %s: %s",
                    handler,
                    event.type,
                    ex,
                    exc_info=True,
                )
