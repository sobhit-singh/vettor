from __future__ import annotations

import importlib
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger("vettor.plugins")


class ProviderPlugin(Protocol):
    """Protocol for Bring-Your-Own-Model / custom inference provider plugins."""

    name: str
    model: str

    async def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any: ...
    async def aclose(self) -> None: ...


HookCallback = Callable[[dict[str, Any]], Awaitable[None] | None]


@dataclass
class PluginRegistry:
    """Central registry for BYOM inference providers, lifecycle hooks, and custom QA sector auditors."""

    providers: dict[str, Callable[..., Any]] = field(default_factory=dict)
    hooks: dict[str, list[HookCallback]] = field(
        default_factory=lambda: {
            "before_action": [],
            "after_action": [],
            "on_finding": [],
            "on_run_end": [],
        }
    )

    def register_provider(self, name: str, factory: Callable[..., Any]) -> None:
        self.providers[name] = factory

    def register_hook(self, event: str, fn: HookCallback) -> None:
        self.hooks.setdefault(event, []).append(fn)

    async def emit_hook(self, event: str, payload: dict[str, Any]) -> None:
        for fn in self.hooks.get(event, []):
            try:
                res = fn(payload)
                if hasattr(res, "__await__"):
                    await res  # type: ignore[misc]
            except Exception as exc:
                logger.warning("Plugin hook '%s' failed: %s", event, exc)

    def load_module_plugin(self, dotted_path: str) -> None:
        """Loads a user plugin module (`pkg.mod`) and calls its `register(registry)` entrypoint."""
        mod = importlib.import_module(dotted_path)
        if hasattr(mod, "register") and callable(mod.register):
            mod.register(self)
        else:
            raise ValueError(f"Plugin module '{dotted_path}' must define register(registry)")


GLOBAL_PLUGINS = PluginRegistry()
