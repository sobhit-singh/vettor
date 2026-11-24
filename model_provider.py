from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import llm_manager
from config import QAConfig

logger = logging.getLogger("vettor.model_provider")

COOLDOWN_S = 30.0


class CompletionError(RuntimeError):
    pass


@dataclass
class Completion:
    text: str
    provider: str
    model: str


class BaseProvider:
    name = "base"

    def __init__(self, cfg: QAConfig, model: str):
        self.cfg = cfg
        self.model = model

    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> Completion:
        raise NotImplementedError

    def close(self) -> None:
        return None


class OpenAICompatProvider(BaseProvider):
    def __init__(self, cfg: QAConfig, model: str, endpoint: str, api_key: str, name: str = "openai-compatible"):
        super().__init__(cfg, model)
        self.endpoint = endpoint
        self.api_key = api_key
        self.name = name
        from openai import OpenAI

        self.client = OpenAI(base_url=endpoint, api_key=api_key or "none")

    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> Completion:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )
        return Completion(_response_text(response), self.name, self.model)

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:
            pass


class LiteLLMProvider(BaseProvider):
    name = "litellm"

    def __init__(self, cfg: QAConfig, model: str, api_base: str = "", api_key: str = ""):
        super().__init__(cfg, model)
        self.api_base = api_base
        self.api_key = api_key
        try:
            import litellm
        except ImportError as exc:
            raise CompletionError(
                "LiteLLM backend requested but 'litellm' is not installed. "
                "Install it with: uv sync --extra litellm"
            ) from exc
        self.litellm = litellm

    def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> Completion:
        params = dict(kwargs)
        if self.api_base:
            params["api_base"] = self.api_base
        if self.api_key:
            params["api_key"] = self.api_key
        response = self.litellm.completion(model=self.model, messages=messages, **params)
        return Completion(_response_text(response), self.name, self.model)


def _response_text(response: Any) -> str:
    message = response.choices[0].message
    text = message.content or ""
    if not str(text).strip():
        text = getattr(message, "reasoning_content", None) or ""
    return str(text)


class ModelRouter:
    """Provider-agnostic completion router with failover and cooldown."""

    def __init__(self, cfg: QAConfig, use_frontier: bool = False):
        self.cfg = cfg
        self.use_frontier = use_frontier
        self._cooldown: dict[str, float] = {}
        self.providers: list[BaseProvider] = self._build()

    def _add(self, providers: list[BaseProvider], name: str) -> None:
        try:
            providers.append(self._create(name))
        except Exception as exc:
            logger.warning("Provider '%s' unavailable: %s", name, exc)

    def _create(self, name: str) -> BaseProvider:
        model_cfg = self.cfg.model
        if self.use_frontier:
            return OpenAICompatProvider(
                self.cfg,
                model_cfg.frontier_model,
                model_cfg.frontier_endpoint,
                model_cfg.frontier_api_key,
                name="frontier",
            )
        if name == "llamacpp":
            return OpenAICompatProvider(
                self.cfg, model_cfg.model_name, model_cfg.endpoint, model_cfg.api_key, name="llamacpp"
            )
        if name == "litellm":
            if not model_cfg.litellm_model:
                raise CompletionError("litellm_model is not configured")
            return LiteLLMProvider(
                self.cfg,
                model_cfg.litellm_model,
                model_cfg.litellm_api_base,
                model_cfg.litellm_api_key,
            )
        if name == "fallback":
            if not model_cfg.fallback_model:
                raise CompletionError("fallback_model is not configured")
            return LiteLLMProvider(
                self.cfg,
                model_cfg.fallback_model,
                model_cfg.fallback_api_base,
                model_cfg.fallback_api_key,
            )
        if name == "openai-compat":
            return OpenAICompatProvider(
                self.cfg, model_cfg.model_name, model_cfg.endpoint, model_cfg.api_key, name="openai-compat"
            )
        raise CompletionError(f"Unknown provider: {name}")

    def _build(self) -> list[BaseProvider]:
        providers: list[BaseProvider] = []
        if self.use_frontier:
            if self.cfg.model.frontier_endpoint and self.cfg.model.frontier_model:
                self._add(providers, "frontier")
            return providers

        backend = self.cfg.model.backend
        chain = self.cfg.model.provider_chain
        if not chain:
            if backend == "heuristic":
                chain = []
            elif backend == "litellm":
                chain = ["litellm"]
            elif backend in ("openai", "ollama"):
                chain = ["openai-compat"]
            elif backend == "auto":
                chain = ["llamacpp", "litellm"]
            else:
                chain = ["llamacpp"]
                if self.cfg.model.fallback_model:
                    chain.append("fallback")

        if "llamacpp" in chain and self.cfg.model.auto_start_llama:
            started = llm_manager.ensure_llama_server(
                base_url=self.cfg.model.endpoint,
                configured_bin=self.cfg.model.llama_bin,
                configured_model=self.cfg.model.model_path,
                context_size=self.cfg.model.context_size,
                use_jinja=self.cfg.model.use_jinja,
                chat_template=self.cfg.model.chat_template,
            )
            if not started:
                logger.warning("llama.cpp server unavailable; local provider will be skipped")
        for name in chain:
            self._add(providers, name)
        return providers

    def _attempts(self, temperature: float, max_tokens: int, timeout: float,
                  response_format: dict | None, extra_body: dict | None) -> list[dict[str, Any]]:
        base: dict[str, Any] = {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout,
        }
        attempts = []
        if extra_body or response_format:
            rich = dict(base)
            if extra_body:
                rich["extra_body"] = extra_body
            if response_format:
                rich["response_format"] = response_format
            attempts.append(rich)
        attempts.append(dict(base))
        return attempts

    def complete(self, messages: list[dict[str, Any]], *, temperature: float, max_tokens: int,
                 timeout: float, response_format: dict | None = None,
                 extra_body: dict | None = None) -> Completion | None:
        attempts = self._attempts(temperature, max_tokens, timeout, response_format, extra_body)
        for provider in self.providers:
            if time.monotonic() < self._cooldown.get(provider.name, 0.0):
                continue
            for attempt in attempts:
                try:
                    completion = provider.complete(messages, **attempt)
                    if completion.text.strip():
                        self._cooldown.pop(provider.name, None)
                        logger.info(
                            "Model response received | provider=%s model=%s attempt_keys=%s",
                            completion.provider, completion.model, sorted(attempt),
                        )
                        return completion
                    logger.warning("Provider %s returned an empty response", provider.name)
                except Exception as exc:
                    logger.warning(
                        "Provider %s failed with %s: %s", provider.name, type(exc).__name__, str(exc)[:300]
                    )
            self._cooldown[provider.name] = time.monotonic() + COOLDOWN_S
        return None

    def close(self) -> None:
        for provider in self.providers:
            provider.close()

    @property
    def names(self) -> list[str]:
        return [provider.name for provider in self.providers]
