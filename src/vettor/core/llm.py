from __future__ import annotations

import abc
import ipaddress
import json
import logging
import re
import time
from typing import Any, Literal, TypeVar
from urllib.parse import urlparse

from openai import AsyncOpenAI
from pydantic import BaseModel

from vettor.config import ProviderConfig, VettorConfig
from vettor.core import llama_server

logger = logging.getLogger("vettor.llm")

COOLDOWN_S = 30.0
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}

T = TypeVar("T", bound=BaseModel)


class ModelError(RuntimeError):
    """Raised when model inference, completion, or structured output validation fails."""


ModelProfile = Literal["small", "frontier"]


def detect_model_profile(model_name: str, configured: str = "auto") -> ModelProfile:
    """Detects whether a model should use the small (ReAct/1-step) or frontier (multi-step batch) profile."""
    if configured in ("small", "frontier"):
        return configured  # type: ignore[return-value]

    name_lower = (model_name or "").lower()
    compact_markers = ("1.5b", "1.7b", "2b", "3b", "4b", "lfm", "tiny", "mini", "local-model")
    if any(m in name_lower for m in compact_markers):
        return "small"
    return "frontier"


def strip_thinking(text: str) -> str:
    """Removes <think>...</think> chain-of-thought blocks from model output."""
    return re.sub(r"(?is)<think>.*?</think>", "", text).strip()


def extract_json_object(raw: str) -> dict[str, Any]:
    """Strips thinking tags and markdown code blocks, then extracts the first valid JSON object."""
    cleaned = strip_thinking(raw).strip()
    # Remove markdown code fences if wrapped
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()

    start = cleaned.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in text: {raw[:100]}")

    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(cleaned[start:])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON content: {exc}") from exc

    if not isinstance(obj, dict):
        raise ValueError(f"Parsed JSON is not an object/dict: {type(obj).__name__}")
    return obj


def _is_private_or_allowed_host(endpoint: str, allowed_hosts: list[str]) -> bool:
    """Checks if an endpoint is loopback, RFC1918 private, Tailscale CGNAT, or in allowed_hosts."""
    if not endpoint:
        return False
    host = (urlparse(endpoint if "://" in endpoint else f"http://{endpoint}").hostname or "").lower()
    if not host:
        return False
    if host in LOCAL_HOSTS:
        return True
    if any(host == ah.lower() for ah in allowed_hosts):
        return True
    if host.endswith(".local") or host.endswith(".internal") or host.endswith(".lan"):
        return True
    try:
        ip = ipaddress.ip_address(host)
        if ip.is_loopback or ip.is_private:
            return True
        # Tailscale CGNAT block (100.64.0.0/10)
        if ip in ipaddress.ip_network("100.64.0.0/10"):
            return True
    except ValueError:
        pass
    return False


def _is_local_url(url: str) -> bool:
    if not url:
        return False
    host = (urlparse(url if "://" in url else f"http://{url}").hostname or "").lower()
    return host in LOCAL_HOSTS


def _extract_response_text(response: Any) -> str:
    if not response or not hasattr(response, "choices") or not response.choices:
        return ""
    message = response.choices[0].message
    text = message.content or ""
    if not str(text).strip():
        text = getattr(message, "reasoning_content", None) or ""
    return str(text)


class CompletionResult(BaseModel):
    """Inference telemetry and text result of a model completion call."""

    text: str
    provider: str
    model: str
    latency_ms: float
    tokens_per_s: float = 0.0


class BaseProvider(abc.ABC):
    """Abstract base class for async model providers."""

    name: str = "base"
    model: str = ""

    def __init__(self, cfg: VettorConfig, model: str):
        self.cfg = cfg
        self.model = model

    @abc.abstractmethod
    async def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> CompletionResult:
        raise NotImplementedError

    async def aclose(self) -> None:
        pass


class OpenAICompatProvider(BaseProvider):
    """Async provider for OpenAI, Ollama, vLLM, and llama-server via AsyncOpenAI."""

    def __init__(
        self,
        cfg: VettorConfig,
        model: str,
        endpoint: str,
        api_key: str,
        name: str = "openai-compatible",
    ):
        super().__init__(cfg, model)
        self.endpoint = endpoint
        self.api_key = api_key
        self.name = name
        self.client = AsyncOpenAI(
            base_url=endpoint,
            api_key=api_key or "none",
            timeout=cfg.model.timeout_s,
        )
        self._model_detected = False

    async def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> CompletionResult:
        model_id = self.model
        if model_id == "local-model" and not self._model_detected and _is_local_url(self.endpoint):
            detected = llama_server.get_server_model_id(self.endpoint)
            if detected:
                self.model = detected
                model_id = detected
            self._model_detected = True

        t0 = time.monotonic()
        response = await self.client.chat.completions.create(
            model=model_id,
            messages=messages,
            **kwargs,
        )
        latency_ms = (time.monotonic() - t0) * 1000.0

        text = _extract_response_text(response)
        tokens_per_s = 0.0
        if hasattr(response, "usage") and response.usage:
            completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0
            if latency_ms > 0 and completion_tokens:
                tokens_per_s = completion_tokens / (latency_ms / 1000.0)

        return CompletionResult(
            text=text,
            provider=self.name,
            model=model_id,
            latency_ms=latency_ms,
            tokens_per_s=tokens_per_s,
        )

    async def aclose(self) -> None:
        try:
            await self.client.close()
        except Exception:
            pass


class LiteLLMProvider(BaseProvider):
    """Async provider using LiteLLM acompletion."""

    name: str = "litellm"

    def __init__(
        self,
        cfg: VettorConfig,
        model: str,
        api_base: str = "",
        api_key: str = "",
        name: str = "litellm",
    ):
        super().__init__(cfg, model)
        self.api_base = api_base
        self.api_key = api_key
        self.name = name
        try:
            import litellm
            self.litellm = litellm
        except ImportError as exc:
            raise ModelError("LiteLLM requested but not installed. Run: uv sync --extra litellm") from exc

    async def complete(self, messages: list[dict[str, Any]], **kwargs: Any) -> CompletionResult:
        params = dict(kwargs)
        if self.api_base:
            params["api_base"] = self.api_base
        if self.api_key:
            params["api_key"] = self.api_key

        t0 = time.monotonic()
        response = await self.litellm.acompletion(model=self.model, messages=messages, **params)
        latency_ms = (time.monotonic() - t0) * 1000.0

        text = _extract_response_text(response)
        tokens_per_s = 0.0
        if hasattr(response, "usage") and response.usage:
            completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0
            if latency_ms > 0 and completion_tokens:
                tokens_per_s = completion_tokens / (latency_ms / 1000.0)

        return CompletionResult(
            text=text,
            provider=self.name,
            model=self.model,
            latency_ms=latency_ms,
            tokens_per_s=tokens_per_s,
        )

    async def aclose(self) -> None:
        pass


class ModelRouter:
    """Async, provider-agnostic model router with constrained structured outputs and privacy enforcement."""

    def __init__(self, cfg: VettorConfig, use_frontier: bool = False):
        self.cfg = cfg
        self.use_frontier = use_frontier
        self._cooldown: dict[str, float] = {}
        self.providers: list[BaseProvider] = self._build()

    def _check_privacy(self, endpoint: str, provider_name: str, model_name: str = "") -> None:
        if not self.cfg.model.confidential:
            return
        if endpoint and not _is_private_or_allowed_host(endpoint, self.cfg.model.allowed_private_hosts):
            raise ModelError(
                f"Privacy lock active (confidential=True): blocked external endpoint '{endpoint}'"
            )
        if provider_name == "litellm" and not model_name.startswith("ollama/"):
            raise ModelError(
                f"Privacy lock active (confidential=True): blocked external provider '{provider_name}' with model '{model_name}'"
            )
        if (
            provider_name in ("frontier", "openai", "openrouter")
            and endpoint
            and not _is_private_or_allowed_host(endpoint, self.cfg.model.allowed_private_hosts)
        ):
            raise ModelError(
                f"Privacy lock active (confidential=True): blocked external provider '{provider_name}'"
            )

    def _create(self, name: str) -> BaseProvider:
        m = self.cfg.model
        if self.use_frontier:
            self._check_privacy(m.frontier_endpoint, "frontier", m.frontier_model)
            return OpenAICompatProvider(
                self.cfg, m.frontier_model, m.frontier_endpoint, m.frontier_api_key, name="frontier"
            )
        if name == "llamacpp":
            self._check_privacy(m.endpoint, "llamacpp", m.model_name)
            return OpenAICompatProvider(
                self.cfg, m.model_name, m.endpoint, m.api_key, name="llamacpp"
            )
        if name == "litellm":
            self._check_privacy(m.providers.get("litellm", ProviderConfig()).base_url, "litellm", m.model_name)
            return LiteLLMProvider(
                self.cfg, m.model_name, m.endpoint, m.api_key, name="litellm"
            )
        if name == "fallback":
            self._check_privacy(m.fallback_api_base, "fallback", m.fallback_model)
            if not m.fallback_model:
                raise ModelError("fallback_model is not configured")
            return LiteLLMProvider(
                self.cfg, m.fallback_model, m.fallback_api_base, m.fallback_api_key, name="fallback"
            )
        if name in ("openai-compat", "openai", "ollama", "openrouter"):
            self._check_privacy(m.endpoint, name, m.model_name)
            return OpenAICompatProvider(
                self.cfg, m.model_name, m.endpoint, m.api_key, name=name
            )
        raise ModelError(f"Unknown provider: {name}")

    def _add(self, providers: list[BaseProvider], name: str) -> None:
        providers.append(self._create(name))

    def _build(self) -> list[BaseProvider]:
        providers: list[BaseProvider] = []
        if self.use_frontier:
            if not self.cfg.model.confidential and self.cfg.model.frontier_endpoint and self.cfg.model.frontier_model:
                providers.append(
                    OpenAICompatProvider(
                        self.cfg,
                        self.cfg.model.frontier_model,
                        self.cfg.model.frontier_endpoint,
                        self.cfg.model.frontier_api_key,
                        name="frontier",
                    )
                )
            return providers

        provider_name, model_id = self.cfg.model.parse_active()
        if provider_name == "heuristic" or self.cfg.model.backend == "heuristic":
            return []

        p_cfg = self.cfg.model.providers.get(provider_name)
        if p_cfg is None:
            if provider_name == "litellm":
                p_cfg = ProviderConfig(type="litellm")
            else:
                p_cfg = ProviderConfig(type="openai_compat", base_url=self.cfg.model.endpoint)

        self._check_privacy(p_cfg.base_url, provider_name, model_id)

        if p_cfg.type == "llamacpp_local":
            if llama_server.is_loopback_url(p_cfg.base_url) and self.cfg.model.auto_start_llama:
                llama_server.ensure_llama_server(
                    base_url=p_cfg.base_url,
                    configured_bin=self.cfg.model.llama_bin,
                    configured_model=self.cfg.model.model_path,
                    context_size=self.cfg.model.context_size,
                    use_jinja=self.cfg.model.use_jinja,
                    chat_template=self.cfg.model.chat_template,
                )
            providers.append(
                OpenAICompatProvider(
                    self.cfg,
                    model_id,
                    p_cfg.base_url,
                    p_cfg.resolve_api_key(),
                    name=provider_name,
                )
            )
        elif p_cfg.type == "openai_compat":
            providers.append(
                OpenAICompatProvider(
                    self.cfg,
                    model_id,
                    p_cfg.base_url,
                    p_cfg.resolve_api_key(),
                    name=provider_name,
                )
            )
        elif p_cfg.type == "litellm" or provider_name == "litellm":
            providers.append(
                LiteLLMProvider(
                    self.cfg,
                    model_id,
                    p_cfg.base_url,
                    p_cfg.resolve_api_key(),
                    name=provider_name,
                )
            )

        if self.cfg.model.fallback_model:
            self._check_privacy(self.cfg.model.fallback_api_base, "fallback", self.cfg.model.fallback_model)
            providers.append(
                LiteLLMProvider(
                    self.cfg,
                    self.cfg.model.fallback_model,
                    self.cfg.model.fallback_api_base,
                    self.cfg.model.fallback_api_key,
                    name="fallback",
                )
            )

        return providers

    @property
    def active_model(self) -> str:
        if self.providers:
            return self.providers[0].model
        return self.cfg.model.model_name

    @property
    def profile(self) -> ModelProfile:
        return detect_model_profile(self.active_model, self.cfg.model.profile)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> CompletionResult:
        """Executes a chat completion call across available providers, failing over on error."""
        if not self.providers:
            raise ModelError("No model providers configured or available.")

        temp = self.cfg.model.temperature if temperature is None else temperature
        tokens = self.cfg.model.max_tokens if max_tokens is None else max_tokens
        t_out = self.cfg.model.timeout_s if timeout is None else timeout

        call_kwargs: dict[str, Any] = {
            "temperature": temp,
            "max_tokens": tokens,
            "timeout": t_out,
            **kwargs,
        }

        # llamacpp repeat_penalty default
        if self.cfg.model.backend == "llamacpp" and "extra_body" not in call_kwargs:
            extra_body = dict(self.cfg.model.model_kwargs)
            if "repeat_penalty" not in extra_body:
                extra_body["repeat_penalty"] = 1.1
            call_kwargs["extra_body"] = extra_body

        last_error: Exception | None = None

        for provider in self.providers:
            if time.monotonic() < self._cooldown.get(provider.name, 0.0):
                continue

            attempt_kwargs = dict(call_kwargs)
            if response_format:
                attempt_kwargs["response_format"] = response_format

            try:
                res = await provider.complete(messages, **attempt_kwargs)
                if res and res.text.strip():
                    self._cooldown.pop(provider.name, None)
                    return res
            except Exception as exc:
                last_error = exc
                err_msg = str(exc).lower()
                # If json_schema failed with 400 or unsupported, retry with json_object
                if response_format and response_format.get("type") == "json_schema" and any(
                    k in err_msg for k in ("400", "schema", "response_format", "not supported", "invalid_request")
                ):
                    logger.info("Provider %s rejected json_schema; retrying with json_object...", provider.name)
                    fallback_kwargs = dict(call_kwargs)
                    fallback_kwargs["response_format"] = {"type": "json_object"}
                    try:
                        res = await provider.complete(messages, **fallback_kwargs)
                        if res and res.text.strip():
                            self._cooldown.pop(provider.name, None)
                            return res
                    except Exception as exc2:
                        last_error = exc2

                logger.warning("Provider %s failed (%s): %s", provider.name, type(exc).__name__, exc)
                self._cooldown[provider.name] = time.monotonic() + COOLDOWN_S

        raise ModelError(f"All model providers failed or are cooling down. Last error: {last_error}")

    async def complete_structured(
        self,
        messages: list[dict[str, Any]],
        response_model: type[T],
        **kwargs: Any,
    ) -> tuple[T, CompletionResult]:
        """Requests constrained JSON structured output matching response_model, retrying with error feedback."""
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": response_model.__name__,
                "strict": True,
                "schema": response_model.model_json_schema(),
            },
        }

        completion_result = await self.complete(messages, response_format=response_format, **kwargs)

        try:
            raw_obj = extract_json_object(completion_result.text)
            validated = response_model.model_validate(raw_obj)
            return validated, completion_result
        except Exception as first_err:
            logger.warning(
                "Structured completion validation failed on first attempt (%s): %s. Retrying with error feedback...",
                type(first_err).__name__,
                first_err,
            )

            correction_messages = list(messages) + [
                {"role": "assistant", "content": completion_result.text},
                {
                    "role": "user",
                    "content": f"Your previous response had validation errors:\n{first_err}\n\nPlease output ONLY a valid JSON object matching the required schema.",
                },
            ]

            completion_result_retry = await self.complete(
                correction_messages, response_format=response_format, **kwargs
            )
            try:
                raw_obj_retry = extract_json_object(completion_result_retry.text)
                validated_retry = response_model.model_validate(raw_obj_retry)
                return validated_retry, completion_result_retry
            except Exception as second_err:
                raise ModelError(
                    f"Failed to validate response against {response_model.__name__} after retry: {second_err}. "
                    f"Raw output: {completion_result_retry.text[:200]}"
                ) from second_err

    async def aclose(self) -> None:
        """Closes underlying provider clients asynchronously."""
        for provider in self.providers:
            try:
                await provider.aclose()
            except Exception:
                pass
