from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger("vettor.config")


class TargetConfig(BaseModel):
    """Target web application settings."""

    url: str = Field(default_factory=lambda: os.getenv("QA_START_URL", ""))
    base_url: str = Field(default_factory=lambda: os.getenv("QA_BASE_URL", ""))
    auth_headers: dict[str, str] = Field(default_factory=dict)
    cookies: list[dict[str, Any]] = Field(default_factory=list)
    storage_state: str | None = Field(default_factory=lambda: os.getenv("QA_STORAGE_STATE", None))
    url_params: dict[str, str] = Field(default_factory=dict)
    local_storage: dict[str, str] = Field(default_factory=dict)
    session_storage: dict[str, str] = Field(default_factory=dict)


class ProviderConfig(BaseModel):
    """Configuration for an upstream LLM inference provider."""

    type: Literal["openai_compat", "litellm", "llamacpp_local"] = "openai_compat"
    base_url: str = ""
    api_key: str = ""
    api_key_env: str = ""

    def resolve_api_key(self) -> str:
        """Resolves the effective API key from direct configuration, environment, or 'none'."""
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            env_val = os.getenv(self.api_key_env, "")
            if env_val:
                return env_val
        return "none"


class ModelConfig(BaseModel):
    """Model-agnostic LLM & VLM inference settings."""

    active: str = Field(
        default_factory=lambda: os.getenv("QA_MODEL", "openai/gpt-4o-mini")
    )
    providers: dict[str, ProviderConfig] = Field(
        default_factory=lambda: {
            "openai": ProviderConfig(
                type="openai_compat",
                base_url="https://api.openai.com/v1",
                api_key_env="OPENAI_API_KEY",
            ),
            "openrouter": ProviderConfig(
                type="openai_compat",
                base_url="https://openrouter.ai/api/v1",
                api_key_env="OPENROUTER_API_KEY",
            ),
            "ollama": ProviderConfig(
                type="openai_compat",
                base_url="http://127.0.0.1:11434/v1",
            ),
            "llamacpp": ProviderConfig(
                type="llamacpp_local",
                base_url="http://127.0.0.1:8080/v1",
            ),
            "litellm": ProviderConfig(
                type="litellm",
            ),
        }
    )
    profile: Literal["auto", "small", "frontier"] = Field(
        default_factory=lambda: os.getenv("QA_MODEL_PROFILE", "auto")  # type: ignore[return-value]
    )
    confidential: bool = Field(default_factory=lambda: os.getenv("QA_CONFIDENTIAL", "false").lower() == "true")
    allowed_private_hosts: list[str] = Field(default_factory=list)
    enable_visual_audit: bool = Field(default_factory=lambda: os.getenv("QA_VISUAL_AUDIT", "false").lower() == "true")
    temperature: float = 0.1
    max_tokens: int = 512
    timeout_s: float = 30.0
    model_kwargs: dict[str, Any] = Field(default_factory=dict)

    # Local llama-server specific tuning
    model_path: str = Field(default_factory=lambda: os.getenv("QA_MODEL_PATH", ""))
    llama_bin: str = Field(default_factory=lambda: os.getenv("QA_LLAMA_BIN", ""))
    auto_start_llama: bool = Field(
        default_factory=lambda: os.getenv("QA_AUTO_START_LLAMA", "true").lower() == "true"
    )
    context_size: int = Field(default_factory=lambda: int(os.getenv("QA_LLAMACPP_CTX", "4096")))
    use_jinja: bool = Field(default_factory=lambda: os.getenv("QA_LLAMACPP_JINJA", "true").lower() == "true")
    chat_template: str = Field(default_factory=lambda: os.getenv("QA_LLAMACPP_TEMPLATE", ""))

    # Fallback & visual models
    vlm_model: str = Field(default_factory=lambda: os.getenv("QA_VLM_MODEL", ""))
    fallback_model: str = Field(default_factory=lambda: os.getenv("QA_FALLBACK_MODEL", ""))
    fallback_api_base: str = Field(default_factory=lambda: os.getenv("QA_FALLBACK_API_BASE", ""))
    fallback_api_key: str = Field(default_factory=lambda: os.getenv("QA_FALLBACK_API_KEY", ""))
    provider_chain: list[str] = Field(default_factory=list)
    frontier_endpoint: str = Field(default_factory=lambda: os.getenv("FRONTIER_URL", ""))
    frontier_api_key: str = Field(default_factory=lambda: os.getenv("FRONTIER_API_KEY", ""))
    frontier_model: str = Field(default_factory=lambda: os.getenv("FRONTIER_MODEL", ""))

    def parse_active(self) -> tuple[str, str]:
        """Splits `active` string on the first '/' into (provider_name, model_id)."""
        raw = (self.active or "").strip()
        if "/" in raw:
            parts = raw.split("/", 1)
            return parts[0].strip(), parts[1].strip()
        if raw in self.providers:
            return raw, "default"
        return "openai", raw

    @property
    def model_name(self) -> str:
        """Compatibility property returning the model_id portion of active."""
        _, model_id = self.parse_active()
        return model_id

    @model_name.setter
    def model_name(self, value: str) -> None:
        if "/" in value:
            self.active = value
        else:
            provider, _ = self.parse_active()
            self.active = f"{provider}/{value}"

    @property
    def backend(self) -> str:
        """Compatibility property returning provider name."""
        provider, _ = self.parse_active()
        return provider

    @backend.setter
    def backend(self, value: str) -> None:
        if value in self.providers:
            _, model_id = self.parse_active()
            self.active = f"{value}/{model_id}"
        elif value == "heuristic":
            self.active = "heuristic/mock"
        else:
            self.active = f"{value}/{self.model_name}"

    @property
    def endpoint(self) -> str:
        """Compatibility property returning base_url of the active provider."""
        provider, _ = self.parse_active()
        p_cfg = self.providers.get(provider)
        return p_cfg.base_url if p_cfg else "http://127.0.0.1:8080/v1"

    @endpoint.setter
    def endpoint(self, value: str) -> None:
        provider, _ = self.parse_active()
        if provider in self.providers:
            self.providers[provider].base_url = value
        else:
            self.providers[provider] = ProviderConfig(type="openai_compat", base_url=value)

    @property
    def api_key(self) -> str:
        """Compatibility property returning resolved API key of active provider."""
        provider, _ = self.parse_active()
        p_cfg = self.providers.get(provider)
        return p_cfg.resolve_api_key() if p_cfg else "none"

    @api_key.setter
    def api_key(self, value: str) -> None:
        provider, _ = self.parse_active()
        if provider in self.providers:
            self.providers[provider].api_key = value


class BrowserConfig(BaseModel):
    """Browser engine and execution settings."""

    engine: Literal["playwright", "rustwright"] = Field(
        default_factory=lambda: os.getenv("QA_ENGINE", "playwright")  # type: ignore[return-value]
    )
    browser_type: Literal["chromium", "firefox", "webkit"] = Field(
        default_factory=lambda: os.getenv("QA_BROWSER_TYPE", "chromium")  # type: ignore[return-value]
    )
    headless: bool = Field(default_factory=lambda: os.getenv("QA_HEADLESS", "true").lower() == "true")
    viewport_width: int = 1280
    viewport_height: int = 800
    slow_mo_ms: int = Field(default_factory=lambda: int(os.getenv("QA_SLOW_MO", "0")))
    step_timeout_ms: int = Field(default_factory=lambda: int(os.getenv("QA_TIMEOUT_MS", "4000")))
    workers: int = 1


class SuiteConfig(BaseModel):
    """Test suite and execution modes."""

    mode: Literal["data_driven", "autonomous_goal", "crawl_smoke"] = Field(
        default_factory=lambda: os.getenv("QA_MODE", "data_driven")  # type: ignore[return-value]
    )
    cases_path: str = Field(default_factory=lambda: os.getenv("QA_CASES_PATH", "tests/fixtures/sample_cases.jsonl"))
    goal: str = Field(default_factory=lambda: os.getenv("QA_GOAL", ""))
    max_steps_per_case: int = 25
    hitl_enabled: bool = Field(default_factory=lambda: os.getenv("QA_HITL", "true").lower() == "true")
    terminal_paths: list[str] = Field(
        default_factory=lambda: [
            "/summary",
            "/confirmation",
            "/thankyou",
            "/thank-you",
            "/complete",
            "/dashboard",
            "/done",
            "/receipt",
            "/order-status",
        ]
    )
    terminal_text_markers: list[str] = Field(
        default_factory=lambda: [
            "thank you",
            "order confirmed",
            "successfully submitted",
            "order complete",
            "payment successful",
            "confirmation number",
            "your receipt",
            "completed successfully",
        ]
    )


class ReportingConfig(BaseModel):
    """Forensics, artifacts, and reporting configuration."""

    output_dir: str = Field(default_factory=lambda: os.getenv("QA_OUTPUT_DIR", "artifacts"))
    results_path: str = Field(default_factory=lambda: os.getenv("QA_RESULTS_PATH", "artifacts/results.csv"))
    recipe_path: str = Field(default_factory=lambda: os.getenv("QA_RECIPE_PATH", "artifacts/recipe.json"))
    memory_path: str = Field(default_factory=lambda: os.getenv("QA_MEMORY_PATH", "artifacts/qa_memory.json"))
    handoff_path: str = Field(default_factory=lambda: os.getenv("QA_HANDOFF_PATH", "DEVELOPMENT_HANDOFF.md"))
    stream_jsonl: bool = Field(default_factory=lambda: os.getenv("QA_STREAM_JSONL", "true").lower() == "true")
    screenshot_mode: str = Field(default_factory=lambda: os.getenv("QA_SCREENSHOT_MODE", "on_failure"))
    generate_html_report: bool = True
    float_tolerance: float = 0.01


class VettorConfig(BaseModel):
    """Unified Vettor configuration combining target, model, browser, suite, and reporting."""

    target: TargetConfig = Field(default_factory=TargetConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    suite: SuiteConfig = Field(default_factory=SuiteConfig)
    reporting: ReportingConfig = Field(default_factory=ReportingConfig)

    @property
    def start_url(self) -> str:
        return self.target.url

    @start_url.setter
    def start_url(self, value: str) -> None:
        self.target.url = value

    @classmethod
    def from_file(cls, path: str | Path) -> VettorConfig:
        """Load configuration from a JSON file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Configuration file not found: {p}")
        return cls.model_validate_json(p.read_text(encoding="utf-8"))

    @classmethod
    def load_default(cls) -> VettorConfig:
        """Loads configuration from ./vettor.json, then ~/.config/vettor/config.json, else default."""
        local_cfg = Path("vettor.json")
        if local_cfg.exists():
            try:
                return cls.from_file(local_cfg)
            except Exception as exc:
                logger.warning("Failed parsing local vettor.json: %s", exc)

        global_cfg = Path.home() / ".config" / "vettor" / "config.json"
        if global_cfg.exists():
            try:
                return cls.from_file(global_cfg)
            except Exception as exc:
                logger.warning("Failed parsing global config %s: %s", global_cfg, exc)

        return cls()

    def save(self, path: str | Path) -> None:
        """Save configuration to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.model_dump_json(indent=2), encoding="utf-8")

    def is_terminal_page(self, url: str, body_text: str = "") -> bool:
        """Determine if a page represents a workflow completion state."""
        u_lower = (url or "").lower()
        b_lower = (body_text or "").lower()

        if any(term in u_lower for term in self.suite.terminal_paths):
            return True
        if any(marker in b_lower for marker in self.suite.terminal_text_markers):
            return True
        return False


# Compatibility alias
QAConfig = VettorConfig
