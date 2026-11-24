from dataclasses import dataclass, field, asdict
import json
import os
from pathlib import Path
from typing import Any, Optional


def _find_default_csv(default_name: str = "sample_cases.csv") -> str:
    env_val = os.getenv("QA_CSV_PATH")
    if env_val:
        return env_val
    p = Path(default_name)
    if p.exists():
        return str(p)
    sub = Path("sample_cases") / default_name
    if sub.exists():
        return str(sub)
    return default_name


@dataclass
class TargetConfig:
    """Target web application settings."""
    url: str = os.getenv("QA_START_URL", "http://localhost:8000/step1")
    base_url: str = os.getenv("QA_BASE_URL", "")
    auth_headers: dict[str, str] = field(default_factory=dict)
    cookies: list[dict[str, Any]] = field(default_factory=list)
    storage_state: Optional[str] = os.getenv("QA_STORAGE_STATE", None)
    url_params: dict[str, str] = field(default_factory=dict)
    local_storage: dict[str, str] = field(default_factory=dict)
    session_storage: dict[str, str] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """LLM / Inference settings."""
    backend: str = os.getenv("QA_LLM_BACKEND", "llamacpp")
    endpoint: str = os.getenv("LLAMA_URL", "http://127.0.0.1:8080/v1")
    api_key: str = os.getenv("QA_API_KEY", os.getenv("LLAMA_API_KEY", "none"))
    model_name: str = os.getenv("QA_MODEL", "LFM3-1.7B")
    model_path: str = os.getenv("QA_MODEL_PATH", "")
    llama_bin: str = os.getenv("QA_LLAMA_BIN", "")
    auto_start_llama: bool = os.getenv("QA_AUTO_START_LLAMA", "true").lower() == "true"
    temperature: float = 0.1
    max_tokens: int = 1200
    timeout_s: float = 30.0
    context_size: int = int(os.getenv("QA_LLAMACPP_CTX", "4096"))
    use_jinja: bool = os.getenv("QA_LLAMACPP_JINJA", "true").lower() == "true"
    chat_template: str = os.getenv("QA_LLAMACPP_TEMPLATE", "")
    model_kwargs: dict[str, Any] = field(default_factory=dict)

    litellm_model: str = os.getenv("QA_LITELLM_MODEL", "")
    litellm_api_base: str = os.getenv("QA_LITELLM_API_BASE", "")
    litellm_api_key: str = os.getenv("QA_LITELLM_API_KEY", "")
    fallback_model: str = os.getenv("QA_FALLBACK_MODEL", "")
    fallback_api_base: str = os.getenv("QA_FALLBACK_API_BASE", "")
    fallback_api_key: str = os.getenv("QA_FALLBACK_API_KEY", "")
    provider_chain: list[str] = field(default_factory=list)

    frontier_endpoint: str = os.getenv("FRONTIER_URL", "")
    frontier_api_key: str = os.getenv("FRONTIER_API_KEY", "")
    frontier_model: str = os.getenv("FRONTIER_MODEL", "")


@dataclass
class BrowserConfig:
    """Browser engine and execution settings."""
    engine: str = os.getenv("QA_ENGINE", "rustwright")
    browser_type: str = os.getenv("QA_BROWSER_TYPE", "chromium")
    headless: bool = os.getenv("QA_HEADLESS", "true").lower() == "true"
    viewport_width: int = 1280
    viewport_height: int = 800
    slow_mo_ms: int = int(os.getenv("QA_SLOW_MO", "0"))
    step_timeout_ms: int = int(os.getenv("QA_TIMEOUT_MS", "3500"))
    record_video: bool = False


@dataclass
class FlowConfig:
    """User lifecycle and multi-stage workflow guidance for any web application."""
    app_context: str = os.getenv("QA_APP_CONTEXT", "Universal multi-step web application workflow")
    stages: list[dict[str, Any]] = field(default_factory=list)
    terminal_paths: list[str] = field(default_factory=lambda: [
        "/summary", "/confirmation", "/thankyou", "/thank-you", "/complete", "/dashboard", "/done", "/quote", "/receipt", "/order-status"
    ])
    terminal_text_markers: list[str] = field(default_factory=lambda: [
        "thank you", "order confirmed", "successfully submitted", "quote summary",
        "booking reference", "order complete", "payment successful",
        "confirmation number", "your receipt",
    ])


@dataclass
class SuiteConfig:
    """Test suite and agentic execution modes."""
    mode: str = os.getenv("QA_MODE", "data_driven")
    cases_path: str = _find_default_csv()
    goal: str = os.getenv("QA_GOAL", "Complete the multi-step workflow, verify submission, and validate expected outcomes.")
    max_pages_per_case: int = 15


@dataclass
class ReportingConfig:
    """Artifacts, forensics, and reporting settings."""
    output_dir: str = os.getenv("QA_OUTPUT_DIR", "artifacts")
    results_path: str = os.getenv("QA_RESULTS_PATH", "results.csv")
    recipe_path: str = os.getenv("QA_RECIPE_PATH", "form_recipe.json")
    failures_dir: str = os.getenv("QA_FAILURES_DIR", "failures")
    memory_path: str = os.getenv("QA_MEMORY_PATH", "qa_memory.json")
    handoff_path: str = os.getenv("QA_HANDOFF_PATH", "DEVELOPMENT_HANDOFF.md")
    stream_jsonl: bool = os.getenv("QA_STREAM_JSONL", "true").lower() == "true"
    screenshot_mode: str = os.getenv("QA_SCREENSHOT_MODE", "on_failure")
    generate_html_report: bool = True
    float_tolerance: float = 0.01


class QAConfig:
    """Master Web QA Harness Configuration with full flat & nested parameter support."""

    def __init__(
        self,
        target: Optional[TargetConfig] = None,
        model: Optional[ModelConfig] = None,
        browser: Optional[BrowserConfig] = None,
        suite: Optional[SuiteConfig] = None,
        reporting: Optional[ReportingConfig] = None,
        flow: Optional[FlowConfig] = None,
        # Flat convenience / backward compatibility kwargs
        engine: Optional[str] = None,
        headless: Optional[bool] = None,
        start_url: Optional[str] = None,
        csv_path: Optional[str] = None,
        recipe_path: Optional[str] = None,
        results_path: Optional[str] = None,
        failures_dir: Optional[str] = None,
        auto_start_llama: Optional[bool] = None,
        llm_backend: Optional[str] = None,
        local_model: Optional[str] = None,
        litellm_model: Optional[str] = None,
        fallback_model: Optional[str] = None,
        provider_chain: Optional[list[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        llama_url: Optional[str] = None,
        llama_key: Optional[str] = None,
        float_tolerance: Optional[float] = None,
        step_timeout_ms: Optional[int] = None,
        max_pages: Optional[int] = None,
        memory_path: Optional[str] = None,
        handoff_path: Optional[str] = None,
        app_context: Optional[str] = None,
        **extra,
    ):
        self.target = target or TargetConfig()
        self.model = model or ModelConfig()
        self.browser = browser or BrowserConfig()
        self.suite = suite or SuiteConfig()
        self.reporting = reporting or ReportingConfig()
        self.flow = flow or FlowConfig()

        if app_context is not None:
            self.flow.app_context = app_context
        if engine is not None:
            self.browser.engine = engine
        if headless is not None:
            self.browser.headless = headless
        if start_url is not None:
            self.target.url = start_url
        if csv_path is not None:
            self.suite.cases_path = csv_path
        if recipe_path is not None:
            self.reporting.recipe_path = recipe_path
        if results_path is not None:
            self.reporting.results_path = results_path
        if failures_dir is not None:
            self.reporting.failures_dir = failures_dir
        if memory_path is not None:
            self.reporting.memory_path = memory_path
        if handoff_path is not None:
            self.reporting.handoff_path = handoff_path
        if auto_start_llama is not None:
            self.model.auto_start_llama = auto_start_llama
        if llm_backend is not None:
            self.model.backend = llm_backend
        if local_model is not None:
            self.model.model_name = local_model
        if litellm_model is not None:
            self.model.litellm_model = litellm_model
        if fallback_model is not None:
            self.model.fallback_model = fallback_model
        if provider_chain is not None:
            self.model.provider_chain = provider_chain
        if temperature is not None:
            self.model.temperature = temperature
        if max_tokens is not None:
            self.model.max_tokens = max_tokens
        if llama_url is not None:
            self.model.endpoint = llama_url
        if llama_key is not None:
            self.model.api_key = llama_key
        if float_tolerance is not None:
            self.reporting.float_tolerance = float_tolerance
        if step_timeout_ms is not None:
            self.browser.step_timeout_ms = step_timeout_ms
        if max_pages is not None:
            self.suite.max_pages_per_case = max_pages

    # -------------------------------------------------------------
    # Backward compatibility properties
    # -------------------------------------------------------------
    @property
    def start_url(self) -> str:
        return self.target.url

    @start_url.setter
    def start_url(self, value: str):
        self.target.url = value

    @property
    def engine(self) -> str:
        return self.browser.engine

    @engine.setter
    def engine(self, value: str):
        self.browser.engine = value

    @property
    def headless(self) -> bool:
        return self.browser.headless

    @headless.setter
    def headless(self, value: bool):
        self.browser.headless = value

    @property
    def csv_path(self) -> str:
        return self.suite.cases_path

    @csv_path.setter
    def csv_path(self, value: str):
        self.suite.cases_path = value

    @property
    def results_path(self) -> str:
        return self.reporting.results_path

    @results_path.setter
    def results_path(self, value: str):
        self.reporting.results_path = value

    @property
    def recipe_path(self) -> str:
        return self.reporting.recipe_path

    @recipe_path.setter
    def recipe_path(self, value: str):
        self.reporting.recipe_path = value

    @property
    def failures_dir(self) -> str:
        return self.reporting.failures_dir

    @failures_dir.setter
    def failures_dir(self, value: str):
        self.reporting.failures_dir = value

    @property
    def llm_backend(self) -> str:
        return self.model.backend

    @llm_backend.setter
    def llm_backend(self, value: str):
        self.model.backend = value

    @property
    def litellm_model(self) -> str:
        return self.model.litellm_model

    @litellm_model.setter
    def litellm_model(self, value: str):
        self.model.litellm_model = value

    @property
    def fallback_model(self) -> str:
        return self.model.fallback_model

    @fallback_model.setter
    def fallback_model(self, value: str):
        self.model.fallback_model = value

    @property
    def temperature(self) -> float:
        return self.model.temperature

    @temperature.setter
    def temperature(self, value: float):
        self.model.temperature = value

    @property
    def max_tokens(self) -> int:
        return self.model.max_tokens

    @max_tokens.setter
    def max_tokens(self, value: int):
        self.model.max_tokens = value

    @property
    def llama_url(self) -> str:
        return self.model.endpoint

    @llama_url.setter
    def llama_url(self, value: str):
        self.model.endpoint = value

    @property
    def llama_key(self) -> str:
        return self.model.api_key

    @llama_key.setter
    def llama_key(self, value: str):
        self.model.api_key = value

    @property
    def llama_bin(self) -> str:
        return self.model.llama_bin

    @llama_bin.setter
    def llama_bin(self, value: str):
        self.model.llama_bin = value

    @property
    def model_path(self) -> str:
        return self.model.model_path

    @model_path.setter
    def model_path(self, value: str):
        self.model.model_path = value

    @property
    def auto_start_llama(self) -> bool:
        return self.model.auto_start_llama

    @auto_start_llama.setter
    def auto_start_llama(self, value: bool):
        self.model.auto_start_llama = value

    @property
    def local_model(self) -> str:
        return self.model.model_name

    @local_model.setter
    def local_model(self, value: str):
        self.model.model_name = value

    @property
    def ollama_url(self) -> str:
        return self.model.endpoint

    @ollama_url.setter
    def ollama_url(self, value: str):
        self.model.endpoint = value

    @property
    def ollama_key(self) -> str:
        return self.model.api_key

    @ollama_key.setter
    def ollama_key(self, value: str):
        self.model.api_key = value

    @property
    def frontier_url(self) -> str:
        return self.model.frontier_endpoint

    @frontier_url.setter
    def frontier_url(self, value: str):
        self.model.frontier_endpoint = value

    @property
    def frontier_key(self) -> str:
        return self.model.frontier_api_key

    @frontier_key.setter
    def frontier_key(self, value: str):
        self.model.frontier_api_key = value

    @property
    def frontier_model(self) -> str:
        return self.model.frontier_model

    @frontier_model.setter
    def frontier_model(self, value: str):
        self.model.frontier_model = value

    @property
    def max_pages(self) -> int:
        return self.suite.max_pages_per_case

    @max_pages.setter
    def max_pages(self, value: int):
        self.suite.max_pages_per_case = value

    @property
    def float_tolerance(self) -> float:
        return self.reporting.float_tolerance

    @float_tolerance.setter
    def float_tolerance(self, value: float):
        self.reporting.float_tolerance = value

    @property
    def step_timeout_ms(self) -> int:
        return self.browser.step_timeout_ms

    @step_timeout_ms.setter
    def step_timeout_ms(self, value: int):
        self.browser.step_timeout_ms = value

    @property
    def memory_path(self) -> str:
        return self.reporting.memory_path

    @memory_path.setter
    def memory_path(self, value: str):
        self.reporting.memory_path = value

    @property
    def handoff_path(self) -> str:
        return self.reporting.handoff_path

    @handoff_path.setter
    def handoff_path(self, value: str):
        self.reporting.handoff_path = value

    @property
    def app_context(self) -> str:
        return self.flow.app_context

    @app_context.setter
    def app_context(self, value: str):
        self.flow.app_context = value

    def get_stage_guidance(self, url: str, page_title: str = "") -> dict[str, Any]:
        """Matches current URL or title against configured flow stages to retrieve domain context."""
        if not self.flow or not self.flow.stages:
            return {}
        u_lower = (url or "").lower()
        t_lower = (page_title or "").lower()
        for stage in self.flow.stages:
            match_pattern = (stage.get("match") or stage.get("name") or "").lower()
            if match_pattern and (match_pattern in u_lower or match_pattern in t_lower):
                return stage
        return {}

    def is_terminal_page(self, url: str, body_text: str = "", plan: Optional[dict] = None, page_title: str = "") -> bool:
        """Determines if the current page is a final submission/summary/confirmation state."""
        if plan and plan.get("is_final_page"):
            return True
        stage = self.get_stage_guidance(url, page_title)
        if stage and stage.get("is_terminal"):
            return True
        u_lower = (url or "").lower()
        b_lower = (body_text or "").lower()
        if any(term in u_lower for term in self.flow.terminal_paths):
            return True
        if any(marker in b_lower for marker in self.flow.terminal_text_markers):
            return True
        return False

    def get_resolved_cases_path(self) -> Path:
        p = Path(self.suite.cases_path)
        if p.exists():
            return p
        sub = Path("sample_cases") / self.suite.cases_path
        if sub.exists():
            return sub
        # Check if same stem exists with .jsonl or .csv
        for ext in (".jsonl", ".csv", ".json"):
            candidate = p.with_suffix(ext)
            if candidate.exists():
                return candidate
            sub_candidate = sub.with_suffix(ext)
            if sub_candidate.exists():
                return sub_candidate
        return p

    def get_resolved_csv_path(self) -> Path:
        return self.get_resolved_cases_path()

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": asdict(self.target),
            "model": asdict(self.model),
            "browser": asdict(self.browser),
            "suite": asdict(self.suite),
            "reporting": asdict(self.reporting),
            "flow": asdict(self.flow),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QAConfig":
        target = TargetConfig(**data.get("target", {})) if "target" in data else TargetConfig()
        model = ModelConfig(**data.get("model", {})) if "model" in data else ModelConfig()
        browser = BrowserConfig(**data.get("browser", {})) if "browser" in data else BrowserConfig()
        suite = SuiteConfig(**data.get("suite", {})) if "suite" in data else SuiteConfig()
        reporting = ReportingConfig(**data.get("reporting", {})) if "reporting" in data else ReportingConfig()
        flow = FlowConfig(**data.get("flow", {})) if "flow" in data else FlowConfig()
        return cls(target=target, model=model, browser=browser, suite=suite, reporting=reporting, flow=flow)

    @classmethod
    def from_file(cls, path: str | Path) -> "QAConfig":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Configuration file not found: {p}")
        text = p.read_text(encoding="utf-8")
        data = json.loads(text)
        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
