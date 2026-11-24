from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vettor.config import QAConfig, VettorConfig
from vettor.core import llama_server
from vettor.core.events import (
    ActionTaken,
    Event,
    EventBus,
    LogMessage,
    RunStarted,
    StepStarted,
)
from vettor.core.llm import (
    BaseProvider,
    CompletionResult,
    ModelRouter,
    detect_model_profile,
    extract_json_object,
)
from vettor.core.schema import (
    ActionBatch,
    AgentAction,
    DOMElement,
    PageObservation,
    TestCase,
)


def test_vettor_config_roundtrip(tmp_path: Path) -> None:
    """Tests saving and loading VettorConfig with modified settings."""
    cfg = VettorConfig()
    cfg.target.url = "http://localhost:9000/app/checkout"
    cfg.model.backend = "openai"
    cfg.model.model_name = "gpt-4o-mini"
    cfg.model.temperature = 0.2
    cfg.browser.engine = "playwright"
    cfg.browser.headless = False
    cfg.suite.mode = "autonomous_goal"
    cfg.suite.goal = "Complete checkout successfully"
    cfg.reporting.output_dir = "custom_artifacts"

    dump_file = tmp_path / "vettor_config.json"
    cfg.save(dump_file)
    assert dump_file.exists()

    loaded = VettorConfig.from_file(dump_file)
    assert loaded.target.url == "http://localhost:9000/app/checkout"
    assert loaded.start_url == "http://localhost:9000/app/checkout"
    assert loaded.model.backend == "openai"
    assert loaded.model.model_name == "gpt-4o-mini"
    assert loaded.model.temperature == 0.2
    assert loaded.browser.engine == "playwright"
    assert loaded.browser.headless is False
    assert loaded.suite.mode == "autonomous_goal"
    assert loaded.suite.goal == "Complete checkout successfully"
    assert loaded.reporting.output_dir == "custom_artifacts"
    assert loaded.model_dump() == cfg.model_dump()

    # QAConfig alias compatibility
    assert QAConfig is VettorConfig

    # Terminal page detection
    assert loaded.is_terminal_page("http://localhost:9000/app/confirmation") is True
    assert loaded.is_terminal_page("http://localhost:9000/app/step2", body_text="Order complete, thank you!") is True
    assert loaded.is_terminal_page("http://localhost:9000/app/step1", body_text="Enter personal details") is False


def test_testcase_from_flat_row() -> None:
    """Tests splitting expected_* keys into expected, case identifiers, and inputs."""
    row = {
        "id": "TC-007",
        "name": "Quote Validation",
        "goal": "Verify policy pricing",
        "tags": "smoke, regression, p0",
        "first_name": "Jane",
        "age": 34,
        "state": "CA",
        "expected_premium": 150.0,
        "expected_status": "APPROVED",
        "expected_discount_eligible": True,
        "note": "high deductible",
    }

    tc = TestCase.from_flat_row(row)
    assert tc.case_id == "TC-007"
    assert tc.name == "Quote Validation"
    assert tc.goal == "Verify policy pricing"
    assert tc.tags == ["smoke", "regression", "p0"]

    # Inputs should contain non-expected and non-metadata keys
    assert tc.inputs == {
        "first_name": "Jane",
        "age": 34,
        "state": "CA",
        "note": "high deductible",
    }

    # Expected should contain expected_* keys with prefix stripped
    assert tc.expected == {
        "premium": 150.0,
        "status": "APPROVED",
        "discount_eligible": True,
    }

    # Fallback to default_id when id/case_id is absent
    minimal_row = {"username": "admin", "expected_code": 200}
    tc_default = TestCase.from_flat_row(minimal_row, default_id="case-99")
    assert tc_default.case_id == "case-99"
    assert tc_default.inputs == {"username": "admin"}
    assert tc_default.expected == {"code": 200}


def test_page_observation_compact_text_and_signature() -> None:
    """Tests compact token-minimal representation and structural signature."""
    obs = PageObservation(
        url="http://localhost:8000/step3",
        pathname="/step3",
        title="Health",
        elements=[
            DOMElement(
                vid=1,
                tag="select",
                label="Do you smoke?",
                options=["no", "yes"],
                value="no",
            ),
            DOMElement(
                vid=2,
                tag="input",
                type="number",
                label="Age",
                required=True,
                value="",
                selector="#age-input",
            ),
            DOMElement(
                vid=3,
                tag="button",
                label="Next Step",
                selector="button.next-btn",
            ),
        ],
        errors=[],
    )

    text = obs.to_compact_text()
    expected_lines = [
        'URL: /step3 | Title: Health',
        '[1] select "Do you smoke?" options=[no, yes] val="no"',
        '[2] input(number) "Age" req val=""',
        '[3] button "Next Step"',
        'Errors: none',
    ]
    assert text == "\n".join(expected_lines)

    # Test error reporting in compact text
    obs.errors = ["Age must be at least 18", "Select a valid state"]
    err_text = obs.to_compact_text()
    assert "Errors: Age must be at least 18; Select a valid state" in err_text

    # Test signature stability
    sig1 = obs.signature()
    sig2 = obs.signature()
    assert sig1 == sig2
    assert sig1[0] == "/step3"
    assert len(sig1[1]) == 3

    # Mutating an element selector changes the signature
    obs.elements[1].selector = "#modified-selector"
    assert obs.signature() != sig1


def test_agent_action_json_schema() -> None:
    """Tests that AgentAction produces a clean flat JSON schema suitable for small LLMs and GBNF."""
    schema = AgentAction.model_json_schema()
    assert schema["type"] == "object"
    props = schema["properties"]

    assert "action" in props
    assert "vid" in props
    assert "value" in props
    assert "reason" in props

    # Check action enum/literal values
    expected_actions = {"fill", "select", "check", "uncheck", "click", "press", "wait", "assert", "done", "stuck"}
    action_schema = props["action"]
    enum_values = action_schema.get("enum") or [item.get("const") for item in action_schema.get("anyOf", []) if "const" in item]
    assert set(enum_values) == expected_actions

    # Check flat schema (no nested object definitions in properties)
    assert props["value"]["type"] == "string"
    assert props["reason"]["type"] == "string"

    # ActionBatch verification
    batch_schema = ActionBatch.model_json_schema()
    assert "actions" in batch_schema["properties"]


@pytest.mark.asyncio
async def test_event_bus_delivery_and_isolation() -> None:
    """Tests typed async EventBus delivering to sync and async subscribers with exception isolation."""
    bus = EventBus()
    received_sync: list[Event] = []
    received_async: list[Event] = []

    def sync_handler(event: Event) -> None:
        received_sync.append(event)

    async def async_handler(event: Event) -> None:
        await asyncio.sleep(0.001)
        received_async.append(event)

    def faulty_sync_handler(event: Event) -> None:
        raise ValueError("Buggy sync subscriber crashed!")

    async def faulty_async_handler(event: Event) -> None:
        await asyncio.sleep(0.001)
        raise RuntimeError("Buggy async subscriber crashed!")

    # Subscribe in mixed order
    bus.subscribe(sync_handler)
    bus.subscribe(faulty_sync_handler)
    bus.subscribe(async_handler)
    bus.subscribe(faulty_async_handler)

    ev1 = RunStarted(total_cases=2)
    ev2 = StepStarted(case_id="case-1", step=1, url="http://localhost:8000/step1")
    ev3 = ActionTaken(case_id="case-1", step=1, action=AgentAction(action="click", vid=3, reason="Submit"))
    ev4 = LogMessage(level="INFO", message="Step completed")

    # Emitting events should NOT raise despite faulty subscribers
    await bus.emit(ev1)
    await bus.emit(ev2)
    await bus.emit(ev3)
    await bus.emit(ev4)

    # Verify all healthy subscribers received every event in order
    assert [e.type for e in received_sync] == ["run_started", "step_started", "action_taken", "log_message"]
    assert [e.type for e in received_async] == ["run_started", "step_started", "action_taken", "log_message"]

    # Test unsubscribe
    bus.unsubscribe(sync_handler)
    ev5 = RunStarted(total_cases=0)
    await bus.emit(ev5)

    assert len(received_sync) == 4
    assert len(received_async) == 5


def test_detect_model_profile() -> None:
    """Tests compact model size heuristic vs frontier detection and explicit override."""
    assert detect_model_profile("LFM3-1.7B") == "small"
    assert detect_model_profile("Qwen2.5-3B-Instruct") == "small"
    assert detect_model_profile("local-model") == "small"
    assert detect_model_profile("gpt-4o") == "frontier"
    assert detect_model_profile("claude-3-5-sonnet") == "frontier"

    # Explicit override should be respected regardless of model name
    assert detect_model_profile("gpt-4o", configured="small") == "small"
    assert detect_model_profile("LFM3-1.7B", configured="frontier") == "frontier"


def test_extract_json_object() -> None:
    """Tests extracting JSON from thinking tags, markdown code blocks, and malformed strings."""
    raw = '<think>reasoning\nstep-by-step</think>\n```json\n{"action": "click", "vid": 1}\n```'
    assert extract_json_object(raw) == {"action": "click", "vid": 1}

    # Plain text without thinking
    assert extract_json_object('{"action": "done", "value": "finished"}') == {
        "action": "done",
        "value": "finished",
    }

    # Failure when no JSON exists
    with pytest.raises(ValueError, match="No JSON object found"):
        extract_json_object("plain text without any json brackets")


@pytest.mark.asyncio
async def test_model_router_complete_structured() -> None:
    """Tests complete_structured validating output with thinking tags into an AgentAction model."""
    cfg = VettorConfig()
    cfg.model.backend = "heuristic"
    router = ModelRouter(cfg)

    class StubProvider(BaseProvider):
        name = "stub"

        async def complete(self, messages, **kwargs):
            return CompletionResult(
                text='<think>analyzing DOM</think>{"action":"fill","vid":2,"value":"Alice","reason":"name"}',
                provider="stub",
                model="stub-model",
                latency_ms=12.5,
                tokens_per_s=110.0,
            )

        async def aclose(self):
            pass

    router.providers = [StubProvider(cfg, "stub-model")]

    action, res = await router.complete_structured(
        [{"role": "user", "content": "fill name field"}],
        AgentAction,
    )

    assert isinstance(action, AgentAction)
    assert action.action == "fill"
    assert action.vid == 2
    assert action.value == "Alice"
    assert action.reason == "name"
    assert res.provider == "stub"
    assert res.latency_ms == 12.5


def test_model_config_parse_active() -> None:
    """Tests parse_active splits provider and model ID across various formats."""
    cfg = VettorConfig()

    cfg.model.active = "openai/gpt-4o"
    assert cfg.model.parse_active() == ("openai", "gpt-4o")
    assert cfg.model.model_name == "gpt-4o"

    cfg.model.active = "litellm/anthropic/claude-sonnet-4"
    assert cfg.model.parse_active() == ("litellm", "anthropic/claude-sonnet-4")
    assert cfg.model.model_name == "anthropic/claude-sonnet-4"

    cfg.model.active = "vps/lfm3-1.7b"
    assert cfg.model.parse_active() == ("vps", "lfm3-1.7b")
    assert cfg.model.model_name == "lfm3-1.7b"


def test_vettor_config_load_default(tmp_path: Path, monkeypatch) -> None:
    """Tests load_default checks local vettor.json and falls back to defaults."""
    monkeypatch.chdir(tmp_path)
    cfg1 = VettorConfig.load_default()
    assert cfg1.target.url == ""

    # Create local vettor.json
    local_cfg = tmp_path / "vettor.json"
    custom_cfg = VettorConfig()
    custom_cfg.target.url = "https://custom.target.local"
    custom_cfg.save(local_cfg)

    cfg2 = VettorConfig.load_default()
    assert cfg2.target.url == "https://custom.target.local"


def test_privacy_allowed_private_and_tailscale_hosts() -> None:
    """Tests that confidential mode allows loopback, RFC1918, Tailscale, and allowed_private_hosts."""
    from vettor.core.llm import _is_private_or_allowed_host

    allowed = ["custom-host.corp"]
    # Loopback
    assert _is_private_or_allowed_host("http://127.0.0.1:8080/v1", allowed) is True
    assert _is_private_or_allowed_host("http://localhost:11434/v1", allowed) is True
    # RFC1918
    assert _is_private_or_allowed_host("http://10.0.1.5:8000/v1", allowed) is True
    assert _is_private_or_allowed_host("http://192.168.1.50:8000/v1", allowed) is True
    assert _is_private_or_allowed_host("http://172.16.0.10:8000/v1", allowed) is True
    # Tailscale CGNAT
    assert _is_private_or_allowed_host("http://100.85.12.34:8080/v1", allowed) is True
    # Explicit allowed private host
    assert _is_private_or_allowed_host("http://custom-host.corp:8080/v1", allowed) is True
    # Public cloud endpoints - must be blocked
    assert _is_private_or_allowed_host("https://api.openai.com/v1", allowed) is False
    assert _is_private_or_allowed_host("https://openrouter.ai/api/v1", allowed) is False


def test_ensure_llama_server_includes_jinja() -> None:
    """Tests that ensure_llama_server passes --jinja when use_jinja=True."""
    with (
        patch("vettor.core.llama_server.is_server_alive", side_effect=[False, True]),
        patch("vettor.core.llama_server.find_llama_executable", return_value="llama-server"),
        patch("vettor.core.llama_server.find_model_gguf", return_value="fake_model.gguf"),
        patch("subprocess.Popen") as mock_popen,
        patch("pathlib.Path.open"),
    ):
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        started = llama_server.ensure_llama_server(use_jinja=True)
        assert started is True
        assert mock_popen.called
        cmd = mock_popen.call_args[0][0]
        assert "--jinja" in cmd

