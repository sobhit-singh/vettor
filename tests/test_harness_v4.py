from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from vettor.agents.functional import FunctionalityAgent
from vettor.agents.planner import generate_deterministic_cases
from vettor.agents.triage import TriageAgent
from vettor.agents.ui import UIAgent
from vettor.config import VettorConfig
from vettor.core.events import EventBus
from vettor.core.llm import detect_model_profile
from vettor.core.memory import QAMemory
from vettor.core.recipe import LocatorFingerprint, RecipeCache, resolve_fingerprint
from vettor.core.schema import AgentAction, DOMElement, PageObservation, TestCase
from vettor.env.browser import PageSession
from vettor.interfaces.mcp_server import handle_mcp_request
from vettor.models.profiles import detect_detailed_profile, rank_elements_for_case
from vettor.plugins.loader import PluginRegistry
from vettor.privacy.redactor import PrivacyAuditLog, Redactor
from vettor.reporting.assertions import evaluate_assertions
from vettor.session.store import SessionTrace, diff_sessions
from vettor.tiers.heuristic import heuristic_next_action
from vettor.tools.registry import (
    build_constrained_action_schema,
    build_gbnf_grammar,
    is_action_allowed,
    parse_line_action,
)


def test_model_profile_detection_sub1b_to_frontier() -> None:
    """B-05: Boundary-aware parameter parsing across sub-1B, small, medium, and frontier."""
    assert detect_model_profile("smollm2-360m-instruct") == "small"
    assert detect_model_profile("qwen2.5-0.5b-instruct") == "small"
    assert detect_model_profile("lfm-1.5b") == "small"
    assert detect_model_profile("qwen2.5-3b") == "small"
    assert detect_model_profile("qwen2.5-14b") == "frontier"
    assert detect_model_profile("qwen2.5-32b") == "frontier"
    assert detect_model_profile("llama-3.1-70b") == "frontier"

    assert detect_detailed_profile("smollm2-360m").tier == "sub1b"
    assert detect_detailed_profile("qwen2.5-0.5b").tier == "sub1b"
    assert detect_detailed_profile("qwen2.5-1.5b").tier == "small"
    assert detect_detailed_profile("qwen2.5-7b").tier == "medium"
    assert detect_detailed_profile("gpt-4o").tier == "frontier"


def test_tool_registry_schema_gbnf_line_parser_and_permissions() -> None:
    schema = build_constrained_action_schema([1, 2, 5])
    assert schema["properties"]["vid"]["anyOf"][0]["enum"] == [1, 2, 5]

    gbnf = build_gbnf_grammar([1, 2])
    assert '"1" | "2" | "null"' in gbnf

    act = parse_line_action('fill 3 "Aarav Sharma"')
    assert act is not None
    assert act.action == "fill" and act.vid == 3 and act.value == "Aarav Sharma"

    assert parse_line_action("done").action == "done"  # type: ignore[union-attr]

    ok_ro, _ = is_action_allowed(AgentAction(action="fill", vid=1, value="x"), permission_mode="read_only")
    assert not ok_ro
    ok_safe, _ = is_action_allowed(
        AgentAction(action="click", vid=2), permission_mode="safe", element_label="Delete Account"
    )
    assert not ok_safe


def test_b01_b22_missing_and_scoped_assertions() -> None:
    """B-01 & B-22: Missing expected values must FAIL; unrelated footer text must not false-PASS."""
    obs = PageObservation(
        url="http://localhost/summary",
        pathname="/summary",
        body_text="Header | Main Content: Welcome | Footer: Status 500",
        regions={"main": "Main Content: Welcome"},
    )
    tc = TestCase(case_id="TC-B01", expected={"premium": "500"})
    results = evaluate_assertions(tc, obs, extracted={})
    assert len(results) == 1
    assert not results[0].passed


@pytest.mark.asyncio
async def test_b02_b06_early_done_fails_and_does_not_poison_recipes(tmp_path: Path) -> None:
    """B-02 & B-06: Model emitting 'done' immediately FAILS oracle check and does NOT write RecipeCache."""
    cfg = VettorConfig()
    cfg.target.url = "http://localhost:8000/step1"
    bus = EventBus()
    memory = QAMemory(storage_path=tmp_path / "mem.json")
    recipe_cache = RecipeCache(path=tmp_path / "recipe.json")

    class StubEarlyDoneRouter:
        profile = "small"
        detailed_profile = detect_detailed_profile("qwen2.5-1.5b")

        async def complete_structured(self, messages, response_model, **kwargs):
            return AgentAction(action="done", reason="skip"), None

    stub = StubEarlyDoneRouter()
    ui_agent = UIAgent(cfg, stub, bus, memory)  # type: ignore[arg-type]
    triage_agent = TriageAgent(cfg, stub, bus, memory)  # type: ignore[arg-type]
    agent = FunctionalityAgent(cfg, stub, bus, memory, recipe_cache, ui_agent, triage_agent)  # type: ignore[arg-type]

    session = PageSession(AsyncMock(), AsyncMock())
    session.goto = AsyncMock()
    session.observe = AsyncMock(
        return_value=PageObservation(
            url="http://localhost:8000/step1",
            pathname="/step1",
            elements=[DOMElement(vid=1, tag="input", name="full_name", selector="#full_name")],
            body_text="Step 1 Form",
        )
    )

    tc = TestCase(
        case_id="TC-EARLY-DONE",
        inputs={"full_name": "Alice"},
        expected={"premium": "345.00"},
    )
    res = await agent.run_case(session, tc)
    assert res.status == "FAIL"
    assert recipe_cache.get("http://localhost:8000/step1") is None


def test_heuristic_tier_and_deterministic_planner() -> None:
    obs = PageObservation(
        url="http://localhost/form",
        pathname="/form",
        elements=[
            DOMElement(vid=1, tag="input", type="text", name="full_name", id="full_name", label="Full Name", selector="#full_name", required=True, region="main"),
            DOMElement(vid=2, tag="input", type="checkbox", name="agree", id="agree", label="Accept Terms", selector="#agree", region="main"),
            DOMElement(vid=3, tag="button", type="submit", label="Calculate Premium", selector="#btn-submit", region="main"),
        ],
    )
    tc = TestCase(case_id="TC-H", inputs={"full_name": "Aarav", "agree": "true"})
    act1 = heuristic_next_action(obs, tc)
    assert act1 is not None and act1.action == "fill" and act1.vid == 1

    obs.elements[0].value = "Aarav"
    act2 = heuristic_next_action(obs, tc)
    assert act2 is not None and act2.action == "check" and act2.vid == 2

    obs.elements[1].checked = True
    act3 = heuristic_next_action(obs, tc)
    assert act3 is not None and act3.action == "click" and act3.vid == 3

    cases = generate_deterministic_cases(obs, url="http://localhost/form")
    assert len(cases) >= 3
    assert any(c.negative for c in cases)


def test_self_healing_fingerprint_and_redaction(tmp_path: Path) -> None:
    fp = LocatorFingerprint(
        tag="button",
        label="Calculate Premium",
        id="old-submit-btn",
        selector="#old-submit-btn",
        region="main",
    )
    drifted_elements = [
        DOMElement(vid=1, tag="input", id="name", selector="#name", region="main"),
        DOMElement(vid=2, tag="button", label="Calculate Premium", id="new-btn-v2", selector="#new-btn-v2", region="main"),
    ]
    healed, method = resolve_fingerprint(fp, drifted_elements)
    assert healed is not None and healed.vid == 2 and method == "role_and_label"

    redactor = Redactor()
    cleaned, count = redactor.redact_text("User alice@example.com with token sk-1234567890abcdef1234")
    assert count == 2
    assert "alice@example.com" not in cleaned
    assert "sk-1234567890abcdef1234" not in cleaned

    audit = PrivacyAuditLog(tmp_path / "audit.jsonl")
    audit.record(provider="local", model="qwen-0.5b", endpoint="http://127.0.0.1:8080/v1", tier="sub1b", redactions=count)
    assert len(audit.read_entries()) == 1


def test_sessions_plugins_and_element_ranking(tmp_path: Path) -> None:
    t1 = SessionTrace("run1", base_dir=tmp_path)
    t1.append("step", {"step": 1})
    t1.append("step", {"step": 2})
    t1.append("case_finished", {"status": "PASS"})
    t2 = t1.fork("run2", up_to_step=1)
    diff = diff_sessions(t1, t2)
    assert diff["steps_a"] == 2 and diff["steps_b"] == 1

    reg = PluginRegistry()
    reg.register_provider("custom_local", lambda cfg, model: None)
    assert "custom_local" in reg.providers

    els = [DOMElement(vid=i, tag="input", name=f"field_{i}", selector=f"#f{i}") for i in range(1, 30)]
    ranked = rank_elements_for_case(els, {"field_25": "val"}, max_elements=5)
    assert len(ranked) == 5
    assert ranked[0].name == "field_25"


@pytest.mark.asyncio
async def test_mcp_initialize_and_tools_list() -> None:
    cfg = VettorConfig()
    init_resp = await handle_mcp_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"}, cfg)
    assert init_resp["result"]["serverInfo"]["name"] == "vettor-qa"

    list_resp = await handle_mcp_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, cfg)
    assert list_resp["result"]["tools"][0]["name"] == "vettor_run_suite"
