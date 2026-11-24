from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from vettor.agents.functional import FunctionalityAgent
from vettor.agents.planner import GeneratedSuite, TestCaseAgent
from vettor.agents.triage import TriageAgent, TriageDecision
from vettor.agents.ui import UIAgent
from vettor.config import VettorConfig
from vettor.core.events import EventBus, FindingLogged
from vettor.core.llm import CompletionResult, ModelRouter
from vettor.core.memory import QAMemory
from vettor.core.recipe import RecipeCache, normalize_page_key
from vettor.core.schema import (
    AgentAction,
    DOMElement,
    PageObservation,
    TestCase,
)
from vettor.env.browser import PageSession


def test_normalize_page_key_shared_by_recipe_and_memory(tmp_path) -> None:
    """Tests that RecipeCache and QAMemory share the exact same normalization stripping volatile parameters."""
    url1 = "http://localhost:8000/form?sid=abc1234&step=2#section"
    url2 = "http://localhost:8000/form?token=secret99&timestamp=1700000000&step=2#section"

    key1 = normalize_page_key(url1)
    key2 = normalize_page_key(url2)

    assert key1 == key2
    assert key1 == "/form?step=2#section"

    # RecipeCache sharing
    recipe_cache = RecipeCache(path=tmp_path / "recipe.json")
    from vettor.core.recipe import PageRecipe, RecipeStep
    recipe = PageRecipe(
        page_key=key1,
        steps=[RecipeStep(action="fill", selector="#name", input_key="name", default_value="test")],
    )
    recipe_cache.set(url1, recipe)
    retrieved_recipe = recipe_cache.get(url2)
    assert retrieved_recipe is not None
    assert retrieved_recipe.page_key == key1

    # QAMemory sharing
    memory = QAMemory(storage_path=tmp_path / "memory.json")
    memory.record_fact(url1, "Form inputs", {"fields": ["name", "email"]})
    records = memory.lookup(url2, kind="fact")
    assert len(records) == 1
    assert records[0].page_key == key1
    assert records[0].data == {"fields": ["name", "email"]}


@pytest.mark.asyncio
async def test_functionality_agent_cold_run_compiles_recipe_and_warm_replays_zero_llm_calls(tmp_path) -> None:
    """Tests cold execution compiling recipe steps and warm execution replaying with 0 LLM calls."""
    cfg = VettorConfig()
    cfg.target.url = "http://localhost:8000/step1"
    cfg.suite.max_steps_per_case = 5

    bus = EventBus()
    memory = QAMemory(storage_path=tmp_path / "mem.json")
    recipe_cache = RecipeCache(path=tmp_path / "recipe.json")

    mock_router = MagicMock(spec=ModelRouter)
    mock_router.profile = "small"
    mock_router.complete_structured = AsyncMock()

    ui_agent = UIAgent(cfg, mock_router, bus, memory)
    triage_agent = TriageAgent(cfg, mock_router, bus, memory)

    agent = FunctionalityAgent(
        cfg=cfg,
        router=mock_router,
        bus=bus,
        memory=memory,
        recipe_cache=recipe_cache,
        ui_agent=ui_agent,
        triage_agent=triage_agent,
    )

    # Mock PageSession
    mock_context = AsyncMock()
    mock_page = AsyncMock()
    session = PageSession(mock_context, mock_page)
    session.goto = AsyncMock()
    session.execute_action = AsyncMock(return_value="#username-input")

    page_1 = PageObservation(
        url="http://localhost:8000/step1",
        pathname="/step1",
        title="Login",
        elements=[
            DOMElement(
                vid=1,
                tag="input",
                name="username",
                id="username-input",
                selector="#username-input",
                required=True,
            )
        ],
    )
    page_confirm = PageObservation(
        url="http://localhost:8000/confirmation",
        pathname="/confirmation",
        title="Success",
        body_text="Order confirmed! thank you for your submission.",
    )

    # Cold Run:
    # 1st observe -> page_1. LLM provides fill action.
    # 2nd observe -> page_confirm. Terminal page reached.
    session.observe = AsyncMock(side_effect=[page_1, page_confirm])
    mock_router.complete_structured.return_value = (
        AgentAction(action="fill", vid=1, value="Alice", reason="Enter name"),
        CompletionResult(text="{}", provider="mock", model="m", latency_ms=10.0),
    )

    tc_cold = TestCase(case_id="TC-COLD", inputs={"username": "Alice"})
    res_cold = await agent.run_case(session, tc_cold)

    assert res_cold.status == "PASS"
    assert res_cold.cache_hits == 0
    assert mock_router.complete_structured.call_count == 1

    # Verify recipe was compiled in cache
    cached_recipe = recipe_cache.get("http://localhost:8000/step1")
    assert cached_recipe is not None
    assert len(cached_recipe.steps) == 1
    assert cached_recipe.steps[0].selector == "#username-input"
    assert cached_recipe.steps[0].input_key == "username"

    # Warm Run:
    # Reset LLM call count. The agent should now replay cached steps without invoking LLM!
    mock_router.complete_structured.reset_mock()
    session.observe = AsyncMock(side_effect=[page_1, page_confirm])

    tc_warm = TestCase(case_id="TC-WARM", inputs={"username": "Bob"})
    res_warm = await agent.run_case(session, tc_warm)

    assert res_warm.status == "PASS"
    assert res_warm.cache_hits == 1
    # Zero LLM completions on warm replay!
    assert mock_router.complete_structured.call_count == 0


@pytest.mark.asyncio
async def test_ui_agent_audits_once_per_signature_and_emits_findings(tmp_path) -> None:
    """Tests UIAgent deduplicating audits by signature and emitting FindingLogged events."""
    cfg = VettorConfig()
    cfg.model.enable_visual_audit = False
    bus = EventBus()
    memory = QAMemory(storage_path=tmp_path / "mem.json")
    mock_router = MagicMock(spec=ModelRouter)
    mock_router.profile = "small"

    logged_events: list[FindingLogged] = []
    bus.subscribe(lambda ev: logged_events.append(ev) if isinstance(ev, FindingLogged) else None)

    ui_agent = UIAgent(cfg, mock_router, bus, memory)

    mock_context = AsyncMock()
    mock_page = AsyncMock()
    mock_page.evaluate = AsyncMock(return_value=[
        {"kind": "layout_overflow", "severity": "P2", "detail": "Overflow on page"}
    ])
    session = PageSession(mock_context, mock_page)

    obs = PageObservation(
        url="http://localhost:8000/view",
        pathname="/view",
        title="View",
        elements=[DOMElement(vid=1, tag="button", selector="button#btn")],
    )

    # First audit: should evaluate probes and emit FindingLogged
    findings1 = await ui_agent.audit_page(session, obs, case_id="TC-10")
    assert len(findings1) == 1
    assert findings1[0].kind == "layout_overflow"
    assert len(logged_events) == 1
    assert logged_events[0].case_id == "TC-10"

    # Second audit on identical observation signature: should deduplicate and skip
    findings2 = await ui_agent.audit_page(session, obs, case_id="TC-10")
    assert len(findings2) == 0
    assert len(logged_events) == 1


@pytest.mark.asyncio
async def test_triage_agent_diagnoses_heal_with_vid(tmp_path) -> None:
    """Tests TriageAgent returning a heal decision with actionable integer vid."""
    cfg = VettorConfig()
    bus = EventBus()
    memory = QAMemory(storage_path=tmp_path / "mem.json")
    mock_router = MagicMock(spec=ModelRouter)

    expected_decision = TriageDecision(
        diagnosis="heal",
        reason="Dismiss overlay cookie consent banner",
        action="click",
        vid=7,
        severity="P2",
    )
    mock_router.complete_structured = AsyncMock(return_value=(
        expected_decision,
        CompletionResult(text="{}", provider="mock", model="m", latency_ms=15.0),
    ))

    triage_agent = TriageAgent(cfg, mock_router, bus, memory)

    obs = PageObservation(
        url="http://localhost:8000/checkout",
        elements=[DOMElement(vid=7, tag="button", label="Accept Cookies", selector="#accept-btn")],
    )
    tc = TestCase(case_id="TC-99", goal="Checkout item")

    decision = await triage_agent.diagnose(obs, tc, errors=["Cookie banner blocking interaction"])
    assert decision.diagnosis == "heal"
    assert decision.action == "click"
    assert decision.vid == 7
    assert "cookie" in decision.reason.lower()


@pytest.mark.asyncio
async def test_test_case_agent_plans_from_page(tmp_path) -> None:
    """Tests TestCaseAgent generating a structured list of TestCase objects from a page observation."""
    cfg = VettorConfig()
    bus = EventBus()
    memory = QAMemory(storage_path=tmp_path / "mem.json")
    mock_router = MagicMock(spec=ModelRouter)

    generated = GeneratedSuite(cases=[
        TestCase(case_id="GEN-1", name="Happy Path Registration", goal="Complete signup", inputs={"email": "alice@example.com"}),
        TestCase(case_id="GEN-2", name="Negative Boundary", goal="Reject blank email", inputs={"email": ""}),
    ])
    mock_router.complete_structured = AsyncMock(return_value=(
        generated,
        CompletionResult(text="{}", provider="mock", model="m", latency_ms=20.0),
    ))

    planner = TestCaseAgent(cfg, mock_router, bus, memory)

    mock_context = AsyncMock()
    mock_page = AsyncMock()
    session = PageSession(mock_context, mock_page)
    session.goto = AsyncMock()
    session.observe = AsyncMock(return_value=PageObservation(url="http://localhost:8000/signup"))

    cases = await planner.plan_from_page(session, "http://localhost:8000/signup", instruction="Focus on email validation")

    assert len(cases) == 2
    assert cases[0].case_id == "GEN-1"
    assert cases[0].inputs == {"email": "alice@example.com"}
    assert cases[1].case_id == "GEN-2"
    assert cases[1].inputs == {"email": ""}
