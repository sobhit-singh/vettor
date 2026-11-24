from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from vettor.config import VettorConfig
from vettor.core.schema import AgentAction, PageObservation
from vettor.env.browser import BrowserPool, PageSession
from vettor.env.observer import capture_som_screenshot, observe_page
from vettor.env.probes import run_ui_probes


@pytest.mark.asyncio
async def test_observe_page_maps_to_page_observation() -> None:
    """Tests observe_page parsing raw JS dictionary into PageObservation with DOMElement items."""
    mock_page = AsyncMock()
    mock_page.url = "http://localhost:8080/checkout"
    mock_page.evaluate.return_value = {
        "url": "http://localhost:8080/checkout",
        "pathname": "/checkout",
        "title": "Secure Checkout",
        "elements": [
            {
                "vid": 1,
                "tag": "input",
                "type": "text",
                "label": "Full Name",
                "name": "full_name",
                "id": "name-input",
                "selector": "#name-input",
                "required": True,
                "value": "",
                "checked": False,
                "options": [],
            },
            {
                "vid": 2,
                "tag": "select",
                "type": "select",
                "label": "Country",
                "name": "country",
                "id": "country-select",
                "selector": "#country-select",
                "required": False,
                "value": "US",
                "checked": False,
                "options": ["US", "CA", "UK"],
            },
        ],
        "dialogs": [],
        "errors": ["Name is required"],
        "body_text": "Secure Checkout Step 1",
    }

    obs = await observe_page(mock_page)

    assert isinstance(obs, PageObservation)
    assert obs.url == "http://localhost:8080/checkout"
    assert obs.pathname == "/checkout"
    assert obs.title == "Secure Checkout"
    assert len(obs.elements) == 2
    assert obs.elements[0].vid == 1
    assert obs.elements[0].label == "Full Name"
    assert obs.elements[0].required is True
    assert obs.elements[1].options == ["US", "CA", "UK"]
    assert obs.errors == ["Name is required"]

    compact = obs.to_compact_text()
    assert 'URL: /checkout | Title: Secure Checkout' in compact
    assert '[1] input(text) "Full Name" req val=""' in compact
    assert '[2] select "Country" options=[US, CA, UK] val="US"' in compact
    assert "Errors: Name is required" in compact


@pytest.mark.asyncio
async def test_capture_som_screenshot() -> None:
    """Tests injecting and clearing Set-of-Mark visual bounding boxes."""
    mock_page = AsyncMock()
    mock_page.screenshot.return_value = b"\x89PNG\r\n\x1a\nfake_image_bytes"

    png_bytes = await capture_som_screenshot(mock_page)
    assert png_bytes.startswith(b"\x89PNG")
    assert mock_page.evaluate.call_count == 2
    mock_page.screenshot.assert_awaited_once_with(type="png")


@pytest.mark.asyncio
async def test_run_ui_probes_converts_anomalies_and_errors() -> None:
    """Tests converting JS anomalies, HTTP network errors, and JS exceptions into typed Findings."""
    mock_page = AsyncMock()
    mock_page.evaluate.return_value = [
        {
            "kind": "layout_overflow",
            "severity": "P2",
            "detail": "Horizontal overflow: scrollWidth=1400 > innerWidth=1280",
        },
        {
            "kind": "broken_image",
            "severity": "P2",
            "detail": "Broken image: /assets/logo.png",
        },
    ]

    network_errors = [
        {"url": "http://localhost:8080/api/pay", "status": 500},
        {"url": "http://localhost:8080/assets/missing.css", "status": 404},
    ]
    page_errors = ["ReferenceError: window.analytics is not defined"]

    findings = await run_ui_probes(
        mock_page,
        page_path="/checkout",
        network_errors=network_errors,
        page_errors=page_errors,
    )

    assert len(findings) == 5

    # Check JS anomalies
    assert findings[0].kind == "layout_overflow"
    assert findings[0].severity == "P2"
    assert findings[0].page == "/checkout"

    assert findings[1].kind == "broken_image"
    assert findings[1].severity == "P2"

    # Check network errors: 500 should be P1, 404 should be P2
    assert findings[2].kind == "network_error"
    assert findings[2].severity == "P1"
    assert "500" in findings[2].detail

    assert findings[3].kind == "network_error"
    assert findings[3].severity == "P2"
    assert "404" in findings[3].detail

    # Check JS exceptions: should be P1
    assert findings[4].kind == "js_exception"
    assert findings[4].severity == "P1"
    assert "ReferenceError" in findings[4].detail


@pytest.mark.asyncio
async def test_page_session_execute_action_resolves_selector_and_controls() -> None:
    """Tests resolving vid to data-vettor-id selector and control-specific execution."""
    mock_context = AsyncMock()
    mock_page = AsyncMock()
    mock_locator = AsyncMock()
    mock_page.locator = MagicMock(return_value=mock_locator)

    session = PageSession(mock_context, mock_page)

    # 1. Action against checkbox with 'fill' action -> should call check()
    mock_page.evaluate.return_value = {"tag": "input", "type": "checkbox", "checked": False}
    action_chk = AgentAction(action="fill", vid=3, value="yes")
    selector_used = await session.execute_action(action_chk)

    assert selector_used == '[data-vettor-id="3"]'
    mock_locator.check.assert_awaited_once()

    # 2. Action against <select> with 'fill' action -> should call select_option()
    mock_locator.reset_mock()
    mock_page.evaluate.return_value = {"tag": "select", "type": "select-one", "checked": False}
    action_sel = AgentAction(action="fill", vid=5, value="California")
    selector_used = await session.execute_action(action_sel)

    assert selector_used == '[data-vettor-id="5"]'
    mock_locator.select_option.assert_awaited()

    # 3. Action against standard text input -> should call fill()
    mock_locator.reset_mock()
    mock_page.evaluate.return_value = {"tag": "input", "type": "text", "checked": False}
    action_text = AgentAction(action="fill", vid=1, value="Alice")
    selector_used = await session.execute_action(action_text)

    assert selector_used == '[data-vettor-id="1"]'
    mock_locator.fill.assert_awaited_once_with("Alice", timeout=4000)

    # 4. Action with wait -> should sleep without calling DOM locator
    mock_locator.reset_mock()
    action_wait = AgentAction(action="wait", value="50")
    res = await session.execute_action(action_wait)
    assert res == "wait"
    mock_locator.fill.assert_not_called()


@pytest.mark.asyncio
async def test_browser_pool_new_session_injects_storage() -> None:
    """Tests that BrowserPool.new_session injects localStorage and sessionStorage via add_init_script."""
    cfg = VettorConfig()
    cfg.target.local_storage = {"auth_token": "token123", "theme": "dark"}
    cfg.target.session_storage = {"session_id": "sess_999"}

    pool = BrowserPool(cfg)

    # Mock browser and context
    mock_browser = AsyncMock()
    mock_context = AsyncMock()
    mock_page = MagicMock()
    mock_page.on = MagicMock()

    mock_browser.new_context.return_value = mock_context
    mock_context.new_page = AsyncMock(return_value=mock_page)

    pool._browser = mock_browser

    session = await pool.new_session()
    assert isinstance(session, PageSession)

    # Verify add_init_script was called with both local and session storage values
    mock_context.add_init_script.assert_awaited_once()
    script_arg = mock_context.add_init_script.call_args[0][0]

    assert "auth_token" in script_arg
    assert "token123" in script_arg
    assert "session_id" in script_arg
    assert "sess_999" in script_arg


@pytest.mark.asyncio
async def test_page_session_goto_with_url_params() -> None:
    """Tests PageSession.goto merges url_params and calls page.goto with load wait_until."""
    mock_context = AsyncMock()
    mock_page = AsyncMock()
    session = PageSession(mock_context, mock_page)

    await session.goto(
        "http://localhost:8000/start?existing=1",
        url_params={"debug": "true", "seed": "42"},
        timeout_ms=10000,
    )

    mock_page.goto.assert_awaited_once()
    called_url, kwargs = mock_page.goto.call_args[0][0], mock_page.goto.call_args[1]
    assert "existing=1" in called_url
    assert "debug=true" in called_url
    assert "seed=42" in called_url
    assert kwargs.get("wait_until") == "load"
    assert kwargs.get("timeout") == 10000

