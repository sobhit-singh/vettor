from __future__ import annotations

import asyncio

import pytest
from textual.widgets import Input

from vettor.config import VettorConfig
from vettor.core.events import CaseFinished, CaseStarted, EventBus, FindingLogged
from vettor.core.schema import CaseResult, Finding
from vettor.tui.app import VettorApp
from vettor.tui.widgets import HITLBar, TelemetryBar, TranscriptLog


@pytest.mark.asyncio
async def test_tui_slash_model_command():
    cfg = VettorConfig()
    app = VettorApp(cfg=cfg)
    async with app.run_test(size=(120, 40)) as pilot:
        inp = app.query_one("#command-input", Input)
        inp.value = "/model gpt-4o --profile frontier"
        await pilot.press("enter")
        assert cfg.model.model_name == "gpt-4o"
        assert cfg.model.profile == "frontier"


@pytest.mark.asyncio
async def test_tui_slash_workers_command():
    cfg = VettorConfig()
    app = VettorApp(cfg=cfg)
    async with app.run_test(size=(120, 40)) as pilot:
        inp = app.query_one("#command-input", Input)
        inp.value = "/workers 4"
        await pilot.press("enter")
        assert cfg.browser.workers == 4


@pytest.mark.asyncio
async def test_tui_slash_toggles():
    cfg = VettorConfig()
    app = VettorApp(cfg=cfg)
    async with app.run_test(size=(120, 40)) as pilot:
        inp = app.query_one("#command-input", Input)

        inp.value = "/visual on"
        await pilot.press("enter")
        assert cfg.model.enable_visual_audit is True

        inp.value = "/private on"
        await pilot.press("enter")
        assert cfg.model.confidential is True

        inp.value = "/headless off"
        await pilot.press("enter")
        assert cfg.browser.headless is False


@pytest.mark.asyncio
async def test_tui_telemetry_event_updates():
    cfg = VettorConfig()
    bus = EventBus()
    app = VettorApp(cfg=cfg, bus=bus)
    async with app.run_test(size=(120, 40)) as pilot:
        tel = app.query_one("#telemetry-bar", TelemetryBar)

        # 1. Emit CaseStarted
        await bus.emit(CaseStarted(case_id="TC-1", name="Checkout Test"))
        await pilot.pause()

        # 2. Emit FindingLogged
        await bus.emit(
            FindingLogged(
                case_id="TC-1",
                finding=Finding(kind="contrast", severity="P1", detail="Low contrast button"),
            )
        )
        await pilot.pause()
        assert tel.findings == 1

        # 3. Emit CaseFinished
        await bus.emit(
            CaseFinished(
                case_id="TC-1",
                status="PASS",
                result=CaseResult(case_id="TC-1", status="PASS", duration_s=1.5, cache_hits=2),
            )
        )
        await pilot.pause()
        assert tel.passed == 1
        assert tel.cache_hits == 2


@pytest.mark.asyncio
async def test_tui_slash_help():
    cfg = VettorConfig()
    app = VettorApp(cfg=cfg)
    async with app.run_test(size=(120, 40)) as pilot:
        inp = app.query_one("#command-input", Input)
        log = app.query_one("#transcript-log", TranscriptLog)

        inp.value = "/help"
        await pilot.press("enter")
        await pilot.pause()

        # Transcript log should contain command descriptions
        content = "\n".join(strip.text for strip in log.lines)
        assert "/model" in content
        assert "/run" in content
        assert "Natural Language Goals" in content


@pytest.mark.asyncio
async def test_tui_autonomous_goal_dispatch():
    cfg = VettorConfig()
    app = VettorApp(cfg=cfg)
    async with app.run_test(size=(120, 40)) as pilot:
        inp = app.query_one("#command-input", Input)
        inp.value = "Verify checkout works on http://localhost:8000/shop"
        await pilot.press("enter")
        await pilot.pause()

        assert cfg.suite.mode == "autonomous_goal"
        assert cfg.suite.goal == "Verify checkout works on http://localhost:8000/shop"


@pytest.mark.asyncio
async def test_tui_inline_hitl_interaction():
    cfg = VettorConfig()
    app = VettorApp(cfg=cfg)
    async with app.run_test(size=(120, 40)) as pilot:
        hitl = app.query_one("#hitl-bar", HITLBar)
        inp = app.query_one("#command-input", Input)

        assert hitl.display is False

        # Simulate agent triggering HITL callback asynchronously
        bridge_task = asyncio.create_task(
            app._hitl_bridge({
                "case_id": "TC-99",
                "reason": "Button not visible",
            })
        )
        await pilot.pause()
        assert hitl.display is True

        # User submits selector override in input
        inp.value = "vid:5=yes"
        await pilot.press("enter")
        await pilot.pause()

        res = await bridge_task
        assert res == {"vid": 5, "value": "yes"}
        assert hitl.display is False


@pytest.mark.asyncio
async def test_tui_slash_target_and_providers():
    cfg = VettorConfig()
    app = VettorApp(cfg=cfg)
    async with app.run_test(size=(120, 40)) as pilot:
        inp = app.query_one("#command-input", Input)

        # 1. /target https://staging.myapp.com
        inp.value = "/target https://staging.myapp.com"
        await pilot.press("enter")
        await pilot.pause()
        assert cfg.target.url == "https://staging.myapp.com"

        # 2. /provider add vps http://10.0.0.5:8080/v1 VPS_API_KEY
        inp.value = "/provider add vps http://10.0.0.5:8080/v1 VPS_API_KEY"
        await pilot.press("enter")
        await pilot.pause()
        assert "vps" in cfg.model.providers
        assert cfg.model.providers["vps"].base_url == "http://10.0.0.5:8080/v1"
        assert cfg.model.providers["vps"].api_key_env == "VPS_API_KEY"

        # 3. /model vps/lfm3-1.7b
        inp.value = "/model vps/lfm3-1.7b"
        await pilot.press("enter")
        await pilot.pause()
        assert cfg.model.active == "vps/lfm3-1.7b"
        assert cfg.model.parse_active() == ("vps", "lfm3-1.7b")

        # 4. /provider list
        log = app.query_one("#transcript-log", TranscriptLog)
        inp.value = "/provider list"
        await pilot.press("enter")
        await pilot.pause()
        content = "\n".join(strip.text for strip in log.lines)
        assert "vps" in content
        assert "[ACTIVE]" in content


@pytest.mark.asyncio
async def test_tui_run_without_target_shows_warning():
    cfg = VettorConfig()
    cfg.target.url = ""
    app = VettorApp(cfg=cfg)
    async with app.run_test(size=(120, 40)) as pilot:
        inp = app.query_one("#command-input", Input)
        log = app.query_one("#transcript-log", TranscriptLog)

        inp.value = "/run"
        await pilot.press("enter")
        await pilot.pause()

        content = "\n".join(strip.text for strip in log.lines)
        assert "No target set. Use /target <url> first." in content


def test_cli_default_launches_tui():
    from unittest.mock import patch

    from vettor.cli import main

    with patch("vettor.tui.app.VettorApp.run", return_value=0) as mock_run:
        ret = main([])
        assert ret == 0
        assert mock_run.called
