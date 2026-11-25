from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import pytest

from vettor.cli import cmd_handoff, cmd_init, cmd_memory
from vettor.config import VettorConfig
from vettor.core.schema import AssertionResult, CaseResult, Finding, PageObservation, TestCase
from vettor.orchestrator import Orchestrator
from vettor.reporting.assertions import evaluate_assertions
from vettor.reporting.handoff import HandoffManager, stream_result_jsonl
from vettor.reporting.html_report import generate_html_report


def test_evaluate_assertions_numeric_text_and_url():
    tc = TestCase(
        case_id="case-1",
        name="Assertion Evaluation Test",
        expected={
            "url": "/quote",
            "message": "Calculated Successfully",
            "price": "250.00",
        },
    )

    obs = PageObservation(
        url="http://127.0.0.1:8000/quote?sid=123",
        title="Quote Summary",
        body_text="Status: Quote Calculated Successfully. Total: $250.00. Thank you.",
    )

    extracted = {
        "price": 250.004,  # Within 0.01 tolerance
        "message": "Quote Calculated Successfully",
    }

    results = evaluate_assertions(tc, obs, extracted, float_tolerance=0.01)
    assert len(results) == 3
    assert all(r.passed for r in results)

    # Test failure on out-of-tolerance numeric
    failing_extracted = {"price": 300.00}
    fail_results = evaluate_assertions(tc, obs, failing_extracted, float_tolerance=0.01)
    price_res = next(r for r in fail_results if r.key == "price")
    assert not price_res.passed
    assert "expected 250.0" in price_res.message


def test_evaluate_assertions_fallback_and_missing():
    tc = TestCase(
        case_id="case-2",
        expected={
            "nonexistent_token": "SecretCode123",
        },
    )
    obs = PageObservation(url="http://test.local", body_text="Hello World")
    results = evaluate_assertions(tc, obs, extracted={})
    assert len(results) == 1
    assert not results[0].passed
    assert "not found" in results[0].message.lower()


def test_handoff_manager_and_streaming(tmp_path: Path):
    handoff_file = tmp_path / "DEVELOPMENT_HANDOFF.md"
    mgr = HandoffManager(handoff_file)

    finding = Finding(
        severity="P1",
        kind="ui_contrast",
        detail="Submit button text contrast ratio is only 2.1:1",
        suggested_fix="Increase button background darkness to #1d4ed8",
        page="/step1",
    )

    # 1. Record finding
    added = mgr.record_finding(finding, case_id="TC-101", url="http://app/step1")
    assert added is True
    assert handoff_file.exists()

    # 2. Deduplication check (same finding should return False)
    duplicate_added = mgr.record_finding(finding, case_id="TC-101", url="http://app/step1")
    assert duplicate_added is False

    # 3. List open items
    items = mgr.list_open_items()
    assert len(items) == 1
    assert "Submit button text contrast ratio" in items[0]["detail"]
    assert items[0]["severity"] == "P1"

    # 4. Stream result JSONL
    out_dir = tmp_path / "artifacts"
    case_res = CaseResult(
        case_id="TC-101",
        status="PASS",
        duration_s=1.23,
        steps=3,
        findings=[finding],
    )
    jsonl_path = stream_result_jsonl(out_dir, case_res)
    assert jsonl_path.exists()
    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    parsed_res = CaseResult.model_validate_json(lines[0])
    assert parsed_res.case_id == "TC-101"
    assert parsed_res.status == "PASS"


def test_generate_html_report(tmp_path: Path):
    html_path = tmp_path / "report.html"
    cfg = VettorConfig()

    results = [
        CaseResult(
            case_id="C1",
            status="PASS",
            duration_s=2.5,
            steps=4,
            assertions=[AssertionResult(key="price", expected="150.00", actual="150.00", passed=True)],
        ),
        CaseResult(
            case_id="C2",
            status="FAIL",
            duration_s=3.1,
            steps=6,
            note="<script>alert('xss')</script> failed",
            findings=[
                Finding(severity="P0", kind="accessibility", detail="Missing <label> on inputs", suggested_fix="Add for= attributes")
            ],
            assertions=[AssertionResult(key="url", expected="/quote", actual="/step2", passed=False, message="URL mismatch")],
        ),
    ]

    generate_html_report(results, cfg, duration_s=5.6, output_path=html_path)
    assert html_path.exists()

    html_content = html_path.read_text(encoding="utf-8")
    assert "Vettor Multi-Sector Web QA Report" in html_content
    assert (tmp_path / "report.md").exists()
    # Dark mode styles check
    assert "#0f172a" in html_content or "#1e293b" in html_content
    # Results check
    assert "C1" in html_content
    assert "C2" in html_content
    # XSS escaping check
    assert "<script>alert('xss')</script>" not in html_content
    assert "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;" in html_content or "&lt;script&gt;" in html_content


def test_orchestrator_load_cases(tmp_path: Path):
    cfg = VettorConfig()
    orch = Orchestrator(cfg)

    # 1. JSONL file
    jsonl_file = tmp_path / "cases.jsonl"
    jsonl_file.write_text(
        '{"case_id": "J1", "name": "Jsonl Case", "name_val": "Alice", "age": 25, "expected_price": 200.0}\n'
        '{"name": "Implicit ID Case", "name_val": "Bob", "age": 30}\n',
        encoding="utf-8",
    )
    cases = orch.load_cases(jsonl_file)
    assert len(cases) == 2
    assert cases[0].case_id == "J1"
    assert cases[0].inputs["name_val"] == "Alice"
    assert cases[0].expected["price"] == 200.0
    assert cases[1].case_id == "2"

    # 2. JSON array file
    json_file = tmp_path / "cases.json"
    json_file.write_text(
        json.dumps([
            {"case_id": "A1", "name": "Array Case", "city": "Seattle", "expected_status": "ok"}
        ]),
        encoding="utf-8",
    )
    cases_json = orch.load_cases(json_file)
    assert len(cases_json) == 1
    assert cases_json[0].case_id == "A1"
    assert cases_json[0].inputs["city"] == "Seattle"

    # 3. CSV file
    csv_file = tmp_path / "cases.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "name", "full_name", "expected_url"])
        writer.writeheader()
        writer.writerow({"case_id": "C-1", "name": "CSV Row", "full_name": "Charlie", "expected_url": "/quote"})

    cases_csv = orch.load_cases(csv_file)
    assert len(cases_csv) == 1
    assert cases_csv[0].case_id == "C-1"
    assert cases_csv[0].inputs["full_name"] == "Charlie"
    assert cases_csv[0].expected["url"] == "/quote"


def test_cli_commands(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # 1. Test init
    class ArgsInit:
        force = False

    ret = cmd_init(ArgsInit())
    assert ret == 0
    assert (tmp_path / "vettor.json").exists()
    assert (tmp_path / "qa_memory.json").exists()
    assert (tmp_path / "DEVELOPMENT_HANDOFF.md").exists()

    # 2. Test memory inspect and clear
    class ArgsMem:
        config = str(tmp_path / "vettor.json")
        action = "inspect"

    assert cmd_memory(ArgsMem()) == 0

    ArgsMem.action = "clear"
    assert cmd_memory(ArgsMem()) == 0

    # 3. Test handoff
    class ArgsHandoff:
        config = str(tmp_path / "vettor.json")
        file = str(tmp_path / "DEVELOPMENT_HANDOFF.md")

    assert cmd_handoff(ArgsHandoff()) == 0


@pytest.mark.skipif(os.getenv("QA_RUN_E2E") != "1", reason="Set QA_RUN_E2E=1 to run full browser E2E")
@pytest.mark.asyncio
async def test_full_suite_run_mock_portal(tmp_path: Path):
    from tests.fixtures.mock_portal import start_server_in_thread

    server, base_url = start_server_in_thread()
    try:
        cfg = VettorConfig()
        cfg.target.url = f"{base_url}/step1"
        cfg.browser.headless = True
        cfg.browser.workers = 1
        cfg.model.backend = "heuristic"  # Heuristic fallback for deterministic offline run
        cfg.reporting.output_dir = str(tmp_path / "artifacts")
        cfg.reporting.results_path = str(tmp_path / "artifacts" / "results.csv")
        cfg.reporting.recipe_path = str(tmp_path / "artifacts" / "recipe.json")
        cfg.reporting.memory_path = str(tmp_path / "artifacts" / "qa_memory.json")
        cfg.reporting.handoff_path = str(tmp_path / "artifacts" / "DEVELOPMENT_HANDOFF.md")

        cases = [
            TestCase(
                case_id="E2E-1",
                name="Standard Pro Quote",
                inputs={"name": "Alex Smith", "age": 30, "plan": "Pro", "agree_terms": "yes"},
                expected={"url": "/quote", "price": "250.00"},
            ),
            TestCase(
                case_id="E2E-2",
                name="Basic Plan Quote",
                inputs={"name": "Bob Jones", "age": 40, "plan": "Basic", "agree_terms": "yes"},
                expected={"url": "/quote", "price": "150.00"},
            ),
        ]

        orch = Orchestrator(cfg)
        summary = await orch.run(cases=cases)
        assert summary.total == 2
        assert Path(cfg.reporting.results_path).exists()
        assert (Path(cfg.reporting.output_dir) / "report.html").exists()
    finally:
        server.shutdown()
        server.server_close()
