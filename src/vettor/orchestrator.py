from __future__ import annotations

import asyncio
import csv
import json
import logging
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from vettor.agents.functional import FunctionalityAgent
from vettor.agents.planner import TestCaseAgent
from vettor.agents.triage import TriageAgent
from vettor.agents.ui import UIAgent
from vettor.config import VettorConfig
from vettor.core.events import EventBus, RunFinished, RunStarted
from vettor.core.llm import ModelRouter
from vettor.core.memory import QAMemory
from vettor.core.recipe import RecipeCache
from vettor.core.schema import CaseResult, TestCase
from vettor.env.browser import BrowserPool
from vettor.reporting.assertions import evaluate_assertions
from vettor.reporting.handoff import HandoffManager, stream_result_jsonl
from vettor.reporting.html_report import generate_html_report

logger = logging.getLogger("vettor.orchestrator")


class RunSummary(BaseModel):
    """Execution telemetry and outcome metrics of an orchestrated QA run."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    cache_hits: int = 0
    heals: int = 0
    findings: int = 0
    duration_s: float = 0.0
    results: list[CaseResult] = Field(default_factory=list)


class Orchestrator:
    """Async execution coordinator managing worker pools, agents, and artifact reporting."""

    def __init__(
        self,
        cfg: VettorConfig,
        bus: EventBus | None = None,
        hitl_callback: Callable[[dict[str, Any]], Awaitable[dict[str, Any] | None]] | None = None,
    ):
        self.cfg = cfg
        self.bus = bus or EventBus()
        self.hitl_callback = hitl_callback
        self.stop_requested = False

        self.router = ModelRouter(cfg)
        self.recipe_cache = RecipeCache(cfg.reporting.recipe_path)
        self.memory = QAMemory(cfg.reporting.memory_path)
        self.handoff = HandoffManager(cfg.reporting.handoff_path)

        self.ui_agent = UIAgent(cfg, self.router, self.bus, self.memory)
        self.triage_agent = TriageAgent(cfg, self.router, self.bus, self.memory)
        self.functional_agent = FunctionalityAgent(
            cfg=cfg,
            router=self.router,
            bus=self.bus,
            memory=self.memory,
            recipe_cache=self.recipe_cache,
            ui_agent=self.ui_agent,
            triage_agent=self.triage_agent,
            hitl_callback=self.hitl_callback,
        )
        self.planner = TestCaseAgent(cfg, self.router, self.bus, self.memory)

    def load_cases(self, path: str | Path | None = None) -> list[TestCase]:
        """Loads test cases from a .jsonl, .json, or .csv file, normalizing each row."""
        raw_path = path or self.cfg.suite.cases_path
        p = Path(raw_path)
        if not p.exists():
            for ext in (".jsonl", ".csv", ".json"):
                cand = p.with_suffix(ext)
                if cand.exists():
                    p = cand
                    break

        if not p.exists():
            logger.warning("Cases file not found: %s", raw_path)
            return []

        ext = p.suffix.lower()
        cases: list[TestCase] = []

        if ext == ".jsonl":
            for idx, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
                line = line.strip()
                if line:
                    try:
                        row = json.loads(line)
                        cases.append(TestCase.from_flat_row(row, default_id=str(idx + 1)))
                    except Exception as exc:
                        logger.warning("Failed parsing line %d in %s: %s", idx, p, exc)

        elif ext == ".json":
            try:
                raw_json = json.loads(p.read_text(encoding="utf-8"))
                rows = raw_json if isinstance(raw_json, list) else [raw_json]
                for idx, row in enumerate(rows):
                    if isinstance(row, dict):
                        cases.append(TestCase.from_flat_row(row, default_id=str(idx + 1)))
            except Exception as exc:
                logger.warning("Failed parsing JSON file %s: %s", p, exc)

        elif ext == ".csv":
            try:
                with p.open("r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for idx, row in enumerate(reader):
                        cases.append(TestCase.from_flat_row(dict(row), default_id=str(idx + 1)))
            except Exception as exc:
                logger.warning("Failed parsing CSV file %s: %s", p, exc)

        return cases

    async def plan_suite(self, url: str, instruction: str = "") -> list[TestCase]:
        """Explores the target URL and generates a structured test suite."""
        async with BrowserPool(self.cfg) as pool:
            session = await pool.new_session()
            try:
                return await self.planner.plan_from_page(session, url, instruction)
            finally:
                await session.close()

    async def run(self, cases: list[TestCase] | None = None) -> RunSummary:
        """Executes the test suite according to the configured mode (data_driven, autonomous_goal, crawl_smoke)."""
        t0 = time.monotonic()
        mode = self.cfg.suite.mode

        test_cases: list[TestCase] = []
        if cases is not None:
            test_cases = cases
        elif mode == "data_driven":
            test_cases = self.load_cases()
        elif mode == "autonomous_goal":
            test_cases = [TestCase(case_id="goal-1", name="Autonomous Goal", goal=self.cfg.suite.goal)]
        elif mode == "crawl_smoke":
            test_cases = await self.plan_suite(self.cfg.target.url, instruction="Smoke test key workflows")

        await self.bus.emit(RunStarted(total_cases=len(test_cases), metadata={"mode": mode}))

        results: list[CaseResult] = []
        concurrency = max(1, self.cfg.browser.workers)
        sem = asyncio.Semaphore(concurrency)

        async with BrowserPool(self.cfg) as pool:

            async def run_single(tc: TestCase) -> CaseResult:
                if self.stop_requested:
                    return CaseResult(case_id=tc.case_id, status="ERROR", note="Run cancelled by user")

                async with sem:
                    session = await pool.new_session()
                    try:
                        res = await self.functional_agent.run_case(session, tc)

                        # Evaluate assertions if test case reached verification
                        final_obs = await session.observe()
                        assertion_results = evaluate_assertions(
                            tc,
                            final_obs,
                            res.extracted,
                            float_tolerance=self.cfg.reporting.float_tolerance,
                        )
                        res.assertions = assertion_results

                        # If any assertion failed, case fails
                        if any(not a.passed for a in assertion_results):
                            res.status = "FAIL"

                        # Log findings to handoff
                        for f in res.findings:
                            self.handoff.record_finding(f, case_id=tc.case_id, url=final_obs.url)

                        # Stream JSONL if enabled
                        if self.cfg.reporting.stream_jsonl:
                            stream_result_jsonl(self.cfg.reporting.output_dir, res)

                        return res
                    except Exception as exc:
                        logger.error("Unhandled error running case %s: %s", tc.case_id, exc)
                        return CaseResult(
                            case_id=tc.case_id,
                            status="ERROR",
                            note=f"Orchestrator error: {exc}",
                        )
                    finally:
                        await session.close()

            # Execute cases
            tasks = [asyncio.create_task(run_single(tc)) for tc in test_cases]
            for task in asyncio.as_completed(tasks):
                res = await task
                results.append(res)
                if self.stop_requested:
                    break

        duration_s = round(time.monotonic() - t0, 3)

        # Aggregate telemetry
        passed = sum(1 for r in results if r.status == "PASS")
        failed = sum(1 for r in results if r.status == "FAIL")
        errors = sum(1 for r in results if r.status == "ERROR")
        cache_hits = sum(r.cache_hits for r in results)
        heals = sum(r.heals for r in results)
        total_findings = sum(len(r.findings) for r in results)

        summary = RunSummary(
            total=len(results),
            passed=passed,
            failed=failed,
            errors=errors,
            cache_hits=cache_hits,
            heals=heals,
            findings=total_findings,
            duration_s=duration_s,
            results=results,
        )

        # Export CSV summary
        try:
            csv_path = Path(self.cfg.reporting.results_path)
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["case_id", "status", "duration_s", "steps", "cache_hits", "heals", "findings", "note"])
                for r in results:
                    writer.writerow([r.case_id, r.status, f"{r.duration_s:.2f}", r.steps, r.cache_hits, r.heals, len(r.findings), r.note])
        except Exception as exc:
            logger.warning("Failed writing CSV results: %s", exc)

        # Export JSON summary
        try:
            out_dir = Path(self.cfg.reporting.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            json_path = out_dir / "results.json"
            json_path.write_text(summary.model_dump_json(indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed writing JSON results: %s", exc)

        # Export HTML report if enabled
        if self.cfg.reporting.generate_html_report:
            try:
                html_path = Path(self.cfg.reporting.output_dir) / "report.html"
                generate_html_report(results, self.cfg, duration_s, html_path)
            except Exception as exc:
                logger.warning("Failed generating HTML report: %s", exc)

        await self.bus.emit(RunFinished(stats=summary.model_dump()))
        await self.router.aclose()
        return summary
