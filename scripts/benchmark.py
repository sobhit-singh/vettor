from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from tests.fixtures.mock_portal import start_server_in_thread  # noqa: E402

from vettor.config import VettorConfig  # noqa: E402
from vettor.core.schema import TestCase  # noqa: E402
from vettor.orchestrator import Orchestrator  # noqa: E402


async def run_eval_benchmark() -> dict[str, Any]:
    """Runs the v0.4 evaluation benchmark against the deterministic mock portal."""
    server, root_url = start_server_in_thread()
    base_url = f"{root_url}/step1"
    out_dir = Path("artifacts/eval_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        TestCase(
            case_id="EVAL-HAPPY-1",
            name="Valid Quote Calculation",
            start_url=base_url,
            inputs={"name": "Aarav Sharma", "age": "30", "agree_terms": "true", "plan": "standard", "coverage": "500000"},
            expected={"url": "/quote"},
            sector="functional",
        ),
        TestCase(
            case_id="EVAL-FALSE-PASS-CHECK",
            name="Wrong Expected Premium Must FAIL",
            start_url=base_url,
            inputs={"name": "Aarav Sharma", "age": "30", "agree_terms": "true", "plan": "standard", "coverage": "500000"},
            expected={"premium": "999.99"},
            sector="functional",
        ),
    ]

    try:
        cfg = VettorConfig()
        cfg.target.url = base_url
        cfg.model.active = "heuristic/zero-llm"
        cfg.reporting.output_dir = str(out_dir)
        cfg.reporting.recipe_path = str(out_dir / "recipe.json")
        cfg.reporting.memory_path = str(out_dir / "memory.json")
        cfg.reporting.results_path = str(out_dir / "results.csv")
        cfg.reporting.handoff_path = str(out_dir / "HANDOFF.md")

        t0 = time.perf_counter()
        summary = await Orchestrator(cfg).run(cases=cases)
        dur = round(time.perf_counter() - t0, 3)

        report = {
            "duration_s": dur,
            "total": summary.total,
            "passed": summary.passed,
            "failed": summary.failed,
            "errors": summary.errors,
            "findings": summary.findings,
            "results": [
                {"case_id": r.case_id, "status": r.status, "tier": r.tier_used, "steps": r.steps, "note": r.note}
                for r in summary.results
            ],
        }
        (out_dir / "eval_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        server.shutdown()


if __name__ == "__main__":
    res = asyncio.run(run_eval_benchmark())
    print(json.dumps(res, indent=2))
