import csv
import json
import logging
import os
from pathlib import Path
import time
import urllib.request

from config import QAConfig
from agent import QAAgent
from engine import QAEngine
from mockinsurancesite import start_server_in_thread
import llm_manager

logging.basicConfig(level=logging.WARNING)

BENCHMARK_CASES = [
    {"case_id": "1", "full_name": "Aarav Sharma", "age": "30", "smoker": "no", "coverage_amount": "500000", "expected_premium": "345.00"},
    {"case_id": "2", "full_name": "Priya Nair", "age": "45", "smoker": "yes", "coverage_amount": "1000000", "expected_premium": "955.00"},
    {"case_id": "3", "full_name": "Rohan Verma", "age": "28", "smoker": "no", "coverage_amount": "750000", "expected_premium": "462.00"},
    {"case_id": "4", "full_name": "Kavita Rao", "age": "22", "smoker": "no", "coverage_amount": "500000", "expected_premium": "313.00"},
    {"case_id": "5", "full_name": "Vikram Patel", "age": "55", "smoker": "yes", "coverage_amount": "1000000", "expected_premium": "995.00"},
    {"case_id": "6", "full_name": "Ananya Sen", "age": "35", "smoker": "yes", "coverage_amount": "750000", "expected_premium": "790.00"},
    {"case_id": "7", "full_name": "Sunil Joshi", "age": "60", "smoker": "no", "coverage_amount": "500000", "expected_premium": "465.00"},
    {"case_id": "8", "full_name": "Meera Iyer", "age": "25", "smoker": "no", "coverage_amount": "1000000", "expected_premium": "575.00"},
    {"case_id": "9", "full_name": "Deepak Gupta", "age": "40", "smoker": "yes", "coverage_amount": "500000", "expected_premium": "685.00"},
    {"case_id": "10", "full_name": "Sita Das", "age": "50", "smoker": "no", "coverage_amount": "750000", "expected_premium": "550.00"},
]


def write_benchmark_csv(path: str = "benchmark_cases.csv"):
    fields = ["case_id", "full_name", "age", "smoker", "coverage_amount", "expected_premium"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(BENCHMARK_CASES)


def benchmark_model_inference(agent: QAAgent) -> dict:
    print("\n--- 1. BENCHMARKING LFM3-1.7B MODEL INFERENCE ---", flush=True)
    tools = [
        (
            "map_page_schema",
            {
                "csv_columns": ["full_name", "age", "smoker", "coverage_amount"],
                "fields": [
                    {"tag": "input", "id": "full_name", "selector": "#full_name", "type": "text", "label": "Full Name", "required": True},
                    {"tag": "input", "id": "age", "selector": "#age", "type": "number", "label": "Age", "required": True},
                    {"tag": "button", "id": "btn-next", "selector": "#btn-next", "label": "Next Step", "type": "submit"},
                ],
            },
        ),
        (
            "resolve_delta_fields",
            {
                "row": {"full_name": "Priya Nair", "age": "45", "smoker": "yes", "cigs_per_day": "15"},
                "fields": [
                    {"tag": "input", "id": "cigs_per_day", "selector": "#cigs_per_day", "type": "number", "label": "Cigarettes Per Day", "required": True},
                ],
                "failed_col": "cigs_per_day",
            },
        ),
        (
            "triage_page_blocker",
            {
                "errors": ["You must accept the terms to calculate a quote."],
                "fields": [
                    {"tag": "input", "id": "agree_terms", "selector": "#agree_terms", "type": "checkbox", "label": "I confirm all details."},
                    {"tag": "button", "id": "btn-submit", "selector": "#btn-submit", "label": "Calculate Premium", "type": "submit"},
                ],
                "row": {"case_id": "1"},
            },
        ),
        (
            "extract_final_result",
            {
                "bodyText": "Your Insurance Quote Summary Applicant: Aarav Sharma Quote Reference: Q-7A8B9C Annual Premium: $345.00"
            },
        ),
    ]

    results = {}
    total_time = 0.0
    for name, payload in tools:
        t0 = time.perf_counter()
        res = agent.call_tool(name, payload)
        dur = round((time.perf_counter() - t0) * 1000, 1)
        total_time += dur
        results[name] = {"latency_ms": dur, "success": bool(res)}
        print(f"  • {name:22s} -> {dur:6.1f} ms | OK: {bool(res)}", flush=True)

    avg_latency = round(total_time / len(tools), 1)
    print(f"  --> Average Decision Latency: {avg_latency} ms", flush=True)
    return {"per_tool": results, "avg_latency_ms": avg_latency}


def benchmark_engine_run(server_port: int, engine_name: str, csv_path: str, cold: bool) -> dict:
    mode_name = "Cold (Model-Discovery)" if cold else "Warm (Recipe-Replay)"
    print(f"\n--- Running: {engine_name.upper()} | {mode_name} ---", flush=True)

    recipe_path = f"benchmark_recipe_{engine_name}.json"
    results_path = f"benchmark_results_{engine_name}_{'cold' if cold else 'warm'}.csv"

    if cold and Path(recipe_path).exists():
        Path(recipe_path).unlink()

    cfg = QAConfig(
        engine=engine_name,
        start_url=f"http://127.0.0.1:{server_port}/step1",
        csv_path=csv_path,
        recipe_path=recipe_path,
        results_path=results_path,
        headless=True,
    )

    t0 = time.perf_counter()
    engine = QAEngine(cfg)
    stats = engine.run_all()
    total_dur = round(time.perf_counter() - t0, 2)

    pass_rate = round((stats.passed / stats.total * 100) if stats.total else 0, 1)
    avg_per_case = round(total_dur / stats.total, 2) if stats.total else 0

    print(f"  • Total Time:     {total_dur}s for {stats.total} test cases", flush=True)
    print(f"  • Avg per Case:   {avg_per_case}s", flush=True)
    print(f"  • Passed:         {stats.passed}/{stats.total} ({pass_rate}%)", flush=True)
    print(f"  • Cache Hits:     {stats.cache_hits}", flush=True)
    print(f"  • Self-Heals:     {stats.heals}", flush=True)
    print(f"  • Execution Errs: {stats.errors}", flush=True)

    return {
        "engine": engine_name,
        "mode": "cold" if cold else "warm",
        "total_time_s": total_dur,
        "avg_case_time_s": avg_per_case,
        "passed": stats.passed,
        "total": stats.total,
        "pass_rate": pass_rate,
        "cache_hits": stats.cache_hits,
        "heals": stats.heals,
        "errors": stats.errors,
    }


def main():
    print("=================================================================", flush=True)
    print("       VETTOR BENCHMARK SUITE (LFM3-1.7B + BROWSER)       ", flush=True)
    print("=================================================================", flush=True)

    # 1. Setup mock server & benchmark CSV
    httpd, thread = start_server_in_thread(port=0)
    port = httpd.server_address[1]
    csv_file = "benchmark_cases.csv"
    write_benchmark_csv(csv_file)
    print(f"Mock server running on port {port} | Test dataset: {len(BENCHMARK_CASES)} cases", flush=True)

    try:
        # 2. Check & ensure model server
        cfg = QAConfig()
        agent = QAAgent(cfg)
        model_metrics = benchmark_model_inference(agent)

        # 3. Benchmark Playwright (Cold & Warm)
        # For cold, run 3 cases to build full recipe
        cold_cases_file = "benchmark_cases_cold.csv"
        with open(cold_cases_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(BENCHMARK_CASES[0].keys()))
            writer.writeheader()
            writer.writerows(BENCHMARK_CASES[:3])

        pw_cold = benchmark_engine_run(port, "playwright", cold_cases_file, cold=True)
        pw_warm = benchmark_engine_run(port, "playwright", csv_file, cold=False)

        # 4. Benchmark Rustwright (Cold & Warm)
        rw_cold = benchmark_engine_run(port, "rustwright", cold_cases_file, cold=True)
        rw_warm = benchmark_engine_run(port, "rustwright", csv_file, cold=False)

        # 5. Compute comparisons
        speedup_pw = round(pw_cold["avg_case_time_s"] / pw_warm["avg_case_time_s"], 1) if pw_warm["avg_case_time_s"] else 0
        speedup_rw = round(rw_cold["avg_case_time_s"] / rw_warm["avg_case_time_s"], 1) if rw_warm["avg_case_time_s"] else 0
        engine_speedup_warm = round(pw_warm["total_time_s"] / rw_warm["total_time_s"], 2) if rw_warm["total_time_s"] else 0

        print("\n=================================================================", flush=True)
        print("                      BENCHMARK SUMMARY                          ", flush=True)
        print("=================================================================", flush=True)
        print(f"Model Inference Avg Latency: {model_metrics['avg_latency_ms']} ms / decision")
        print(f"Playwright Cold (Model Discovery): {pw_cold['avg_case_time_s']}s/case  --> Warm (Cache): {pw_warm['avg_case_time_s']}s/case ({speedup_pw}x faster)")
        print(f"Rustwright Cold (Model Discovery): {rw_cold['avg_case_time_s']}s/case  --> Warm (Cache): {rw_warm['avg_case_time_s']}s/case ({speedup_rw}x faster)")
        print(f"Warm Replay Engine Comparison: Rustwright is {engine_speedup_warm}x faster than Playwright")
        print("=================================================================\n", flush=True)

        # Clean temporary test CSVs
        for f in [csv_file, cold_cases_file]:
            if Path(f).exists():
                Path(f).unlink()

    finally:
        httpd.shutdown()
        httpd.server_close()


if __name__ == "__main__":
    main()
