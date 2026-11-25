from __future__ import annotations

import base64
import html
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vettor.config import VettorConfig
from vettor.core.schema import CaseResult, Finding
from vettor.sectors.auditor import ALL_QA_SECTORS


def _esc(val: Any) -> str:
    return html.escape(str(val) if val is not None else "", quote=True)


def _compute_sector_summary(results: list[CaseResult]) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {
        sec: {"findings": 0, "p0": 0, "p1": 0, "p2": 0, "p3": 0, "cases": 0, "passed": 0}
        for sec in ALL_QA_SECTORS
    }
    for r in results:
        sec = r.sector if r.sector in summary else "functional"
        summary[sec]["cases"] += 1
        if r.status == "PASS":
            summary[sec]["passed"] += 1
        for f in r.findings:
            f_sec = f.sector if f.sector in summary else "ui"
            summary[f_sec]["findings"] += 1
            sev_key = f.severity.lower()
            if sev_key in summary[f_sec]:
                summary[f_sec][sev_key] += 1
    return summary


def generate_markdown_report(
    results: list[CaseResult],
    cfg: VettorConfig,
    duration_s: float,
    output_path: str | Path,
) -> Path:
    """Generates a full Markdown QA report with sector scorecards, test cases, and findings."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    total = len(results)
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    errors = sum(1 for r in results if r.status == "ERROR")
    pass_pct = round((passed / total * 100) if total else 0.0, 1)
    sector_stats = _compute_sector_summary(results)

    lines = [
        "# Vettor Autonomous Web QA Report",
        "",
        f"- **Generated:** {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"- **Target:** `{cfg.target.url or '(per-case)'}`",
        f"- **Model:** `{cfg.model.active}` (profile: `{cfg.model.profile}`)",
        f"- **Duration:** {duration_s:.2f}s",
        f"- **Summary:** {passed}/{total} PASSED ({pass_pct}%), {failed} FAILED, {errors} ERRORS",
        "",
        "## Sector-by-Sector QA Scorecard",
        "",
        "| Sector | Cases | Passed | Findings | P0 | P1 | P2 | P3 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for sec, st in sector_stats.items():
        lines.append(
            f"| {sec} | {st['cases']} | {st['passed']} | {st['findings']} | {st['p0']} | {st['p1']} | {st['p2']} | {st['p3']} |"
        )

    lines.extend(["", "## Test Case Execution Matrix", "", "| Case ID | Sector | Status | Tier | Steps | Cache | Heals | Note |", "| :--- | :--- | :--- | :--- | ---: | ---: | ---: | :--- |"])
    for r in results:
        lines.append(
            f"| `{r.case_id}` | {r.sector} | **{r.status}** | {r.tier_used or '-'} | {r.steps} | {r.cache_hits} | {r.heals} | {r.note or '-'} |"
        )

    all_findings: list[tuple[str, Finding]] = [(r.case_id, f) for r in results for f in r.findings]
    lines.extend(["", "## Findings & Defects Register", ""])
    if not all_findings:
        lines.append("No defects or sector anomalies detected.")
    else:
        for cid, f in all_findings:
            lines.append(f"- **[{f.severity}] ({f.sector}/{f.kind})** `{cid}` on `{f.page or '/'}`: {f.detail}")
            if f.suggested_fix:
                lines.append(f"  - *Suggested fix:* {f.suggested_fix}")

    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_file


def generate_html_report(
    results: list[CaseResult],
    cfg: VettorConfig,
    duration_s: float,
    output_path: str | Path,
) -> Path:
    """Generates a self-contained, dark-mode multi-sector HTML QA report with KPI cards and evidence."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Also write companion Markdown report
    generate_markdown_report(results, cfg, duration_s, out_file.with_suffix(".md"))

    total = len(results)
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    errors = sum(1 for r in results if r.status == "ERROR")
    cache_hits = sum(r.cache_hits for r in results)
    heals = sum(r.heals for r in results)
    model_calls = sum(r.model_calls for r in results)
    pass_pct = round((passed / total * 100) if total else 0.0, 1)
    sector_stats = _compute_sector_summary(results)

    sector_cards = []
    for sec, st in sector_stats.items():
        status_cls = "val-pass" if st["p0"] == 0 and st["p1"] == 0 else "val-fail"
        sector_cards.append(
            f'<div class="kpi-card"><div class="kpi-label">{_esc(sec)}</div>'
            f'<div class="kpi-val {status_cls}">{st["findings"]}</div>'
            f'<div class="timestamp">P0:{st["p0"]} P1:{st["p1"]} P2:{st["p2"]}</div></div>'
        )

    table_rows: list[str] = []
    finding_rows: list[str] = []
    for r in results:
        badge_class = {
            "PASS": "badge-pass",
            "FAIL": "badge-fail",
            "ERROR": "badge-error",
        }.get(r.status, "badge-unknown")

        scr_html = ""
        if r.screenshot:
            scr_p = Path(r.screenshot)
            if scr_p.exists():
                try:
                    b64 = base64.b64encode(scr_p.read_bytes()).decode("utf-8")
                    scr_html = f'<a href="data:image/png;base64,{b64}" target="_blank"><img class="thumb" src="data:image/png;base64,{b64}" alt="Screenshot" /></a>'
                except Exception:
                    scr_html = f'<a href="{_esc(r.screenshot)}" target="_blank">View File</a>'
            elif r.screenshot.startswith("data:image") or len(r.screenshot) > 100:
                scr_html = f'<a href="{_esc(r.screenshot)}" target="_blank"><img class="thumb" src="{_esc(r.screenshot)}" alt="Screenshot" /></a>'
            else:
                scr_html = f'<span class="text-muted">{_esc(r.screenshot)}</span>'

        extracted_str = ", ".join(f"{k}={v}" for k, v in r.extracted.items()) if r.extracted else "-"
        findings_count = len(r.findings)
        findings_badge = f'<span class="badge badge-findings">{findings_count}</span>' if findings_count else "-"

        for f in r.findings:
            finding_rows.append(
                f"<tr><td><strong>{_esc(f.severity)}</strong></td><td>{_esc(f.sector)}</td>"
                f"<td>{_esc(f.kind)}</td><td>{_esc(r.case_id)}</td><td>{_esc(f.page or '/')}</td>"
                f"<td>{_esc(f.detail)}</td><td>{_esc(f.suggested_fix or '-')}</td></tr>"
            )

        table_rows.append(
            f"<tr><td><strong>{_esc(r.case_id)}</strong></td><td>{_esc(r.sector)}</td>"
            f'<td><span class="badge {badge_class}">{_esc(r.status)}</span></td>'
            f"<td>{_esc(r.tier_used or '-')}</td><td>{r.duration_s:.2f}s</td><td>{r.steps}</td>"
            f"<td>{r.cache_hits}</td><td>{r.heals}</td><td>{_esc(extracted_str)}</td>"
            f'<td>{findings_badge}</td><td class="note-cell">{_esc(r.note or "-")}</td><td>{scr_html}</td></tr>'
        )

    now_iso = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    html_content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Vettor Multi-Sector QA Report</title>
  <style>
    :root {{ --bg: #0f172a; --card-bg: #1e293b; --border: #334155; --text: #f8fafc; --text-muted: #94a3b8; --pass: #10b981; --fail: #ef4444; --error: #f59e0b; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 32px 24px; }}
    .container {{ max-width: 1280px; margin: 0 auto; }}
    header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 24px; border-bottom: 1px solid var(--border); padding-bottom: 16px; }}
    h1 {{ margin: 0; font-size: 1.75rem; }}
    h2 {{ margin: 28px 0 12px; font-size: 1.2rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.04em; }}
    .timestamp {{ color: var(--text-muted); font-size: 0.85rem; }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 14px; margin-bottom: 20px; }}
    .kpi-card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 14px; text-align: center; }}
    .kpi-val {{ font-size: 1.6rem; font-weight: 700; margin-top: 4px; }}
    .kpi-label {{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-muted); }}
    .val-pass {{ color: var(--pass); }} .val-fail {{ color: var(--fail); }} .val-error {{ color: var(--error); }}
    .meta-box {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 8px; padding: 16px; margin-bottom: 24px; font-size: 0.875rem; display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px; }}
    table {{ width: 100%; border-collapse: collapse; background: var(--card-bg); border-radius: 8px; overflow: hidden; border: 1px solid var(--border); margin-bottom: 24px; }}
    th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); font-size: 0.85rem; }}
    th {{ background: #182234; color: var(--text-muted); font-weight: 600; text-transform: uppercase; font-size: 0.72rem; }}
    .badge {{ display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }}
    .badge-pass {{ background: rgba(16, 185, 129, 0.2); color: var(--pass); }}
    .badge-fail {{ background: rgba(239, 68, 68, 0.2); color: var(--fail); }}
    .badge-error {{ background: rgba(245, 158, 11, 0.2); color: var(--error); }}
    .badge-findings {{ background: #3b82f633; color: #60a5fa; }}
    .thumb {{ max-width: 60px; max-height: 40px; border-radius: 4px; border: 1px solid var(--border); }}
    .note-cell {{ max-width: 240px; word-break: break-word; color: var(--text-muted); }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div><h1>Vettor Multi-Sector Web QA Report</h1><div class="timestamp">Generated {now_iso}</div></div>
      <div><span class="badge {'badge-pass' if failed == 0 and errors == 0 else 'badge-fail'}">{pass_pct}% Pass Rate</span></div>
    </header>
    <div class="kpi-grid">
      <div class="kpi-card"><div class="kpi-label">Total Cases</div><div class="kpi-val">{total}</div></div>
      <div class="kpi-card"><div class="kpi-label">Passed</div><div class="kpi-val val-pass">{passed}</div></div>
      <div class="kpi-card"><div class="kpi-label">Failed</div><div class="kpi-val val-fail">{failed}</div></div>
      <div class="kpi-card"><div class="kpi-label">Errors</div><div class="kpi-val val-error">{errors}</div></div>
      <div class="kpi-card"><div class="kpi-label">Cache Hits</div><div class="kpi-val">{cache_hits}</div></div>
      <div class="kpi-card"><div class="kpi-label">Self-Heals</div><div class="kpi-val">{heals}</div></div>
      <div class="kpi-card"><div class="kpi-label">Model Calls</div><div class="kpi-val">{model_calls}</div></div>
      <div class="kpi-card"><div class="kpi-label">Duration</div><div class="kpi-val">{duration_s:.1f}s</div></div>
    </div>
    <div class="meta-box">
      <div><strong>Target URL:</strong> {_esc(cfg.target.url or '(per-case)')}</div>
      <div><strong>Active Model:</strong> {_esc(cfg.model.active)} ({_esc(cfg.model.profile)})</div>
      <div><strong>Browser:</strong> {_esc(cfg.browser.engine)} ({_esc(cfg.browser.browser_type)}, workers={cfg.browser.workers})</div>
      <div><strong>Privacy Lock:</strong> {'ENABLED' if cfg.model.confidential else 'DISABLED'}</div>
    </div>
    <h2>Sector Scorecard (Findings by QA Sector)</h2>
    <div class="kpi-grid">{"".join(sector_cards)}</div>
    <h2>Test Case Results</h2>
    <table>
      <thead><tr><th>Case ID</th><th>Sector</th><th>Status</th><th>Tier</th><th>Duration</th><th>Steps</th><th>Cache</th><th>Heals</th><th>Extracted</th><th>Findings</th><th>Note</th><th>Evidence</th></tr></thead>
      <tbody>{"".join(table_rows) if table_rows else '<tr><td colspan="12">No test cases executed.</td></tr>'}</tbody>
    </table>
    <h2>Findings &amp; Defects Register</h2>
    <table>
      <thead><tr><th>Severity</th><th>Sector</th><th>Kind</th><th>Case ID</th><th>Page</th><th>Detail</th><th>Suggested Fix</th></tr></thead>
      <tbody>{"".join(finding_rows) if finding_rows else '<tr><td colspan="7">No findings logged.</td></tr>'}</tbody>
    </table>
  </div>
</body>
</html>"""

    out_file.write_text(html_content, encoding="utf-8")
    return out_file
