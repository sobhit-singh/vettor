from __future__ import annotations

import base64
import html
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vettor.config import VettorConfig
from vettor.core.schema import CaseResult


def _esc(val: Any) -> str:
    return html.escape(str(val) if val is not None else "", quote=True)


def generate_html_report(
    results: list[CaseResult],
    cfg: VettorConfig,
    duration_s: float,
    output_path: str | Path,
) -> Path:
    """Generates a self-contained, dark-mode HTML execution report with KPI cards and embedded evidence."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    total = len(results)
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    errors = sum(1 for r in results if r.status == "ERROR")
    cache_hits = sum(r.cache_hits for r in results)
    heals = sum(r.heals for r in results)
    pass_pct = round((passed / total * 100) if total else 0.0, 1)

    table_rows: list[str] = []
    for r in results:
        badge_class = {
            "PASS": "badge-pass",
            "FAIL": "badge-fail",
            "ERROR": "badge-error",
        }.get(r.status, "badge-unknown")

        scr_html = ""
        if r.screenshot:
            # Check if screenshot is a file path or base64 string
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

        row = f"""
        <tr>
          <td><strong>{_esc(r.case_id)}</strong></td>
          <td><span class="badge {badge_class}">{_esc(r.status)}</span></td>
          <td>{r.duration_s:.2f}s</td>
          <td>{r.steps}</td>
          <td>{r.cache_hits}</td>
          <td>{r.heals}</td>
          <td>{_esc(extracted_str)}</td>
          <td>{findings_badge}</td>
          <td class="note-cell">{_esc(r.note or "-")}</td>
          <td>{scr_html}</td>
        </tr>
        """
        table_rows.append(row)

    now_iso = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    html_content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Vettor QA Execution Report</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --pass: #10b981;
      --fail: #ef4444;
      --error: #f59e0b;
      --accent: #3b82f6;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      margin: 0;
      padding: 32px 24px;
    }}
    .container {{
      max-width: 1200px;
      margin: 0 auto;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 24px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 16px;
    }}
    h1 {{
      margin: 0;
      font-size: 1.75rem;
      font-weight: 700;
    }}
    .timestamp {{
      color: var(--text-muted);
      font-size: 0.875rem;
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .kpi-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      text-align: center;
    }}
    .kpi-val {{
      font-size: 1.75rem;
      font-weight: 700;
      margin-top: 4px;
    }}
    .kpi-label {{
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--text-muted);
    }}
    .val-pass {{ color: var(--pass); }}
    .val-fail {{ color: var(--fail); }}
    .val-error {{ color: var(--error); }}
    .meta-box {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 24px;
      font-size: 0.875rem;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 12px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--card-bg);
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid var(--border);
    }}
    th, td {{
      padding: 12px 14px;
      text-align: left;
      border-bottom: 1px solid var(--border);
      font-size: 0.875rem;
    }}
    th {{
      background: #182234;
      color: var(--text-muted);
      font-weight: 600;
      text-transform: uppercase;
      font-size: 0.75rem;
    }}
    .badge {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 0.75rem;
      font-weight: 600;
    }}
    .badge-pass {{ background: rgba(16, 185, 129, 0.2); color: var(--pass); }}
    .badge-fail {{ background: rgba(239, 68, 68, 0.2); color: var(--fail); }}
    .badge-error {{ background: rgba(245, 158, 11, 0.2); color: var(--error); }}
    .badge-findings {{ background: #3b82f633; color: #60a5fa; }}
    .thumb {{
      max-width: 60px;
      max-height: 40px;
      border-radius: 4px;
      border: 1px solid var(--border);
    }}
    .note-cell {{
      max-width: 200px;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div>
        <h1>Vettor QA Harness Report</h1>
        <div class="timestamp">Generated on {now_iso}</div>
      </div>
      <div>
        <span class="badge {'badge-pass' if pass_pct >= 90 else 'badge-fail'}" style="font-size: 1rem; padding: 6px 12px;">{pass_pct}% Passed</span>
      </div>
    </header>

    <div class="kpi-grid">
      <div class="kpi-card"><div class="kpi-label">Total Cases</div><div class="kpi-val">{total}</div></div>
      <div class="kpi-card"><div class="kpi-label">Passed</div><div class="kpi-val val-pass">{passed}</div></div>
      <div class="kpi-card"><div class="kpi-label">Failed</div><div class="kpi-val val-fail">{failed}</div></div>
      <div class="kpi-card"><div class="kpi-label">Errors</div><div class="kpi-val val-error">{errors}</div></div>
      <div class="kpi-card"><div class="kpi-label">Cache Hits</div><div class="kpi-val">{cache_hits}</div></div>
      <div class="kpi-card"><div class="kpi-label">Heals</div><div class="kpi-val">{heals}</div></div>
      <div class="kpi-card"><div class="kpi-label">Duration</div><div class="kpi-val">{duration_s:.1f}s</div></div>
    </div>

    <div class="meta-box">
      <div><strong>Target URL:</strong> {_esc(cfg.target.url)}</div>
      <div><strong>Model Backend:</strong> {_esc(cfg.model.backend)} ({_esc(cfg.model.model_name)})</div>
      <div><strong>Browser Engine:</strong> {_esc(cfg.browser.engine)} ({_esc(cfg.browser.browser_type)})</div>
      <div><strong>Suite Mode:</strong> {_esc(cfg.suite.mode)}</div>
    </div>

    <table>
      <thead>
        <tr>
          <th>Case ID</th>
          <th>Status</th>
          <th>Duration</th>
          <th>Steps</th>
          <th>Hits</th>
          <th>Heals</th>
          <th>Extracted</th>
          <th>Findings</th>
          <th>Note</th>
          <th>Evidence</th>
        </tr>
      </thead>
      <tbody>
        {"".join(table_rows) if table_rows else "<tr><td colspan='10' style='text-align:center;'>No test cases executed</td></tr>"}
      </tbody>
    </table>
  </div>
</body>
</html>
"""
    out_file.write_text(html_content, encoding="utf-8")
    return out_file
