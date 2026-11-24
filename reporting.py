import base64
from datetime import datetime
import html
import json
from pathlib import Path
from typing import Any


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def generate_html_report(run_data: dict[str, Any], output_path: str | Path) -> str:
    """Generates a standalone, beautiful HTML test execution report."""
    stats = run_data.get("stats", {})
    results = run_data.get("results", [])
    config = run_data.get("config", {})
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    total = stats.get("total", len(results))
    passed = stats.get("passed", 0)
    failed = stats.get("failed", 0)
    errors = stats.get("errors", 0)
    heals = stats.get("heals", 0)
    cache_hits = stats.get("cache_hits", 0)
    duration_s = stats.get("duration_s", 0.0)

    pass_pct = round((passed / total * 100) if total else 0, 1)

    table_rows = []
    for r in results:
        status = r.get("status", "UNKNOWN")
        badge_class = {
            "PASS": "badge-pass",
            "FAIL": "badge-fail",
            "ERROR": "badge-error",
        }.get(status, "badge-unknown")

        scr_html = ""
        scr_path = r.get("screenshot")
        if scr_path and Path(scr_path).exists():
            try:
                b64 = base64.b64encode(Path(scr_path).read_bytes()).decode("utf-8")
                scr_html = f'<a href="data:image/png;base64,{b64}" target="_blank"><img class="thumb" src="data:image/png;base64,{b64}" alt="Screenshot" /></a>'
            except Exception:
                scr_html = f'<a href="{_esc(scr_path)}" target="_blank">View</a>'

        row_html = f"""
        <tr>
          <td><strong>{_esc(r.get('case_id', ''))}</strong></td>
          <td><span class="badge {badge_class}">{_esc(status)}</span></td>
          <td>{_esc(r.get('actual', ''))}</td>
          <td>{_esc(r.get('expected', ''))}</td>
          <td>{float(r.get('duration_s', 0.0)):.2f}s</td>
          <td>{_esc(r.get('note', ''))}</td>
          <td>{scr_html}</td>
        </tr>
        """
        table_rows.append(row_html)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Vettor: Web QA Test Execution Report</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #0f172a;
      color: #e2e8f0;
      margin: 0;
      padding: 24px;
    }}
    .container {{
      max-width: 1200px;
      margin: 0 auto;
    }}
    h1 {{
      margin: 0 0 8px 0;
      font-size: 26px;
      color: #38bdf8;
    }}
    .subtitle {{
      color: #94a3b8;
      font-size: 14px;
      margin-bottom: 24px;
    }}
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .kpi-card {{
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 8px;
      padding: 16px;
      text-align: center;
    }}
    .kpi-val {{
      font-size: 28px;
      font-weight: bold;
      margin-top: 4px;
    }}
    .kpi-val.green {{ color: #4ade80; }}
    .kpi-val.red {{ color: #f87171; }}
    .kpi-val.yellow {{ color: #fbbf24; }}
    .kpi-val.blue {{ color: #38bdf8; }}
    .kpi-label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: #94a3b8;
    }}
    .meta-box {{
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 24px;
      display: flex;
      flex-wrap: wrap;
      gap: 24px;
      font-size: 13px;
    }}
    .meta-item strong {{
      color: #94a3b8;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #1e293b;
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid #334155;
    }}
    th, td {{
      padding: 12px 16px;
      text-align: left;
      border-bottom: 1px solid #334155;
      font-size: 14px;
    }}
    th {{
      background: #0f172a;
      color: #94a3b8;
      text-transform: uppercase;
      font-size: 12px;
      letter-spacing: 0.5px;
    }}
    tr:hover {{
      background: #273549;
    }}
    .badge {{
      display: inline-block;
      padding: 4px 8px;
      border-radius: 4px;
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
    }}
    .badge-pass {{ background: #14532d; color: #4ade80; border: 1px solid #22c55e; }}
    .badge-fail {{ background: #7f1d1d; color: #f87171; border: 1px solid #ef4444; }}
    .badge-error {{ background: #713f12; color: #facc15; border: 1px solid #eab308; }}
    .thumb {{
      height: 38px;
      border-radius: 4px;
      border: 1px solid #475569;
      cursor: pointer;
      transition: transform 0.2s;
    }}
    .thumb:hover {{
      transform: scale(2.5);
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Vettor: Web QA Test Execution Report</h1>
    <div class="subtitle">Generated on {_esc(now)} | Target: {_esc(config.get('target', {}).get('url', 'N/A'))}</div>

    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label">Pass Rate</div>
        <div class="kpi-val {'green' if pass_pct >= 90 else ('yellow' if pass_pct >= 50 else 'red')}">{pass_pct}%</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Passed</div>
        <div class="kpi-val green">{passed} / {total}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Failed</div>
        <div class="kpi-val red">{failed}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Errors</div>
        <div class="kpi-val yellow">{errors}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Self-Heals</div>
        <div class="kpi-val blue">{heals}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-label">Duration</div>
        <div class="kpi-val blue">{duration_s:.1f}s</div>
      </div>
    </div>

    <div class="meta-box">
      <div class="meta-item"><strong>Browser Engine:</strong> {_esc(config.get('browser', {}).get('engine', 'N/A')).upper()}</div>
      <div class="meta-item"><strong>Model Backend:</strong> {_esc(config.get('model', {}).get('backend', 'N/A'))}</div>
      <div class="meta-item"><strong>Model Name:</strong> {_esc(config.get('model', {}).get('model_name', 'N/A'))}</div>
      <div class="meta-item"><strong>Execution Mode:</strong> {_esc(config.get('suite', {}).get('mode', 'data_driven'))}</div>
    </div>

    <table>
      <thead>
        <tr>
          <th>Case ID</th>
          <th>Status</th>
          <th>Actual</th>
          <th>Expected</th>
          <th>Time</th>
          <th>Details / Note</th>
          <th>Screenshot</th>
        </tr>
      </thead>
      <tbody>
        {''.join(table_rows)}
      </tbody>
    </table>
  </div>
</body>
</html>
"""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_content, encoding="utf-8")
    return str(out)
