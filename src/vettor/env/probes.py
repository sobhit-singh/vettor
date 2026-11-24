from __future__ import annotations

import logging
from typing import Any

from vettor.core.schema import Finding

logger = logging.getLogger("vettor.env.probes")

PROBE_JS = """
() => {
  const anomalies = [];
  if (document.documentElement.scrollWidth > window.innerWidth + 2) {
    anomalies.push({
      kind: "layout_overflow",
      severity: "P2",
      detail: `Horizontal overflow: scrollWidth=${document.documentElement.scrollWidth} > innerWidth=${window.innerWidth}`,
    });
  }
  for (const img of document.querySelectorAll('img')) {
    const style = window.getComputedStyle(img);
    if (style.display !== 'none' && img.complete && img.naturalWidth === 0) {
      anomalies.push({
        kind: "broken_image",
        severity: "P2",
        detail: `Broken image: ${img.src || img.alt || 'unnamed'}`,
      });
    }
  }
  const interactives = Array.from(document.querySelectorAll('button, a[href], input, select'))
    .filter(el => el.getClientRects().length > 0 && window.getComputedStyle(el).visibility !== 'hidden')
    .slice(0, 60);

  for (let i = 0; i < interactives.length; i++) {
    const r1 = interactives[i].getBoundingClientRect();
    if (r1.width === 0 || r1.height === 0) continue;
    for (let j = i + 1; j < interactives.length; j++) {
      if (interactives[i].contains(interactives[j]) || interactives[j].contains(interactives[i])) continue;
      const r2 = interactives[j].getBoundingClientRect();
      if (r2.width === 0 || r2.height === 0) continue;
      const overlapX = Math.max(0, Math.min(r1.right, r2.right) - Math.max(r1.left, r2.left));
      const overlapY = Math.max(0, Math.min(r1.bottom, r2.bottom) - Math.max(r1.top, r2.top));
      const area = overlapX * overlapY;
      const minArea = Math.min(r1.width * r1.height, r2.width * r2.height);
      if (minArea > 0 && (area / minArea) > 0.25) {
        anomalies.push({
          kind: "element_overlap",
          severity: "P2",
          detail: `Interactive overlap (${Math.round(area / minArea * 100)}%): ${interactives[i].tagName}#${interactives[i].id} & ${interactives[j].tagName}#${interactives[j].id}`,
        });
      }
    }
  }
  return anomalies;
}
"""


async def run_ui_probes(
    page: Any,
    *,
    page_path: str = "",
    network_errors: list[dict[str, Any]] | None = None,
    page_errors: list[str] | None = None,
) -> list[Finding]:
    """Evaluates deterministic client-side probes and aggregates runtime network/JS errors into Findings."""
    findings: list[Finding] = []

    try:
        raw_anomalies = await page.evaluate(PROBE_JS) or []
        for anomaly in raw_anomalies:
            findings.append(
                Finding(
                    kind=anomaly.get("kind", "layout_anomaly"),
                    severity=anomaly.get("severity", "P2"),
                    page=page_path,
                    detail=anomaly.get("detail", ""),
                    element_vid=anomaly.get("element_vid"),
                    suggested_fix=anomaly.get("suggested_fix", ""),
                    source="probe",
                )
            )
    except Exception as exc:
        logger.debug("Client probe execution failed: %s", exc)

    for net in (network_errors or []):
        status = int(net.get("status", 500))
        if status >= 400:
            severity = "P1" if status >= 500 else "P2"
            findings.append(
                Finding(
                    kind="network_error",
                    severity=severity,
                    page=page_path,
                    detail=f"HTTP {status} on {net.get('url', '')}",
                    source="probe",
                )
            )

    for err in (page_errors or []):
        findings.append(
            Finding(
                kind="js_exception",
                severity="P1",
                page=page_path,
                detail=f"Uncaught JS error: {str(err)}",
                source="probe",
            )
        )

    return findings
