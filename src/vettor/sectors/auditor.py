from __future__ import annotations

import logging
from typing import Any

from vettor.core.schema import Finding

logger = logging.getLogger("vettor.sectors.auditor")

MULTI_SECTOR_AUDIT_JS = """
() => {
  const findings = [];

  // 1. Accessibility (a11y) checks
  const imgsNoAlt = Array.from(document.querySelectorAll('img')).filter(
    img => !img.hasAttribute('alt') && img.getAttribute('role') !== 'presentation'
  );
  if (imgsNoAlt.length > 0) {
    findings.push({
      sector: 'a11y',
      kind: 'missing_img_alt',
      severity: 'P2',
      detail: `${imgsNoAlt.length} <img> element(s) missing alt attribute`,
      suggested_fix: 'Add descriptive alt text or alt="" for decorative images.',
    });
  }

  const unlabeledInputs = Array.from(document.querySelectorAll('input:not([type="hidden"]), select, textarea')).filter(el => {
    const hasLabel = Boolean(el.labels && el.labels.length > 0);
    const hasAria = Boolean(el.getAttribute('aria-label') || el.getAttribute('aria-labelledby'));
    const hasPlaceholder = Boolean(el.getAttribute('placeholder'));
    return !hasLabel && !hasAria && !hasPlaceholder;
  });
  if (unlabeledInputs.length > 0) {
    findings.push({
      sector: 'a11y',
      kind: 'unlabeled_form_control',
      severity: 'P1',
      detail: `${unlabeledInputs.length} form control(s) have no <label>, aria-label, or placeholder`,
      suggested_fix: 'Associate a <label for="..."> or aria-label with each input.',
    });
  }

  if (!document.documentElement.getAttribute('lang')) {
    findings.push({
      sector: 'a11y',
      kind: 'missing_html_lang',
      severity: 'P3',
      detail: '<html> element is missing a lang attribute',
      suggested_fix: 'Add lang="en" (or appropriate BCP-47 tag) on <html>.',
    });
  }

  // 2. SEO & Metadata checks
  if (!document.title || !document.title.trim()) {
    findings.push({
      sector: 'seo',
      kind: 'missing_title',
      severity: 'P2',
      detail: 'Page is missing a <title> tag',
      suggested_fix: 'Add a descriptive <title> in <head>.',
    });
  }
  const h1Count = document.querySelectorAll('h1').length;
  if (h1Count === 0) {
    findings.push({
      sector: 'seo',
      kind: 'missing_h1',
      severity: 'P3',
      detail: 'Page has no <h1> heading',
      suggested_fix: 'Include one primary <h1> heading per page.',
    });
  }

  // 3. Security Hygiene checks
  const isHttps = location.protocol === 'https:';
  if (isHttps) {
    const insecureForms = Array.from(document.querySelectorAll('form[action^="http://"]'));
    if (insecureForms.length > 0) {
      findings.push({
        sector: 'security',
        kind: 'insecure_form_action',
        severity: 'P0',
        detail: `${insecureForms.length} form(s) submit over plain HTTP from an HTTPS page`,
        suggested_fix: 'Use HTTPS or relative URLs for form actions.',
      });
    }
  }
  const unsafeBlankLinks = Array.from(document.querySelectorAll('a[target="_blank"]')).filter(a => {
    const rel = (a.getAttribute('rel') || '').toLowerCase();
    return !rel.includes('noopener') && !rel.includes('noreferrer');
  });
  if (unsafeBlankLinks.length > 0) {
    findings.push({
      sector: 'security',
      kind: 'tabnabbing_risk',
      severity: 'P3',
      detail: `${unsafeBlankLinks.length} target="_blank" link(s) missing rel="noopener"`,
      suggested_fix: 'Add rel="noopener noreferrer" to external target="_blank" links.',
    });
  }

  // 4. Performance & Web Vitals timing
  let perfMetrics = {};
  try {
    const nav = performance.getEntriesByType('navigation')[0];
    if (nav) {
      perfMetrics = {
        ttfb_ms: Math.round(nav.responseStart - nav.requestStart),
        dom_content_loaded_ms: Math.round(nav.domContentLoadedEventEnd - nav.startTime),
        load_ms: Math.round(nav.loadEventEnd - nav.startTime),
      };
      if (perfMetrics.load_ms > 5000) {
        findings.push({
          sector: 'performance',
          kind: 'slow_page_load',
          severity: 'P2',
          detail: `Page load time ${perfMetrics.load_ms}ms exceeds 5000ms budget`,
          suggested_fix: 'Optimize critical rendering path and bundle size.',
        });
      }
    }
  } catch (e) {}

  return { findings, perfMetrics };
}
"""

ALL_QA_SECTORS = (
    "functional",
    "ui",
    "a11y",
    "network",
    "console",
    "performance",
    "security",
    "seo",
)


async def run_sector_audits(
    page: Any,
    *,
    page_path: str = "",
    sectors: list[str] | None = None,
) -> tuple[list[Finding], dict[str, Any]]:
    """Runs deterministic multi-sector web QA audits (a11y, seo, security, performance)."""
    enabled = set(sectors or ALL_QA_SECTORS)
    findings: list[Finding] = []
    perf_metrics: dict[str, Any] = {}

    try:
        raw = await page.evaluate(MULTI_SECTOR_AUDIT_JS)
        if isinstance(raw, dict):
            perf_metrics = raw.get("perfMetrics") or {}
            for item in raw.get("findings") or []:
                sec = item.get("sector", "ui")
                if sec not in enabled:
                    continue
                findings.append(
                    Finding(
                        kind=item.get("kind", "sector_finding"),
                        sector=sec,
                        severity=item.get("severity", "P2"),
                        page=page_path,
                        detail=item.get("detail", ""),
                        suggested_fix=item.get("suggested_fix", ""),
                        source="sector_probe",
                        signature=f"{sec}:{item.get('kind')}:{page_path}",
                    )
                )
    except Exception as exc:
        logger.debug("Multi-sector audit evaluation skipped/failed: %s", exc)

    return findings, perf_metrics
