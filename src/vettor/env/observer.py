from __future__ import annotations

import hashlib
import logging
from typing import Any

from vettor.core.schema import DOMElement, PageObservation

logger = logging.getLogger("vettor.env.observer")

PRUNE_JS = """
() => {
  // B-04 fix: Clear all stale data-vettor-id attributes and reset in-page registry
  document.querySelectorAll('[data-vettor-id]').forEach(el => el.removeAttribute('data-vettor-id'));
  window.__vettorVids = new Map();

  const getScopedSelector = (el) => {
    const testId = el.getAttribute('data-testid');
    if (testId) return `[data-testid="${CSS.escape(testId)}"]`;
    if (el.id) return `#${CSS.escape(el.id)}`;
    if (el.name) {
      if (el.type === 'radio' && el.value) {
        return `input[name="${CSS.escape(el.name)}"][value="${CSS.escape(el.value)}"]`;
      }
      return `${el.tagName.toLowerCase()}[name="${CSS.escape(el.name)}"]`;
    }
    const parts = [];
    let cur = el;
    for (let depth = 0; cur && depth < 4; depth++) {
      if (cur.getAttribute && cur.getAttribute('data-testid')) {
        parts.unshift(`[data-testid="${CSS.escape(cur.getAttribute('data-testid'))}"]`);
        break;
      }
      if (cur.id) {
        parts.unshift(`#${CSS.escape(cur.id)}`);
        break;
      }
      const parent = cur.parentElement;
      if (!parent) {
        parts.unshift(cur.tagName.toLowerCase());
        break;
      }
      const siblings = Array.from(parent.children).filter(c => c.tagName === cur.tagName);
      const idx = siblings.indexOf(cur) + 1;
      parts.unshift(siblings.length > 1 ? `${cur.tagName.toLowerCase()}:nth-of-type(${idx})` : cur.tagName.toLowerCase());
      cur = parent;
    }
    return parts.join(' > ');
  };

  const isVisible = (el) => {
    if (!el || !el.isConnected) return false;
    const style = window.getComputedStyle(el);
    return style.visibility !== 'hidden' && style.display !== 'none' && parseFloat(style.opacity || '1') > 0 && el.getClientRects().length > 0;
  };

  const detectRegion = (el) => {
    if (el.closest('dialog, [role="dialog"], [aria-modal="true"], .modal')) return 'dialog';
    if (el.closest('header, [role="banner"]')) return 'header';
    if (el.closest('nav, [role="navigation"]')) return 'nav';
    if (el.closest('footer, [role="contentinfo"]')) return 'footer';
    if (el.closest('main, [role="main"], form')) return 'main';
    return 'body';
  };

  const collectInteractives = (root, framePath = '') => {
    const found = [];
    const selector = [
      'input', 'select', 'textarea', 'button', 'a[href]',
      '[role="button"]', '[role="tab"]', '[role="checkbox"]',
      '[role="combobox"]', '[role="switch"]', '[role="menuitem"]',
      '[role="radio"]', '[role="slider"]', '[contenteditable="true"]',
      '[tabindex]:not([tabindex="-1"])'
    ].join(', ');

    const walk = (nodeRoot) => {
      if (!nodeRoot || !nodeRoot.querySelectorAll) return;
      for (const el of nodeRoot.querySelectorAll(selector)) {
        if (!isVisible(el) || el.type === 'hidden' || el.disabled || el.getAttribute('aria-disabled') === 'true') continue;
        found.push({ el, framePath });
      }
      // Shadow DOM traversal
      for (const host of nodeRoot.querySelectorAll('*')) {
        if (host.shadowRoot) {
          walk(host.shadowRoot);
        }
      }
    };
    walk(root);

    // Same-origin iframe traversal
    if (!framePath && root === document) {
      const iframes = document.querySelectorAll('iframe');
      iframes.forEach((ifr, idx) => {
        try {
          const doc = ifr.contentDocument || ifr.contentWindow?.document;
          if (doc) {
            found.push(...collectInteractives(doc, `iframe[${idx}]`));
          }
        } catch (e) { /* cross-origin iframe */ }
      });
    }
    return found;
  };

  const rawEntries = collectInteractives(document);

  const priorityOf = (entry) => {
    const region = detectRegion(entry.el);
    if (region === 'dialog') return 0;
    if (region === 'main') return 1;
    const r = entry.el.getBoundingClientRect();
    const inView = r.top >= 0 && r.bottom <= window.innerHeight && r.left >= 0 && r.right <= window.innerWidth;
    if (inView && region !== 'nav' && region !== 'footer') return 2;
    return 3;
  };
  rawEntries.sort((a, b) => priorityOf(a) - priorityOf(b));

  const elements = [];
  let vid = 1;
  for (const { el, framePath } of rawEntries) {
    el.setAttribute('data-vettor-id', String(vid));
    window.__vettorVids.set(vid, el);

    const label = (
      el.labels?.[0]?.innerText ||
      el.closest('label')?.innerText ||
      el.getAttribute('aria-label') ||
      el.placeholder ||
      el.innerText ||
      ''
    ).trim().replace(/\\s+/g, ' ').slice(0, 80);

    const rect = el.getBoundingClientRect();
    const inViewport = rect.width > 0 && rect.height > 0 && rect.top >= 0 && rect.bottom <= window.innerHeight && rect.left >= 0 && rect.right <= window.innerWidth;
    let covered = false;
    if (inViewport && !framePath) {
      const cx = Math.min(window.innerWidth - 1, Math.max(0, rect.left + rect.width / 2));
      const cy = Math.min(window.innerHeight - 1, Math.max(0, rect.top + rect.height / 2));
      const topEl = document.elementFromPoint(cx, cy);
      if (topEl && topEl !== el && !el.contains(topEl) && !topEl.contains(el)) {
        covered = true;
      }
    }

    const tag = el.tagName.toLowerCase();
    const item = {
      vid,
      tag,
      type: el.type || (tag === 'button' ? 'button' : (tag === 'a' ? 'link' : 'text')),
      role: el.getAttribute('role') || '',
      id: el.id || '',
      name: el.name || '',
      label,
      placeholder: el.placeholder || '',
      selector: getScopedSelector(el),
      required: Boolean(el.required || el.getAttribute('aria-required') === 'true'),
      disabled: Boolean(el.disabled),
      value: el.value !== undefined ? String(el.value || '') : (el.isContentEditable ? (el.innerText || '').slice(0, 120) : ''),
      checked: Boolean(el.checked || el.getAttribute('aria-checked') === 'true'),
      in_viewport: inViewport,
      covered,
      region: detectRegion(el),
      frame_path: framePath,
      options: [],
    };
    if (tag === 'select') {
      item.options = Array.from(el.options).map(o => o.value || o.text.trim()).slice(0, 15);
    }
    elements.push(item);
    vid++;
  }

  const dialogs = [];
  for (const d of document.querySelectorAll('dialog, [role="dialog"], .modal, .cookie-banner, [aria-modal="true"]')) {
    if (isVisible(d)) {
      const dismissBtn = d.querySelector('button, [role="button"], .close, .dismiss, #accept, #agree');
      dialogs.push({
        selector: getScopedSelector(d),
        text: (d.innerText || '').slice(0, 120),
        dismiss_selector: dismissBtn ? getScopedSelector(dismissBtn) : null,
      });
    }
  }

  const errors = [];
  for (const e of document.querySelectorAll('[role="alert"], .error, .alert, .invalid-feedback, .field-error, :invalid')) {
    const txt = (e.innerText || e.validationMessage || '').trim();
    if (txt && !errors.includes(txt)) errors.push(txt);
  }

  const headings = Array.from(document.querySelectorAll('h1, h2, h3'))
    .filter(h => isVisible(h))
    .map(h => (h.innerText || '').trim().replace(/\\s+/g, ' ').slice(0, 100))
    .filter(Boolean)
    .slice(0, 10);

  const mainEl = document.querySelector('main, [role="main"], form');
  const regions = {
    main: mainEl ? (mainEl.innerText || '').replace(/\\s+/g, ' ').slice(0, 3000) : '',
    alert: errors.join(' | ').slice(0, 1000),
    dialog: dialogs.map(d => d.text).join(' | ').slice(0, 1000),
  };

  return {
    url: location.href,
    pathname: location.pathname + location.search + location.hash,
    title: document.title || '',
    elements,
    dialogs,
    errors,
    visible_headings: headings,
    regions,
    body_text: (document.body?.innerText || '').replace(/\\s+/g, ' ').slice(0, 6000),
  };
}
"""

SOM_INJECT_JS = """
() => {
  document.querySelectorAll('.vettor-som-box').forEach(e => e.remove());
  for (const el of document.querySelectorAll('[data-vettor-id]')) {
    const rect = el.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) continue;
    const vid = el.getAttribute('data-vettor-id');
    const box = document.createElement('div');
    box.className = 'vettor-som-box';
    box.style.cssText = `position:fixed;left:${rect.left}px;top:${rect.top}px;width:${rect.width}px;height:${rect.height}px;border:2px solid #ef4444;pointer-events:none;z-index:2147483647;box-sizing:border-box;`;
    const tag = document.createElement('span');
    tag.textContent = vid;
    tag.style.cssText = 'position:absolute;top:-14px;left:-2px;background:#ef4444;color:#fff;font:bold 11px monospace;padding:0 3px;border-radius:2px;';
    box.appendChild(tag);
    document.body.appendChild(box);
  }
}
"""

SOM_CLEAR_JS = "() => document.querySelectorAll('.vettor-som-box').forEach(e => e.remove())"


def compute_dom_hash(obs: PageObservation) -> str:
    """Computes a stable hash of the page URL, elements, errors, and main text."""
    sig_parts = [
        obs.pathname or obs.url,
        "|".join(f"{e.selector}:{e.value}:{e.checked}" for e in obs.elements[:50]),
        "|".join(obs.errors[:10]),
        obs.body_text[:400],
    ]
    return hashlib.sha256("\n".join(sig_parts).encode("utf-8", errors="ignore")).hexdigest()[:16]


def compute_dom_diff(prev: PageObservation | None, curr: PageObservation) -> list[str]:
    """Computes a concise human-readable diff between two observations."""
    if prev is None:
        return []
    diffs: list[str] = []
    if prev.url != curr.url:
        diffs.append(f"URL changed: {prev.pathname or prev.url} -> {curr.pathname or curr.url}")
    prev_map = {e.selector: e for e in prev.elements if e.selector}
    curr_map = {e.selector: e for e in curr.elements if e.selector}
    added = [sel for sel in curr_map if sel not in prev_map]
    removed = [sel for sel in prev_map if sel not in curr_map]
    if added:
        diffs.append(f"+{len(added)} elements appeared")
    if removed:
        diffs.append(f"-{len(removed)} elements disappeared")
    for sel, el in curr_map.items():
        if sel in prev_map and prev_map[sel].value != el.value:
            diffs.append(f"Changed {el.label or sel}: '{prev_map[sel].value}' -> '{el.value}'")
    if curr.errors != prev.errors:
        diffs.append(f"Errors updated: {curr.errors or ['none']}")
    return diffs[:8]


async def observe_page(page: Any, prev_obs: PageObservation | None = None) -> PageObservation:
    """Evaluates PRUNE_JS in the browser page and parses it into a PageObservation model."""
    try:
        raw = await page.evaluate(PRUNE_JS)
        if not isinstance(raw, dict):
            raw = {}
        elements_data = raw.get("elements") or raw.get("fields") or []
        elements = [
            DOMElement.model_validate(el) if isinstance(el, dict) else el
            for el in elements_data
        ]
        obs = PageObservation(
            url=raw.get("url") or getattr(page, "url", ""),
            pathname=raw.get("pathname", ""),
            title=raw.get("title", ""),
            elements=elements,
            errors=raw.get("errors", []),
            dialogs=raw.get("dialogs", []),
            visible_headings=raw.get("visible_headings", []),
            regions=raw.get("regions", {}),
            body_text=raw.get("body_text") or raw.get("bodyText", ""),
        )
        obs.dom_hash = compute_dom_hash(obs)
        obs.dom_diff = compute_dom_diff(prev_obs, obs)
        return obs
    except Exception as exc:
        logger.warning("DOM observe_page evaluation failed: %s", exc)
        return PageObservation(
            url=getattr(page, "url", ""),
            pathname="",
            title="",
            elements=[],
            errors=[f"Observe error: {exc}"],
            dialogs=[],
            body_text="",
        )


async def capture_som_screenshot(page: Any) -> bytes:
    """Injects Set-of-Mark visual bounding boxes, captures screenshot, and clears overlay."""
    try:
        await page.evaluate(SOM_INJECT_JS)
        return await page.screenshot(type="png")
    finally:
        try:
            await page.evaluate(SOM_CLEAR_JS)
        except Exception:
            pass
