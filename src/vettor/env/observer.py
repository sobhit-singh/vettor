from __future__ import annotations

import logging
from typing import Any

from vettor.core.schema import DOMElement, PageObservation

logger = logging.getLogger("vettor.env.observer")

PRUNE_JS = """
() => {
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
    if (!el) return false;
    const style = window.getComputedStyle(el);
    return style.visibility !== 'hidden' && style.display !== 'none' && el.getClientRects().length > 0;
  };

  const elements = [];
  const nodes = document.querySelectorAll('input, select, textarea, button, [role="button"], a[href], [role="tab"]');
  let vid = 1;
  for (const el of nodes) {
    if (!isVisible(el) || el.type === 'hidden' || el.disabled) continue;
    el.setAttribute('data-vettor-id', String(vid));
    const label = (
      el.labels?.[0]?.innerText ||
      el.closest('label')?.innerText ||
      el.getAttribute('aria-label') ||
      el.placeholder ||
      el.innerText ||
      ''
    ).trim().replace(/\\s+/g, ' ').slice(0, 80);

    const tag = el.tagName.toLowerCase();
    const item = {
      vid,
      tag,
      type: el.type || (tag === 'button' ? 'button' : (tag === 'a' ? 'link' : 'text')),
      id: el.id || '',
      name: el.name || '',
      label,
      selector: getScopedSelector(el),
      required: Boolean(el.required || el.getAttribute('aria-required') === 'true'),
      value: el.value || '',
      checked: Boolean(el.checked),
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

  return {
    url: location.href,
    pathname: location.pathname + location.search + location.hash,
    title: document.title || '',
    elements,
    dialogs,
    errors,
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


async def observe_page(page: Any) -> PageObservation:
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
        return PageObservation(
            url=raw.get("url") or getattr(page, "url", ""),
            pathname=raw.get("pathname", ""),
            title=raw.get("title", ""),
            elements=elements,
            errors=raw.get("errors", []),
            dialogs=raw.get("dialogs", []),
            body_text=raw.get("body_text") or raw.get("bodyText", ""),
        )
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
