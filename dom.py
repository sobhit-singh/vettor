PRUNE_JS = """
() => {
  const getSelector = (el) => {
    if (el.id) return `#${CSS.escape(el.id)}`;
    if (el.getAttribute('data-testid')) return `[data-testid="${el.getAttribute('data-testid')}"]`;
    if (el.name) {
      if (el.type === 'radio' && el.value) {
        return `input[name="${CSS.escape(el.name)}"][value="${CSS.escape(el.value)}"]`;
      }
      return `${el.tagName.toLowerCase()}[name="${CSS.escape(el.name)}"]`;
    }
    const parent = el.parentElement;
    if (!parent) return el.tagName.toLowerCase();
    const siblings = Array.from(parent.children).filter(c => c.tagName === el.tagName);
    const idx = siblings.indexOf(el) + 1;
    return `${el.tagName.toLowerCase()}:nth-of-type(${idx})`;
  };

  const isVisible = (el) => {
    const style = window.getComputedStyle(el);
    return style.visibility !== 'hidden' && style.display !== 'none' && el.getClientRects().length > 0;
  };

  const fields = [];
  const nodes = document.querySelectorAll('input, select, textarea, button, [role="button"], a.btn, [role="tab"]');
  for (const el of nodes) {
    if (!isVisible(el) || el.type === 'hidden' || el.disabled) continue;
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
      tag,
      type: el.type || (tag === 'button' ? 'button' : (tag === 'a' ? 'link' : 'text')),
      id: el.id || '',
      name: el.name || '',
      label,
      selector: getSelector(el),
      required: Boolean(el.required || el.getAttribute('aria-required') === 'true'),
      value: el.value || '',
      checked: Boolean(el.checked),
    };

    if (tag === 'select') {
      item.options = Array.from(el.options).map(o => o.value || o.text.trim()).slice(0, 15);
    }
    fields.push(item);
  }

  // Detect visible dialogs, popups, cookie consent banners
  const dialogNodes = document.querySelectorAll('dialog, [role="dialog"], .modal, .cookie-banner, [aria-modal="true"]');
  const dialogs = [];
  for (const d of dialogNodes) {
    if (isVisible(d)) {
      const dismissBtn = d.querySelector('button, [role="button"], .close, .dismiss, #accept, #agree');
      dialogs.push({
        selector: getSelector(d),
        text: (d.innerText || '').slice(0, 120),
        dismiss_selector: dismissBtn ? getSelector(dismissBtn) : null,
      });
    }
  }

  // Detect server alerts & explicit error messages
  const errorElements = Array.from(
    document.querySelectorAll('[role="alert"], .error, .alert, .invalid-feedback, .field-error, :invalid')
  );
  const errors = [];
  for (const e of errorElements) {
    const txt = (e.innerText || e.validationMessage || '').trim();
    if (txt && !errors.includes(txt)) {
      errors.push(txt);
    }
  }

  return {
    url: location.href,
    pathname: location.pathname + location.hash,
    title: document.title || '',
    fields,
    dialogs,
    errors,
    bodyText: (document.body?.innerText || '').replace(/\\s+/g, ' ').slice(0, 8000),
  };
}
"""


def snapshot_page(page) -> dict:
    """Extracts a structured, token-efficient DOM snapshot for the QA agent."""
    try:
        return page.evaluate(PRUNE_JS)
    except Exception as exc:
        return {
            "url": getattr(page, "url", ""),
            "pathname": "",
            "title": "",
            "fields": [],
            "dialogs": [],
            "errors": [f"Snapshot error: {exc}"],
            "bodyText": "",
        }
