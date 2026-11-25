from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from vettor.config import VettorConfig
from vettor.core.schema import AgentAction, PageObservation
from vettor.env.observer import observe_page

logger = logging.getLogger("vettor.env.browser")


class PageSession:
    """Wraps an active browser context and page with live diagnostics listeners."""

    def __init__(self, context: Any, page: Any):
        self.context = context
        self.page = page
        self.console_logs: list[str] = []
        self.page_errors: list[str] = []
        self.network_errors: list[dict[str, Any]] = []
        self.network_requests: list[dict[str, Any]] = []
        self._last_obs: PageObservation | None = None

    async def goto(
        self,
        url: str,
        url_params: dict[str, str] | None = None,
        timeout_ms: int = 12000,
    ) -> None:
        """Navigates to URL, cleanly appending query parameters if provided."""
        final_url = url
        if url_params:
            parsed = urlparse(url)
            existing_params = dict(parse_qsl(parsed.query))
            existing_params.update(url_params)
            new_query = urlencode(existing_params)
            final_url = urlunparse((
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                parsed.params,
                new_query,
                parsed.fragment,
            ))

        await self.page.goto(final_url, timeout=timeout_ms, wait_until="load")

    async def observe(self) -> PageObservation:
        """Extracts the interactive DOM structure and page observation."""
        obs = await observe_page(self.page, prev_obs=self._last_obs)
        self._last_obs = obs
        return obs

    async def _settle_after_action(self, wait_ms: int = 600) -> None:
        """Briefly waits for SPA DOM/network settle without blocking on long-polling."""
        try:
            await self.page.wait_for_load_state("networkidle", timeout=wait_ms)
        except Exception:
            await asyncio.sleep(0.15)

    async def execute_action(
        self,
        action: AgentAction,
        selector_override: str | None = None,
        timeout_ms: int = 4000,
    ) -> str:
        """Executes an action against the page using the integer vid or selector override."""
        if action.action == "wait":
            sleep_ms = min(int(action.value or 500), 5000)
            await asyncio.sleep(sleep_ms / 1000.0)
            return "wait"

        if action.action == "scroll":
            direction = (action.value or "down").strip().lower()
            delta = -500 if direction == "up" else 500
            await self.page.evaluate(f"window.scrollBy(0, {delta})")
            await asyncio.sleep(0.15)
            return f"scroll:{direction}"

        if action.action == "goto":
            if not action.value:
                raise ValueError("Action 'goto' requires a URL in 'value'")
            await self.goto(action.value)
            return f"goto:{action.value}"

        # Resolve selector
        if selector_override:
            target_selector = selector_override
        elif action.vid is not None:
            target_selector = f'[data-vettor-id="{action.vid}"]'
        else:
            raise ValueError(f"Cannot execute action {action.action} without vid or selector_override")

        # Inspect DOM element tag and type to ensure appropriate control interaction
        tag_info = await self.page.evaluate(
            """(sel) => {
                const el = document.querySelector(sel);
                if (!el) return null;
                return {
                    tag: el.tagName.toLowerCase(),
                    type: (el.type || '').toLowerCase(),
                    checked: Boolean(el.checked),
                };
            }""",
            target_selector,
        )

        locator = self.page.locator(target_selector)
        tag = (tag_info.get("tag") if tag_info else "").lower()
        input_type = (tag_info.get("type") if tag_info else "").lower()

        # Checkbox / Radio handling
        if tag == "input" and input_type in ("checkbox", "radio"):
            val_lower = (action.value or "").strip().lower()
            if action.action == "uncheck" or val_lower in ("no", "false", "0", "uncheck", "off"):
                await locator.uncheck(timeout=timeout_ms)
            elif action.action == "click":
                await locator.click(timeout=timeout_ms)
            else:
                await locator.check(timeout=timeout_ms)
            return target_selector

        # Select dropdown handling
        if tag == "select":
            if action.action in ("select", "fill"):
                opt_val = action.value
                try:
                    await locator.select_option(value=opt_val, timeout=timeout_ms)
                except Exception:
                    try:
                        await locator.select_option(label=opt_val, timeout=timeout_ms)
                    except Exception:
                        await locator.select_option(index=0, timeout=timeout_ms)
            elif action.action == "click":
                await locator.click(timeout=timeout_ms)
            return target_selector

        # Standard input or textarea
        if tag in ("input", "textarea"):
            if action.action in ("fill", "type"):
                await locator.fill(action.value, timeout=timeout_ms)
            elif action.action == "press":
                await locator.press(action.value or "Enter", timeout=timeout_ms)
                await self._settle_after_action()
            elif action.action == "click":
                await locator.click(timeout=timeout_ms)
                await self._settle_after_action()
            else:
                await locator.fill(action.value, timeout=timeout_ms)
            return target_selector

        # Button, link, or general interactive element
        if action.action in ("click", "fill"):
            await locator.click(timeout=timeout_ms)
            await self._settle_after_action()
        elif action.action == "press":
            await locator.press(action.value or "Enter", timeout=timeout_ms)
            await self._settle_after_action()
        elif action.action in ("check", "select"):
            await locator.click(timeout=timeout_ms)
            await self._settle_after_action()

        return target_selector

    async def close(self) -> None:
        """Safely closes the underlying context."""
        try:
            await self.context.close()
        except Exception as exc:
            logger.debug("Error closing PageSession context: %s", exc)


class BrowserPool:
    """Async browser pool manager creating PageSessions with pre-seeded storage and diagnostics."""

    def __init__(self, cfg: VettorConfig):
        self.cfg = cfg
        self._pw: Any = None
        self._browser: Any = None

    async def start(self) -> None:
        """Initializes Playwright or Rustwright without throwaway test browser runs."""
        engine = self.cfg.browser.engine
        async_playwright_fn = None

        if engine == "rustwright":
            try:
                from rustwright.async_api import async_playwright as rw_playwright
                async_playwright_fn = rw_playwright
            except ImportError:
                logger.info("rustwright not found; falling back to standard playwright")

        if async_playwright_fn is None:
            from playwright.async_api import async_playwright as pw_playwright
            async_playwright_fn = pw_playwright

        self._pw = await async_playwright_fn().start()
        b_type = getattr(self._pw, self.cfg.browser.browser_type, self._pw.chromium)
        self._browser = await b_type.launch(
            headless=self.cfg.browser.headless,
            slow_mo=self.cfg.browser.slow_mo_ms if self.cfg.browser.slow_mo_ms > 0 else None,
        )

    async def new_session(self, storage_state: str | None = None) -> PageSession:
        """Spawns an isolated PageSession with target headers, storage state, and init scripts."""
        if self._browser is None:
            await self.start()

        context_kwargs: dict[str, Any] = {
            "viewport": {
                "width": self.cfg.browser.viewport_width,
                "height": self.cfg.browser.viewport_height,
            },
        }

        if self.cfg.target.auth_headers:
            context_kwargs["extra_http_headers"] = self.cfg.target.auth_headers

        effective_storage = storage_state or self.cfg.target.storage_state
        if effective_storage and Path(effective_storage).exists():
            context_kwargs["storage_state"] = effective_storage

        context = await self._browser.new_context(**context_kwargs)

        if self.cfg.target.cookies:
            await context.add_cookies(self.cfg.target.cookies)

        if self.cfg.target.local_storage or self.cfg.target.session_storage:
            ls_data = json.dumps(self.cfg.target.local_storage)
            ss_data = json.dumps(self.cfg.target.session_storage)
            init_js = f"""
            (() => {{
                try {{
                    const ls = {ls_data};
                    for (const [k, v] of Object.entries(ls)) {{
                        window.localStorage.setItem(k, v);
                    }}
                }} catch (e) {{}}
                try {{
                    const ss = {ss_data};
                    for (const [k, v] of Object.entries(ss)) {{
                        window.sessionStorage.setItem(k, v);
                    }}
                }} catch (e) {{}}
            }})();
            """
            await context.add_init_script(init_js)

        page = await context.new_page()
        session = PageSession(context, page)

        page.on("console", lambda msg: session.console_logs.append(f"[{msg.type}] {msg.text}"))
        page.on("pageerror", lambda err: session.page_errors.append(str(err)))

        def on_response(resp: Any) -> None:
            try:
                status = getattr(resp, "status", 200)
                if status >= 400:
                    session.network_errors.append({
                        "url": getattr(resp, "url", ""),
                        "status": status,
                    })
            except Exception:
                pass

        page.on("response", on_response)

        return session

    async def close(self) -> None:
        """Closes browser and stops Playwright."""
        if self._browser:
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None
        if self._pw:
            try:
                await self._pw.stop()
            except Exception:
                pass
            self._pw = None

    async def __aenter__(self) -> "BrowserPool":
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
