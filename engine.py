import csv
import json
import logging
import os
from pathlib import Path
import re
import time
from urllib.parse import urlencode, urlparse, urlunparse, parse_qs
from dataclasses import dataclass, field
from typing import Callable, Any

from agent import QAAgent
from assertions import QAAssertions, AssertionResult
from cache import RecipeCache
from config import QAConfig
from dom import snapshot_page
from memory import QAMemory
from handoff import HandoffManager, stream_test_result_jsonl
import reporting

logger = logging.getLogger("vettor.engine")


@dataclass
class RunStats:
    total: int = 0
    completed: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    cache_hits: int = 0
    heals: int = 0
    duration_s: float = 0.0
    results: list[dict] = field(default_factory=list)


def _find_chromium_executable() -> str | None:
    env_chrom = os.getenv("RUSTWRIGHT_CHROMIUM")
    if env_chrom and Path(env_chrom).exists():
        return env_chrom

    local_app_data = os.getenv("LOCALAPPDATA", "")
    if local_app_data:
        pw_dir = Path(local_app_data) / "ms-playwright"
        if pw_dir.exists():
            for chrome_path in sorted(pw_dir.glob("chromium-*/chrome-win64/chrome.exe"), reverse=True):
                if chrome_path.exists():
                    return str(chrome_path)
            for chrome_path in sorted(
                pw_dir.glob("chromium_headless_shell-*/chrome-headless-shell-win64/chrome-headless-shell.exe"),
                reverse=True,
            ):
                if chrome_path.exists():
                    return str(chrome_path)

    for candidate in [
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]:
        if Path(candidate).exists():
            return candidate

    return None


def _get_playwright(engine_name: str, browser_type: str = "chromium"):
    if engine_name == "rustwright" and browser_type != "chromium":
        logger.warning("Rustwright supports only chromium; ignoring browser_type=%s", browser_type)
        browser_type = "chromium"
    if engine_name == "rustwright":
        try:
            chrom_path = _find_chromium_executable()
            if chrom_path and "RUSTWRIGHT_CHROMIUM" not in os.environ:
                os.environ["RUSTWRIGHT_CHROMIUM"] = chrom_path
            from rustwright.sync_api import sync_playwright

            with sync_playwright() as p:
                b = p.chromium.launch(executable_path=chrom_path, headless=True)
                b.close()
            return sync_playwright, "rustwright", chrom_path
        except Exception as exc:
            logger.warning("Rustwright launch failed (%s). Falling back to Playwright.", exc)
            from playwright.sync_api import sync_playwright

            return sync_playwright, "playwright (fallback)", None

    from playwright.sync_api import sync_playwright

    return sync_playwright, "playwright", None


def _page_signature(snap: dict) -> tuple:
    return (
        snap.get("pathname", ""),
        tuple((f.get("selector"), f.get("tag")) for f in snap.get("fields", [])),
    )


class QAEngine:
    """Configurable agentic execution harness for Web QA."""

    def __init__(self, cfg: QAConfig, on_event: Callable[[dict], None] | None = None):
        self.cfg = cfg
        self.on_event = on_event or (lambda _: None)
        self.cache = RecipeCache(cfg.recipe_path)
        self.agent = QAAgent(cfg)
        self.memory = QAMemory(cfg.reporting.memory_path)
        self.handoff = HandoffManager(cfg.reporting.handoff_path)
        self.stop_requested = False
        self.stats: RunStats | None = None

        self.fail_dir = Path(cfg.failures_dir)
        self.fail_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = Path(cfg.reporting.output_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _load_test_cases(self) -> list[dict[str, Any]]:
        """Loads test cases from .jsonl, .json, or .csv files with standard key normalization."""
        resolved = self.cfg.get_resolved_cases_path()
        if not resolved.exists():
            return []

        ext = resolved.suffix.lower()
        rows: list[dict[str, Any]] = []

        if ext == ".jsonl":
            with open(resolved, encoding="utf-8") as f:
                for idx, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    if "case_id" not in item and "id" in item:
                        item["case_id"] = str(item["id"])
                    elif "case_id" not in item:
                        item["case_id"] = str(idx)
                    rows.append(item)
        elif ext == ".json":
            with open(resolved, encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for idx, item in enumerate(data, start=1):
                        if "case_id" not in item and "id" in item:
                            item["case_id"] = str(item["id"])
                        elif "case_id" not in item:
                            item["case_id"] = str(idx)
                        rows.append(item)
        else:
            with open(resolved, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for idx, row in enumerate(reader, start=1):
                    item = dict(row)
                    if "case_id" not in item and "id" in item:
                        item["case_id"] = str(item["id"])
                    elif "case_id" not in item:
                        item["case_id"] = str(idx)
                    rows.append(item)

        return rows

    def emit(self, event_type: str, **data):
        self.on_event({"type": event_type, **data})

    def _apply_action(self, page, selector: str, value: str, action: str):
        t = self.cfg.step_timeout_ms
        if not selector:
            return

        try:
            if not page.is_visible(selector):
                return
        except Exception:
            return

        try:
            elem_info = page.evaluate(
                """(sel) => {
                    const el = document.querySelector(sel);
                    if (!el) return null;
                    return { tag: el.tagName.toLowerCase(), type: el.type || '' };
                }""",
                selector,
            )
        except Exception:
            elem_info = None

        if elem_info:
            tag = elem_info["tag"]
            el_type = elem_info["type"]
            if tag in ("button", "a") or action == "click":
                page.click(selector, timeout=t)
                return
            if el_type == "checkbox":
                if str(value).strip().lower() in ("true", "yes", "1", "on"):
                    page.check(selector, timeout=t)
                else:
                    page.uncheck(selector, timeout=t)
                return
            if tag == "select":
                page.select_option(selector, str(value), timeout=t)
                return
            if tag in ("input", "textarea"):
                page.fill(selector, str(value), timeout=t)
                return

        # Fallback if DOM evaluation was unresolvable
        if action == "click":
            page.click(selector, timeout=t)
        elif action == "check":
            if str(value).strip().lower() in ("true", "yes", "1", "on"):
                page.check(selector, timeout=t)
            else:
                page.uncheck(selector, timeout=t)
        elif action == "select":
            page.select_option(selector, str(value), timeout=t)
        else:
            page.fill(selector, str(value), timeout=t)

    def _run_row(self, browser, row: dict, stats: RunStats) -> dict:
        case_id = row.get("case_id", "unknown")
        context_kwargs: dict[str, Any] = {
            "viewport": {
                "width": self.cfg.browser.viewport_width,
                "height": self.cfg.browser.viewport_height,
            }
        }
        if self.cfg.target.storage_state:
            state_path = Path(self.cfg.target.storage_state)
            if state_path.exists():
                context_kwargs["storage_state"] = self.cfg.target.storage_state
            else:
                logger.warning("Configured storage_state file not found: %s", state_path)
                self.emit("log", msg=f"[yellow]Warning:[/] storage_state file not found: {state_path}")
        if self.cfg.target.auth_headers:
            context_kwargs["extra_http_headers"] = self.cfg.target.auth_headers

        ctx = browser.new_context(**context_kwargs)
        if self.cfg.target.cookies:
            ctx.add_cookies(self.cfg.target.cookies)

        page = ctx.new_page()
        console_logs = []
        page_errors = []
        page.on("console", lambda msg: console_logs.append(f"[{msg.type}] {msg.text}"))
        page.on("pageerror", lambda err: page_errors.append(str(err)))

        # Testability Seams: LocalStorage & SessionStorage pre-seeding
        if self.cfg.target.local_storage or self.cfg.target.session_storage:
            seed_script = ""
            if self.cfg.target.local_storage:
                for k, v in self.cfg.target.local_storage.items():
                    seed_script += f"window.localStorage.setItem({json.dumps(k)}, {json.dumps(str(v))});\n"
            if self.cfg.target.session_storage:
                for k, v in self.cfg.target.session_storage.items():
                    seed_script += f"window.sessionStorage.setItem({json.dumps(k)}, {json.dumps(str(v))});\n"
            try:
                page.add_init_script(seed_script)
            except Exception:
                pass

        # Testability Seams: URL parameters injection
        target_nav_url = self.cfg.start_url
        url_params = dict(self.cfg.target.url_params)
        if "test_key" in row and "test_value" in row:
            url_params["test_key"] = str(row["test_key"])
            url_params["test_value"] = str(row["test_value"])
        if url_params:
            sep = "&" if "?" in target_nav_url else "?"
            target_nav_url = f"{target_nav_url}{sep}{urlencode(url_params)}"

        t0 = time.perf_counter()
        screenshot_path: str | None = None
        applied_memories: list[str] = []

        try:
            page.goto(target_nav_url, wait_until="load", timeout=12000)

            for step in range(self.cfg.max_pages):
                page.wait_for_timeout(200)
                parsed_u = urlparse(page.url)
                page_key = parsed_u.path if parsed_u.path else "/"

                # Consult persistent memory for known cautions or quirks on this page
                mem_records = self.memory.lookup(page.url, min_trust=0.6)
                for rec in mem_records:
                    if rec.kind == "caution":
                        workaround = rec.data
                        fix_sel = workaround.get("fix_selector")
                        if fix_sel:
                            try:
                                if page.is_visible(fix_sel):
                                    self.emit("log", msg=f"[Case {case_id}] Memory auto-recovery [{rec.id}]: {rec.summary}")
                                    self._apply_action(page, fix_sel, workaround.get("fix_value", "yes"), workaround.get("action", "check"))
                                    applied_memories.append(rec.id)
                            except Exception:
                                pass

                snap = snapshot_page(page)
                stage_info = self.cfg.get_stage_guidance(page.url, snap.get("title", ""))
                plan = self.cache.get(page_key)

                # Check if terminal or summary page dynamically
                is_summary = (
                    self.cfg.is_terminal_page(
                        page.url, snap.get("bodyText", ""), plan, page_title=snap.get("title", "")
                    )
                    or (stage_info.get("is_terminal", False))
                )

                if is_summary:
                    target_keys = [k.replace("expected_", "") for k in row.keys() if k.startswith("expected_")]
                    extracted = self.agent.call_tool(
                        "extract_final_result",
                        {
                            "bodyText": snap["bodyText"],
                            "target_keys": target_keys,
                            "app_context": self.cfg.flow.app_context,
                            "stage_guidance": stage_info.get("guidance", ""),
                        },
                    )
                    # Evaluate assertions
                    assertion_results = QAAssertions.evaluate_case_assertions(
                        row=row,
                        page_state=snap,
                        extracted=extracted,
                        float_tolerance=self.cfg.float_tolerance,
                    )

                    all_passed = all(a.passed for a in assertion_results)
                    status = "PASS" if all_passed else "FAIL"
                    dur = round(time.perf_counter() - t0, 2)

                    # Determine screenshot capture
                    if self.cfg.reporting.screenshot_mode == "always" or (
                        self.cfg.reporting.screenshot_mode == "on_failure" and not all_passed
                    ):
                        screenshot_path = str(self.out_dir / f"case_{case_id}_{status.lower()}.png")
                        try:
                            page.screenshot(path=screenshot_path)
                        except Exception:
                            pass

                    first_mismatch = next((a.message for a in assertion_results if not a.passed), "")
                    note = first_mismatch or extracted.get("quote_id", extracted.get("order_id", "Completed"))

                    exp_keys = [k for k in row.keys() if k.startswith("expected_")]
                    if exp_keys:
                        primary_col = exp_keys[0]
                        expected_raw = row[primary_col]
                        clean_col = primary_col.replace("expected_", "")
                        primary_actual = extracted.get(clean_col, next(iter(extracted.values()), "OK"))
                    else:
                        expected_raw = row.get("expected", "")
                        primary_actual = next(iter(extracted.values()), "OK") if extracted else "OK"

                    clean_exp = re.sub(r"[^\d.]", "", str(expected_raw)) if expected_raw else ""
                    exp_val = float(clean_exp) if clean_exp and clean_exp.replace(".", "", 1).isdigit() else expected_raw

                    if all_passed:
                        for mid in applied_memories:
                            self.memory.confirm(mid)
                        if plan:
                            self.memory.record_fact(
                                target_url=page.url,
                                summary=f"Verified form plan for {page_key}",
                                data=plan,
                            )
                    else:
                        for mid in applied_memories:
                            self.memory.contradict(mid)
                        self.handoff.record_issue(
                            title=f"Assertion Failure in Case {case_id}",
                            issue_type="Bug",
                            severity="P1",
                            observed=f"Assertion failed: {note}",
                            expected=f"Expected: {exp_val}, Actual: {primary_actual}",
                            evidence=f"Case {case_id}; screenshot: {screenshot_path or 'none'}",
                            suggested_fix="Verify business calculation rules or UI text format",
                            related_tests=[str(case_id)],
                        )

                    return {
                        "case_id": case_id,
                        "status": status,
                        "actual": primary_actual,
                        "expected": exp_val,
                        "duration_s": dur,
                        "note": note,
                        "screenshot": screenshot_path,
                        "assertions": [a.to_dict() for a in assertion_results],
                    }

                # Schema mapping on cache miss
                if plan is None:
                    self.emit("log", msg=f"[Case {case_id}] Cache miss on {page_key} -> mapping via {self.cfg.local_model}")
                    plan = self.agent.call_tool(
                        "map_page_schema",
                        {
                            "csv_columns": list(row.keys()),
                            "fields": snap["fields"],
                            "page_url": page.url,
                            "page_title": snap.get("title", ""),
                            "app_context": self.cfg.flow.app_context,
                            "stage_name": stage_info.get("name", ""),
                            "stage_guidance": stage_info.get("guidance", ""),
                        },
                    )
                    self.cache.set(page_key, plan)
                else:
                    stats.cache_hits += 1

                # Replay planned field actions
                executed_selectors = set()
                for item in plan.get("field_map", []):
                    sel = item.get("selector", "")
                    col = item.get("csv_col", "")
                    if col and col.startswith("expected_"):
                        continue

                    val = row.get(col) if col else None
                    if val is None or str(val).strip() == "":
                        val = item.get("default_value")

                    if val is None or str(val).strip() == "":
                        continue

                    try:
                        self._apply_action(page, sel, val, item.get("action", "fill"))
                        executed_selectors.add(sel)
                    except Exception:
                        stats.heals += 1
                        self.emit("log", msg=f"[Case {case_id}] Self-healing '{col}' on {page_key}")
                        healed = self.agent.call_tool(
                            "resolve_delta_fields",
                            {"row": row, "fields": snapshot_page(page)["fields"], "failed_col": col},
                        )
                        for act in healed.get("actions", []):
                            self._apply_action(page, act["selector"], act["value"], act["action"])
                            executed_selectors.add(act["selector"])

                # Check for newly revealed conditional fields or unfilled required inputs
                curr_snap = snapshot_page(page)
                curr_fields = curr_snap.get("fields", [])
                delta_needed = [
                    f
                    for f in curr_fields
                    if f.get("tag") in ("input", "select", "textarea")
                    and f.get("selector") not in executed_selectors
                    and (f.get("required") or not f.get("value"))
                ]

                if delta_needed:
                    stats.heals += 1
                    self.emit("log", msg=f"[Case {case_id}] Detected {len(delta_needed)} unmapped/revealed fields on {page_key}")
                    healed = self.agent.call_tool(
                        "resolve_delta_fields",
                        {"row": row, "fields": curr_fields, "unmapped_fields": delta_needed},
                    )
                    for act in healed.get("actions", []):
                        self._apply_action(page, act["selector"], act["value"], act["action"])
                        executed_selectors.add(act["selector"])
                        healed_col = act.get("csv_col", "")
                        plan.setdefault("field_map", []).append({
                            "csv_col": healed_col,
                            "selector": act["selector"],
                            "action": act["action"],
                            "default_value": act["value"] if healed_col and healed_col in row else "",
                        })
                    self.cache.set(page_key, plan)

                next_btn = plan.get("next_button")
                if not next_btn:
                    candidates = [
                        f for f in curr_snap.get("fields", [])
                        if (f.get("tag") in ("button", "a") or f.get("type") in ("button", "submit"))
                        and not any(
                            k in f"{f.get('label') or ''} {f.get('id') or ''} {f.get('name') or ''}".lower()
                            for k in ("back", "cancel", "close", "dismiss", "decline", "reject")
                        )
                    ]
                    keywords = ("next", "submit", "calculate", "continue", "proceed", "checkout", "confirm", "pay", "save", "sign in", "login")
                    for f in candidates:
                        button_text = f"{f.get('label') or ''} {f.get('id') or ''} {f.get('name') or ''}".lower()
                        if f.get("type") == "submit" or any(kw in button_text for kw in keywords):
                            next_btn = f.get("selector", "")
                            break
                    if not next_btn and candidates:
                        next_btn = candidates[0].get("selector", "")

                prev_url = page.url
                if next_btn:
                    page.click(next_btn, timeout=self.cfg.step_timeout_ms)
                    page.wait_for_timeout(400)

                post = snapshot_page(page)
                advanced = page.url != prev_url or _page_signature(post) != _page_signature(curr_snap)
                if not advanced and not self.cfg.is_terminal_page(
                    page.url, post.get("bodyText", ""), page_title=post.get("title", "")
                ):
                    errors = post.get("errors", [])
                    triage = self.agent.call_tool(
                        "triage_page_blocker",
                        {
                            "errors": errors,
                            "row": row,
                            "fields": post["fields"],
                            "dialogs": post.get("dialogs", []),
                            "app_context": self.cfg.flow.app_context,
                            "stage_guidance": stage_info.get("guidance", ""),
                        },
                    )

                    effective_triage = triage
                    if (
                        not triage.get("fix_selector")
                        and triage.get("status") == "escalate"
                        and self.agent.frontier_client
                    ):
                        self.emit("log", msg=f"[Case {case_id}] Escalating blocker to frontier model...")
                        effective_triage = self.agent.call_tool(
                            "triage_page_blocker",
                            {"errors": errors, "row": row, "fields": post["fields"]},
                            use_frontier=True,
                        )

                    if effective_triage.get("fix_selector"):
                        self.emit(
                            "log",
                            msg=f"[Case {case_id}] Self-healing fix: {effective_triage.get('fix_selector')}={effective_triage.get('fix_value')}",
                        )
                        stats.heals += 1
                        caution_rec = self.memory.record_caution(
                            target_url=page.url,
                            summary=f"Requires {effective_triage.get('fix_selector')} ({effective_triage.get('action', 'fill')}): {effective_triage.get('reason', 'form blocker')}",
                            workaround=effective_triage,
                        )
                        applied_memories.append(caution_rec.id)
                        self.handoff.record_issue(
                            title=f"Form Blocker on {page_key}",
                            issue_type="UX",
                            severity="P2",
                            observed=f"Form advance blocked by unhandled requirement: {effective_triage.get('reason', 'Field required')}",
                            expected="Clear inline validation hints or accessible focus on invalid field",
                            evidence=f"Case {case_id} on {page.url}; healed using {effective_triage.get('fix_selector')}",
                            suggested_fix=f"Highlight invalid control {effective_triage.get('fix_selector')} before user attempts progression",
                            related_tests=[str(case_id)],
                        )
                        self._apply_action(
                            page,
                            effective_triage["fix_selector"],
                            effective_triage.get("fix_value", "10"),
                            effective_triage.get("action", "fill"),
                        )
                        if next_btn:
                            page.click(next_btn, timeout=self.cfg.step_timeout_ms)
                            page.wait_for_timeout(350)
                    else:
                        raise RuntimeError(
                            f"{effective_triage.get('status', 'ERROR').upper()}: {effective_triage.get('reason', 'Stalled')}"
                        )

            raise RuntimeError("Exceeded max_pages without reaching terminal state")

        except Exception as exc:
            dur = round(time.perf_counter() - t0, 2)
            screenshot_path = str(self.fail_dir / f"case_{case_id}_error.png")
            try:
                page.screenshot(path=screenshot_path)
                (self.fail_dir / f"case_{case_id}.json").write_text(
                    json.dumps({"row": row, "error": str(exc), "console": console_logs, "page_errors": page_errors}, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                pass
            for mid in applied_memories:
                self.memory.contradict(mid)
            self.handoff.record_issue(
                title=f"Runtime Error in Case {case_id}",
                issue_type="Bug",
                severity="P0",
                observed=f"Execution failed with error: {str(exc)[:150]}",
                expected="Flow completes or handles errors gracefully without crashing runner",
                evidence=f"Case {case_id}; screenshot: {screenshot_path}",
                suggested_fix="Inspect error logs, network timeouts, and element selectors",
                related_tests=[str(case_id)],
            )
            return {
                "case_id": case_id,
                "status": "ERROR",
                "actual": "",
                "expected": row.get("expected_premium", ""),
                "duration_s": dur,
                "note": str(exc)[:120],
                "screenshot": screenshot_path,
                "assertions": [],
            }
        finally:
            ctx.close()

    def _record_result(self, res: dict, stats: RunStats) -> None:
        stats.completed += 1
        stats.total = max(stats.total, stats.completed)
        stats.results.append(res)
        if res["status"] == "PASS":
            stats.passed += 1
        elif res["status"] == "FAIL":
            stats.failed += 1
        else:
            stats.errors += 1
        if self.cfg.reporting.stream_jsonl:
            try:
                stream_test_result_jsonl(self.out_dir, res)
            except Exception as err:
                logger.warning(f"Failed to stream result to JSONL: {err}")
        self.emit("row_done", result=res, stats=stats)

    def _run_autonomous_goal(self, browser) -> dict:
        case_id = "goal"
        ctx = browser.new_context(
            viewport={
                "width": self.cfg.browser.viewport_width,
                "height": self.cfg.browser.viewport_height,
            }
        )
        page = ctx.new_page()
        page_errors: list[str] = []
        page.on("pageerror", lambda err: page_errors.append(str(err)))
        t0 = time.perf_counter()
        screenshot_path: str | None = None
        try:
            page.goto(self.cfg.start_url, wait_until="load", timeout=12000)
            for step in range(self.cfg.max_pages):
                snap = snapshot_page(page)
                decision = self.agent.call_tool(
                    "plan_autonomous_step",
                    {
                        "goal": self.cfg.suite.goal,
                        "url": snap.get("url", page.url),
                        "fields": snap.get("fields", []),
                    },
                )
                action = str(decision.get("action", "")).lower()
                note = str(decision.get("reason", ""))
                if action == "goal_achieved":
                    return {
                        "case_id": case_id,
                        "status": "PASS",
                        "actual": note or "goal achieved",
                        "expected": self.cfg.suite.goal,
                        "duration_s": round(time.perf_counter() - t0, 2),
                        "note": note,
                        "screenshot": None,
                        "assertions": [],
                    }
                if action == "blocked":
                    raise RuntimeError(f"BLOCKED: {note or 'agent reported a blocker'}")
                selector = decision.get("selector", "")
                if selector:
                    resolved_action = action if action in ("click", "fill", "select", "check") else "click"
                    self._apply_action(page, selector, str(decision.get("value", "")), resolved_action)
                    page.wait_for_timeout(300)
            raise RuntimeError("Exceeded max_pages without achieving goal")
        except Exception as exc:
            dur = round(time.perf_counter() - t0, 2)
            screenshot_path = str(self.fail_dir / "goal_error.png")
            try:
                page.screenshot(path=screenshot_path)
            except Exception:
                pass
            self.handoff.record_issue(
                title="Autonomous Goal Run Failed",
                issue_type="Bug",
                severity="P1",
                observed=f"Goal run failed with error: {str(exc)[:150]}",
                expected="Agent reaches the stated goal without runtime errors",
                evidence=f"Goal: {self.cfg.suite.goal}; screenshot: {screenshot_path}",
                suggested_fix="Inspect planner decisions, selectors, and page state transitions",
            )
            return {
                "case_id": case_id,
                "status": "ERROR",
                "actual": "",
                "expected": self.cfg.suite.goal,
                "duration_s": dur,
                "note": str(exc)[:120],
                "screenshot": screenshot_path,
                "assertions": [],
            }
        finally:
            ctx.close()

    def _run_crawl_smoke(self, browser) -> list[dict]:
        ctx = browser.new_context(
            viewport={
                "width": self.cfg.browser.viewport_width,
                "height": self.cfg.browser.viewport_height,
            }
        )
        page = ctx.new_page()
        page_errors: list[str] = []
        page.on("pageerror", lambda err: page_errors.append(str(err)))
        results: list[dict] = []
        try:
            page.goto(self.cfg.start_url, wait_until="load", timeout=12000)
            base_netloc = urlparse(self.cfg.start_url).netloc
            links = page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)")
            targets = list(dict.fromkeys(
                link for link in links if urlparse(link).netloc == base_netloc
            ))[: self.cfg.max_pages]

            for idx, target in enumerate(targets, start=1):
                case_id = f"smoke-{idx}"
                before_errors = len(page_errors)
                t0 = time.perf_counter()
                try:
                    page.goto(target, wait_until="load", timeout=12000)
                    snap = snapshot_page(page)
                    new_errors = page_errors[before_errors:]
                    problems = list(snap.get("errors", [])) + new_errors
                    passed = not problems
                    results.append({
                        "case_id": case_id,
                        "status": "PASS" if passed else "FAIL",
                        "actual": target,
                        "expected": "no page errors",
                        "duration_s": round(time.perf_counter() - t0, 2),
                        "note": "; ".join(problems)[:200] if problems else "clean page load",
                        "screenshot": None,
                        "assertions": [],
                    })
                except Exception as exc:
                    results.append({
                        "case_id": case_id,
                        "status": "ERROR",
                        "actual": target,
                        "expected": "no page errors",
                        "duration_s": round(time.perf_counter() - t0, 2),
                        "note": str(exc)[:120],
                        "screenshot": None,
                        "assertions": [],
                    })
            if not results:
                results.append({
                    "case_id": "smoke-0",
                    "status": "FAIL",
                    "actual": self.cfg.start_url,
                    "expected": "at least one internal link",
                    "duration_s": 0.0,
                    "note": "No internal links found on the start page",
                    "screenshot": None,
                    "assertions": [],
                })
            return results
        except Exception as exc:
            return [{
                "case_id": "smoke-0",
                "status": "ERROR",
                "actual": self.cfg.start_url,
                "expected": "no page errors",
                "duration_s": 0.0,
                "note": str(exc)[:120],
                "screenshot": None,
                "assertions": [],
            }]
        finally:
            ctx.close()

    def run_all(self) -> RunStats:
        self.stop_requested = False
        t_start = time.perf_counter()
        mode = self.cfg.suite.mode
        rows = self._load_test_cases()
        if mode == "data_driven" and not rows:
            resolved = self.cfg.get_resolved_cases_path()
            err_msg = f"Test suite not found or empty at {resolved}"
            self.emit("log", msg=f"[bold red]Error:[/] {err_msg}")
            return RunStats()

        stats = RunStats(total=len(rows) if mode == "data_driven" else 1)
        sync_pw, active_engine, chrom_exe = _get_playwright(self.cfg.engine, self.cfg.browser.browser_type)
        if active_engine.startswith("rustwright") and self.cfg.browser.browser_type != "chromium":
            self.emit("log", msg="[yellow]Rustwright supports only chromium; switching browser type to chromium.[/]")
            self.cfg.browser.browser_type = "chromium"
        self.emit(
            "log",
            msg=f"Starting {stats.total} targets | Mode: {mode} | Engine: {active_engine} | Target: {self.cfg.start_url} | Model: {self.cfg.local_model}",
        )
        self.emit("init", rows=rows, stats=stats)

        launch_kwargs: dict[str, Any] = {
            "headless": self.cfg.headless,
            "slow_mo": self.cfg.browser.slow_mo_ms,
        }
        if chrom_exe and active_engine.startswith("rustwright"):
            launch_kwargs["executable_path"] = chrom_exe

        with sync_pw() as p:
            browser_launcher = getattr(p, self.cfg.browser.browser_type, p.chromium)
            browser = browser_launcher.launch(**launch_kwargs)
            if mode == "autonomous_goal":
                self.emit("row_start", case_id="goal")
                self._record_result(self._run_autonomous_goal(browser), stats)
            elif mode == "crawl_smoke":
                for res in self._run_crawl_smoke(browser):
                    if self.stop_requested:
                        break
                    self.emit("row_start", case_id=res["case_id"])
                    self._record_result(res, stats)
            else:
                for row in rows:
                    if self.stop_requested:
                        self.emit("log", msg="Run aborted by user.")
                        break
                    self.emit("row_start", case_id=row.get("case_id", "unknown"))
                    self._record_result(self._run_row(browser, row, stats), stats)
            browser.close()

        stats.duration_s = round(time.perf_counter() - t_start, 2)
        self.stats = stats
        self._export_artifacts(stats)
        self.emit("done", stats=stats)
        return stats

    def _export_artifacts(self, stats: RunStats):
        if not stats.results:
            return

        # 1. Export CSV
        fields = ["case_id", "status", "actual", "expected", "duration_s", "note"]
        out_csv = Path(self.cfg.results_path)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(stats.results)

        # 2. Export JSON
        run_data = {
            "stats": {
                "total": stats.total,
                "completed": stats.completed,
                "passed": stats.passed,
                "failed": stats.failed,
                "errors": stats.errors,
                "heals": stats.heals,
                "cache_hits": stats.cache_hits,
                "duration_s": stats.duration_s,
            },
            "config": self.cfg.to_dict(),
            "results": stats.results,
        }
        json_path = self.out_dir / "results.json"
        json_path.write_text(json.dumps(run_data, indent=2), encoding="utf-8")

        # 3. Export Standalone HTML Report
        if self.cfg.reporting.generate_html_report:
            html_path = self.out_dir / "report.html"
            reporting.generate_html_report(run_data, html_path)
            self.emit("log", msg=f"[bold green]Generated HTML Report:[/] {html_path}")
