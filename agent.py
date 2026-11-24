import json
import logging
import re
from typing import Any
from config import QAConfig
from model_provider import ModelRouter
import llm_manager

logger = logging.getLogger("vettor.agent")


def generate_synthetic_value(field_info: dict) -> tuple[str, str]:
    """Generates a contextual synthetic value and action for any standard HTML input."""
    f_type = (field_info.get("type") or "text").lower()
    f_tag = (field_info.get("tag") or "input").lower()
    f_name = (field_info.get("name") or "").lower()
    f_id = (field_info.get("id") or "").lower()
    f_label = (field_info.get("label") or "").lower()
    haystack = f"{f_id} {f_name} {f_label}".lower()

    if f_tag == "select":
        options = field_info.get("options", [])

        def is_placeholder(option: object) -> bool:
            text = str(option).strip().lower()
            return (
                not text
                or text in {"-", "--", "---", "select", "choose", "none"}
                or text.startswith(("select ", "choose ", "please ", "pick "))
            )

        val = next((o for o in options if not is_placeholder(o)), options[0] if options else "default")
        return val, "select"

    if f_type == "checkbox" or any(k in haystack for k in ("terms", "agree", "confirm", "accept", "consent")):
        return "yes", "check"

    if f_type == "email" or "email" in haystack:
        return "qa.test@example.com", "fill"

    if f_type == "password" or "password" in haystack:
        return "P@ssw0rd123!", "fill"

    if f_type == "tel" or any(k in haystack for k in ("phone", "tel", "mobile")):
        return "5551234567", "fill"

    if any(k in haystack for k in ("zip", "postal", "pincode")):
        return "90210", "fill"

    if any(k in haystack for k in ("city", "town")):
        return "Metropolis", "fill"

    if any(k in haystack for k in ("address", "street")):
        return "123 Innovation Way", "fill"

    if any(k in haystack for k in ("state", "province")):
        return "CA", "fill"

    if any(k in haystack for k in ("name", "first", "last", "user", "nominee")):
        return "Alex Tester", "fill"

    if f_type == "number" or any(k in haystack for k in ("age", "qty", "quantity", "count", "term", "cigs")):
        return "10", "fill"

    if f_type == "date" or "date" in haystack:
        return "2026-01-01", "fill"

    return "QA Test Value", "fill"


class QAAgent:
    """Fully configurable agentic brain for web test automation and self-healing."""

    def __init__(self, cfg: QAConfig):
        self.cfg = cfg
        self._is_heuristic = cfg.model.backend == "heuristic"
        self.router = ModelRouter(cfg)
        self.frontier_router = ModelRouter(cfg, use_frontier=True)
        self.frontier_client = self.frontier_router.providers[0] if self.frontier_router.providers else None

        if "lfm3" in cfg.model.model_name.lower() and not cfg.model.model_kwargs:
            cfg.model.model_kwargs = {"top_k": 50, "repeat_penalty": 1.1, "enable_thinking": False}
        logger.info(
            "QAAgent providers: local=%s frontier=%s",
            self.router.names,
            self.frontier_router.names,
        )

    def _strip_thinking(self, text: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text.strip()

    def _parse_llm_json(self, raw_str: str) -> dict:
        text = self._strip_thinking(raw_str)
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        decoder = json.JSONDecoder()
        for match in re.finditer(r"\{", text):
            try:
                value, _ = decoder.raw_decode(text[match.start() :])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value

        candidates = [text]
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end >= start:
            candidates.append(text[start : end + 1])
        for candidate in candidates:
            try:
                value = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        raise ValueError("No JSON object found in model response")

    def _extraction_is_grounded(self, parsed: dict, body: str) -> bool:
        if not isinstance(parsed, dict) or not parsed:
            return False
        text = (body or "").lower()
        if not text.strip():
            return False

        body_numbers = {
            float(value.replace(",", ""))
            for value in re.findall(r"\d[\d,]*(?:\.\d+)?", body or "")
        }
        for value in parsed.values():
            if value is None or value == "":
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if not any(abs(number - float(value)) <= 0.01 for number in body_numbers):
                    return False
            else:
                token = str(value).strip().lower()
                if token in {"ok", "confirmed", "n/a", "none", "true", "false"}:
                    continue
                if token not in text:
                    return False
        return True

    def _build_compact_fields(self, fields: list[dict]) -> tuple[list[dict], list[dict]]:
        inputs = []
        buttons = []
        for f in fields:
            c = {
                "tag": f.get("tag"),
                "selector": f.get("selector"),
            }
            if f.get("id"):
                c["id"] = f.get("id")
            if f.get("name"):
                c["name"] = f.get("name")
            if f.get("type"):
                c["type"] = f.get("type")
            if f.get("label"):
                c["label"] = f.get("label")
            if f.get("required"):
                c["required"] = True

            if f.get("tag") in ("button", "a") or f.get("type") in ("button", "submit"):
                buttons.append(c)
            elif f.get("tag") in ("input", "select", "textarea"):
                inputs.append(c)
        return inputs, buttons

    def _build_lfm3_prompts(self, tool_name: str, payload: dict) -> tuple[str, str]:
        app_context = payload.get("app_context", "") or "Web application multi-step workflow"
        stage_name = payload.get("stage_name", "")
        stage_guidance = payload.get("stage_guidance", "")

        if tool_name == "map_page_schema":
            csv_cols = [c for c in payload.get("csv_columns", []) if not c.startswith("expected_") and c != "case_id"]
            inputs, buttons = self._build_compact_fields(payload.get("fields", []))
            system_msg = (
                "You are an automated, domain-agnostic Web QA test harness agent. "
                f"Application Context: {app_context}\n"
                f"Current Workflow Stage: {stage_name or 'Auto-detected stage'}\n"
                f"Stage Guidance: {stage_guidance or 'Identify interactive inputs and pair with test parameters'}\n"
                "Map test case columns to HTML form inputs on the current page and identify the next/submit button. "
                "Return ONLY a valid JSON object matching the requested schema."
            )
            user_msg = (
                f"Current Page URL: {payload.get('page_url', '')}\n"
                f"Page Title: {payload.get('page_title', '')}\n"
                f"Test parameters available: {json.dumps(csv_cols)}\n"
                f"Form inputs on page: {json.dumps(inputs)}\n"
                f"Buttons on page: {json.dumps(buttons)}\n\n"
                "Instructions:\n"
                "- Pair each form input to a matching test column if one exists.\n"
                "- If a form input is not in the test parameters, set csv_col to \"\" and provide a sensible default_value.\n"
                "- CRITICAL: ONLY output selectors from 'Form inputs on page'. Do not invent selectors.\n"
                "- action must be one of: 'fill', 'select', 'check'.\n"
                "- If there are no form inputs and this is a completion/summary page, set is_final_page to true.\n\n"
                "Respond with ONLY valid JSON adhering to this exact schema:\n"
                "{\n"
                '  "field_map": [\n'
                '    {"csv_col": "field_name", "selector": "#selector", "action": "fill", "default_value": ""}\n'
                "  ],\n"
                '  "next_button": "#btn-next",\n'
                '  "is_final_page": false\n'
                "}"
            )
            return system_msg, user_msg

        elif tool_name == "resolve_delta_fields":
            inputs, _ = self._build_compact_fields(payload.get("fields", []))
            row = payload.get("row", {})
            failed_col = payload.get("failed_col")
            system_msg = (
                "You are an automated Web QA test harness agent. "
                "Provide input actions for newly revealed or missing form fields using the test case data. "
                "Return ONLY a valid JSON object matching the requested schema."
            )
            user_msg = (
                f"Test data: {json.dumps(row)}\n"
                f"Form fields needing input: {json.dumps(inputs)}\n"
                f"Failed parameter: {failed_col}\n\n"
                "Instructions:\n"
                "- CRITICAL: ONLY output selectors from 'Form fields needing input'.\n"
                "- action must be one of: 'fill', 'select', 'check'.\n\n"
                "Respond with ONLY valid JSON adhering to this exact schema:\n"
                "{\n"
                '  "actions": [\n'
                '    {"selector": "#field_id", "value": "value", "action": "fill", "csv_col": "col_name"}\n'
                "  ]\n"
                "}"
            )
            return system_msg, user_msg

        elif tool_name == "triage_page_blocker":
            errors = payload.get("errors", [])
            dialogs = payload.get("dialogs", [])
            inputs, buttons = self._build_compact_fields(payload.get("fields", []))
            row = payload.get("row", {})
            system_msg = (
                "You are an automated Web QA test harness agent. "
                "Diagnose why the web flow failed to advance and prescribe a fix (format repair, dismiss modal, check declaration) or report an app bug. "
                "Return ONLY a valid JSON object matching the requested schema."
            )
            user_msg = (
                f"Validation errors on page: {json.dumps(errors)}\n"
                f"Active dialogs / modals: {json.dumps(dialogs)}\n"
                f"Form fields: {json.dumps(inputs)}\n"
                f"Test data: {json.dumps(row)}\n\n"
                "Instructions:\n"
                "- status must be one of: 'fix_format', 'app_bug', 'escalate'.\n"
                "- action must be one of: 'fill', 'select', 'check', 'click'.\n\n"
                "Respond with ONLY valid JSON adhering to this exact schema:\n"
                "{\n"
                '  "status": "fix_format",\n'
                '  "reason": "Missing required field or terms checkbox",\n'
                '  "fix_selector": "#agree_terms",\n'
                '  "fix_value": "yes",\n'
                '  "action": "check"\n'
                "}"
            )
            return system_msg, user_msg

        elif tool_name == "extract_final_result":
            body_text = payload.get("bodyText", "")[:2500]
            target_keys = payload.get("target_keys", []) or ["total", "premium", "order_id", "quote_id", "status"]
            system_msg = (
                "You are an automated Web QA test harness agent. "
                f"Application Context: {app_context}\n"
                f"Extract target test assertion values ({', '.join(target_keys)}) from the confirmation/summary page. "
                "Return ONLY a valid JSON dictionary mapping each target key to its extracted value."
            )
            sample_out = {k: "123.45" if any(w in k for w in ("total", "amount", "price", "premium")) else "CONF-1234" for k in target_keys}
            user_msg = (
                f"Page Text:\n{body_text}\n\n"
                f"Target keys to extract: {json.dumps(target_keys)}\n\n"
                "Respond with ONLY valid JSON matching this schema:\n"
                f"{json.dumps(sample_out, indent=2)}"
            )
            return system_msg, user_msg

        elif tool_name == "plan_autonomous_step":
            goal = payload.get("goal", "")
            inputs, buttons = self._build_compact_fields(payload.get("fields", []))
            system_msg = (
                "You are an autonomous Web QA agent. Given the target goal and current page state, "
                "decide the next atomic action to progress toward completing the test goal."
            )
            user_msg = (
                f"Test Goal: {goal}\n"
                f"Current URL: {payload.get('url', '')}\n"
                f"Interactive inputs: {json.dumps(inputs)}\n"
                f"Buttons / links: {json.dumps(buttons)}\n\n"
                "Respond with ONLY valid JSON adhering to this schema:\n"
                "{\n"
                '  "action": "click" | "fill" | "select" | "check" | "goal_achieved" | "blocked",\n'
                '  "selector": "#selector",\n'
                '  "value": "text_if_filling",\n'
                '  "reasoning": "Explanation of action"\n'
                "}"
            )
            return system_msg, user_msg

        return (
            "You are a QA automation assistant.",
            f"Invoke tool {tool_name} with: {json.dumps(payload)}",
        )

    def _heuristic_fallback(self, tool_name: str, payload: dict) -> dict:
        """Domain-agnostic deterministic heuristic fallback for web automation."""
        if tool_name == "map_page_schema":
            csv_columns = [c for c in payload.get("csv_columns", []) if not c.startswith("expected_") and c != "case_id"]
            fields = payload.get("fields", [])
            field_map = []
            next_button = ""

            input_fields = [f for f in fields if f.get("tag") in ("input", "select", "textarea")]
            buttons = [f for f in fields if f.get("tag") in ("button", "a") or f.get("type") in ("button", "submit")]

            if not input_fields and not buttons:
                return {"field_map": [], "next_button": "", "is_final_page": True}

            for f in input_fields:
                f_name = (f.get("name") or "").lower()
                f_id = (f.get("id") or "").lower()
                f_label = (f.get("label") or "").lower()
                selector = f.get("selector", "")
                f_type = f.get("type", "text")

                matched_col = None
                fn_norm = f_name.replace("_", "").replace(" ", "")
                fi_norm = f_id.replace("_", "").replace(" ", "")
                fl_norm = f_label.replace("_", "").replace(" ", "")

                # 1. Exact normalized match first
                for col in csv_columns:
                    c_norm = col.lower().replace("_", "").replace(" ", "")
                    if c_norm and c_norm in (fn_norm, fi_norm, fl_norm):
                        matched_col = col
                        break

                # 2. Token-level match
                if not matched_col:
                    field_tokens = set(re.findall(r"\w+", f"{f_name} {f_id} {f_label}".lower()))
                    for col in sorted(csv_columns, key=len, reverse=True):
                        col_tokens = set(re.findall(r"\w+", col.lower()))
                        if col_tokens and col_tokens.issubset(field_tokens):
                            matched_col = col
                            break

                synth_val, synth_act = generate_synthetic_value(f)
                action = "select" if f.get("tag") == "select" else ("check" if f_type == "checkbox" else "fill")
                default_val = "" if matched_col else synth_val

                field_map.append({
                    "csv_col": matched_col or "",
                    "selector": selector,
                    "action": action if matched_col else synth_act,
                    "default_value": default_val,
                })

            keywords = ("next", "submit", "calculate", "continue", "proceed", "checkout", "confirm", "pay", "save", "sign in", "login")
            blockers = ("back", "cancel", "close", "dismiss", "decline", "reject")
            candidates = [
                b for b in buttons
                if not any(k in f"{b.get('label') or ''} {b.get('id') or ''} {b.get('name') or ''}".lower() for k in blockers)
            ]
            for b in candidates:
                b_text = f"{b.get('label') or ''} {b.get('id') or ''} {b.get('name') or ''}".lower()
                if b.get("type") == "submit" or any(kw in b_text for kw in keywords):
                    next_button = b.get("selector", "")
                    break
            if not next_button and candidates:
                next_button = candidates[0].get("selector", "")

            is_final = len(input_fields) == 0 and not next_button
            return {"field_map": field_map, "next_button": next_button, "is_final_page": is_final}

        elif tool_name == "resolve_delta_fields":
            fields = payload.get("fields", [])
            row = payload.get("row", {})
            failed_col = payload.get("failed_col")
            actions = []

            for f in fields:
                if f.get("tag") in ("input", "select", "textarea"):
                    selector = f.get("selector", "")
                    synth_val, synth_act = generate_synthetic_value(f)

                    val = ""
                    if failed_col and failed_col in row:
                        val = str(row[failed_col])
                    else:
                        # Match row key by token
                        f_haystack = f"{f.get('name', '')} {f.get('id', '')} {f.get('label', '')}".lower()
                        for k, v in row.items():
                            if k.lower() in f_haystack:
                                val = str(v)
                                break
                    if not val:
                        val = synth_val

                    if val:
                        actions.append({"selector": selector, "value": val, "action": synth_act, "csv_col": failed_col or ""})

            return {"actions": actions}

        elif tool_name == "triage_page_blocker":
            errors = payload.get("errors", [])
            fields = payload.get("fields", [])
            dialogs = payload.get("dialogs", [])
            err_text = " ".join(errors).lower()

            # Check if there's an active dialog or popup blocking progression
            for d in dialogs:
                if d.get("dismiss_selector"):
                    return {
                        "status": "fix_format",
                        "reason": f"Dismissing blocking modal: {d.get('text', '')[:40]}",
                        "fix_selector": d["dismiss_selector"],
                        "fix_value": "",
                        "action": "click",
                    }

            if any(k in err_text for k in ("term", "agree", "declaration", "consent", "accept")):
                agree_field = next(
                    (f for f in fields if any(k in f.get("id", "") + f.get("name", "") + f.get("label", "") for k in ("term", "agree", "consent", "accept"))),
                    None,
                )
                return {
                    "status": "fix_format",
                    "reason": "Terms agreement or declaration must be accepted",
                    "fix_selector": agree_field.get("selector") if agree_field else "#agree_terms",
                    "fix_value": "yes",
                    "action": "check",
                }

            # Check error text against form fields
            for f in fields:
                f_name = (f.get("name") or "").lower()
                f_id = (f.get("id") or "").lower()
                f_label = (f.get("label") or "").lower()
                f_tokens = re.findall(r"\w+", f"{f_name} {f_id} {f_label}")

                if any(t in err_text for t in f_tokens if len(t) > 2) or (f.get("required") and not f.get("value")):
                    synth_val, synth_act = generate_synthetic_value(f)
                    return {
                        "status": "fix_format",
                        "reason": f"Resolved missing or invalid field: {f_label or f_name or f_id}",
                        "fix_selector": f.get("selector", ""),
                        "fix_value": synth_val,
                        "action": synth_act,
                    }

            return {
                "status": "app_bug",
                "reason": f"Unhandled blocker: {errors[0] if errors else 'Workflow stalled'}",
            }

        elif tool_name == "extract_final_result":
            body = payload.get("bodyText", "")
            target_keys = payload.get("target_keys", []) or []
            extracted: dict[str, Any] = {}

            prem_match = re.search(r"(?:Annual\s+Premium|Premium|Total|Amount|Price|Balance|Subtotal):\s*\$?([0-9,]+(?:\.[0-9]+)?)", body, re.IGNORECASE)
            if not prem_match:
                prem_match = re.search(r"\$([0-9,]+(?:\.[0-9]+)?)", body)
            val_num = float(prem_match.group(1).replace(",", "")) if prem_match else 0.0

            id_match = re.search(r"(?:Quote\s+Reference|Reference|Quote\s+ID|Order\s+ID|Order\s+Number|Confirmation\s+Code|Booking\s+ID|Confirmation):\s*([A-Za-z0-9\-]+)", body, re.IGNORECASE)
            if not id_match:
                id_match = re.search(r"\b([A-Z0-9]{2,5}-[A-Z0-9]{4,8})\b", body)
            val_id = id_match.group(1) if id_match else "CONFIRMED"

            extracted["premium"] = val_num
            extracted["total"] = val_num
            extracted["amount"] = val_num
            extracted["price"] = val_num
            extracted["quote_id"] = val_id
            extracted["order_id"] = val_id
            extracted["ref_id"] = val_id

            for key in target_keys:
                clean_k = key.lower().strip()
                if clean_k in extracted:
                    continue
                if any(w in clean_k for w in ("total", "price", "amount", "cost", "premium", "balance", "fee")):
                    extracted[clean_k] = val_num
                elif any(w in clean_k for w in ("id", "ref", "code", "number", "ticket")):
                    extracted[clean_k] = val_id
                elif "status" in clean_k:
                    extracted[clean_k] = "Confirmed"
                else:
                    m = re.search(rf"{clean_k}[:\s]+([^\n\r<]+)", body, re.IGNORECASE)
                    extracted[clean_k] = m.group(1).strip() if m else "OK"

            return extracted

        elif tool_name == "plan_autonomous_step":
            if self.cfg.is_terminal_page(payload.get("url", "")):
                return {"action": "goal_achieved", "reason": "terminal page reached"}

            fields = payload.get("fields", [])
            for f in fields:
                if f.get("tag") not in ("input", "select", "textarea"):
                    continue
                if f.get("type") == "checkbox":
                    if f.get("checked"):
                        continue
                elif f.get("value"):
                    continue
                value, action = generate_synthetic_value(f)
                return {
                    "action": action,
                    "selector": f.get("selector", ""),
                    "value": value,
                    "reason": f"populate {f.get('name') or f.get('id') or f.get('label') or 'field'}",
                }

            buttons = [
                f for f in fields
                if (f.get("tag") in ("button", "a") or f.get("type") in ("button", "submit"))
                and not any(
                    k in f"{f.get('label') or ''} {f.get('id') or ''}".lower()
                    for k in ("back", "cancel", "close", "dismiss")
                )
            ]
            for b in buttons:
                button_text = f"{b.get('label') or ''} {b.get('id') or ''}".lower()
                if b.get("type") == "submit" or any(
                    kw in button_text for kw in ("next", "submit", "calculate", "continue", "confirm", "pay")
                ):
                    return {"action": "click", "selector": b.get("selector", ""), "reason": "advance workflow"}
            if buttons:
                return {"action": "click", "selector": buttons[0].get("selector", ""), "reason": "advance workflow"}
            return {"action": "blocked", "reason": "no actionable control on page"}

        return {}

    def call_tool(self, tool_name: str, payload: dict, use_frontier: bool = False) -> dict:
        if self._is_heuristic and not (use_frontier and self.frontier_client):
            return self._heuristic_fallback(tool_name, payload)

        is_frontier = use_frontier and bool(self.frontier_client)
        router = self.frontier_router if is_frontier else self.router

        system_msg, user_msg = self._build_lfm3_prompts(tool_name, payload)
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        try:
            completion = router.complete(
                messages,
                temperature=self.cfg.model.temperature,
                max_tokens=self.cfg.model.max_tokens,
                timeout=self.cfg.model.timeout_s,
                response_format={"type": "json_object"},
                extra_body=self.cfg.model.model_kwargs or None,
            )
            if completion is None:
                logger.warning("No model provider could answer '%s'; using heuristic fallback", tool_name)
                return self._heuristic_fallback(tool_name, payload)

            content = completion.text
            parsed = self._parse_llm_json(content)

            # Sanitize and validate map_page_schema
            if tool_name == "map_page_schema" and "field_map" in parsed:
                inputs, buttons = self._build_compact_fields(payload.get("fields", []))
                input_selectors = {f["selector"]: f for f in inputs if f.get("selector")}
                btn = parsed.get("next_button", "")
                if not btn and buttons:
                    btn = buttons[0].get("selector", "")
                    parsed["next_button"] = btn

                sanitized_map = []
                mapped_selectors = set()
                raw_field_map = parsed.get("field_map", [])
                if not isinstance(raw_field_map, list):
                    raw_field_map = []

                for item in raw_field_map:
                    if not isinstance(item, dict):
                        continue
                    sel = item.get("selector", "")
                    col = item.get("csv_col", "")
                    if sel not in input_selectors or sel == btn or sel in mapped_selectors:
                        continue
                    if col and (col.startswith("expected_") or col == "case_id"):
                        continue

                    f_info = input_selectors.get(sel, {})
                    f_name = (f_info.get("name") or "").lower()
                    f_id = (f_info.get("id") or "").lower()
                    f_label = (f_info.get("label") or "").lower()
                    f_type = f_info.get("type", "")
                    f_text = f"{f_name} {f_id} {f_label}"

                    # Clear col if zero overlap with field attributes
                    if col:
                        col_tokens = col.lower().replace("_", " ").split()
                        if not any(token in f_text for token in col_tokens):
                            item["csv_col"] = ""
                            col = ""

                    if f_type == "checkbox" or any(k in f_id for k in ("agree", "terms", "confirm", "accept")):
                        item["action"] = "check"
                        if not item.get("default_value"):
                            item["default_value"] = "yes"

                    sanitized_map.append(item)
                    mapped_selectors.add(sel)

                # Complement with heuristics for any unmapped input fields
                fallback = self._heuristic_fallback(tool_name, payload)
                for item in fallback.get("field_map", []):
                    if item["selector"] not in mapped_selectors and item["selector"] != btn:
                        sanitized_map.append(item)
                        mapped_selectors.add(item["selector"])

                parsed["field_map"] = sanitized_map
                return parsed

            if tool_name == "resolve_delta_fields" and "actions" in parsed:
                inputs, _ = self._build_compact_fields(payload.get("fields", []))
                input_selectors = {f["selector"] for f in inputs if f.get("selector")}
                raw_actions = parsed.get("actions", [])
                if isinstance(raw_actions, list):
                    parsed["actions"] = [a for a in raw_actions if isinstance(a, dict) and a.get("selector") in input_selectors]
                if not parsed.get("actions"):
                    return self._heuristic_fallback(tool_name, payload)
                return parsed

            if tool_name == "triage_page_blocker" and ("status" in parsed or "fix_selector" in parsed):
                if parsed.get("fix_selector") and parsed.get("status") == "app_bug":
                    parsed["status"] = "fix_format"
                return parsed

            if tool_name == "extract_final_result" and parsed and isinstance(parsed, dict):
                raw_num = parsed.get("premium", parsed.get("total", parsed.get("amount", parsed.get("price"))))
                if raw_num is not None:
                    clean_str = re.sub(r"[^\d.]", "", str(raw_num))
                    try:
                        num_val = float(clean_str) if clean_str else raw_num
                    except Exception:
                        num_val = raw_num
                    parsed["premium"] = num_val
                    parsed.setdefault("total", num_val)
                    parsed.setdefault("amount", num_val)
                    parsed.setdefault("price", num_val)
                id_val = parsed.get("quote_id", parsed.get("order_id", parsed.get("ref_id", parsed.get("id"))))
                if id_val is not None:
                    parsed.setdefault("quote_id", id_val)
                    parsed.setdefault("order_id", id_val)
                    parsed.setdefault("ref_id", id_val)
                if not self._extraction_is_grounded(parsed, payload.get("bodyText", "")):
                    logger.warning("LLM extraction was not grounded in page text; using heuristic fallback")
                    return self._heuristic_fallback(tool_name, payload)
                return parsed

            logger.warning("LLM response for %s lacked expected keys: %s", tool_name, parsed)
            return self._heuristic_fallback(tool_name, payload)

        except Exception as exc:
            logger.warning("LLM invocation '%s' failed (%s). Using heuristic fallback.", tool_name, exc)
            return self._heuristic_fallback(tool_name, payload)
