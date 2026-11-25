from __future__ import annotations

import re
from typing import Any

from vettor.core.schema import AgentAction, DOMElement, PageObservation, TestCase

SYNONYM_GROUPS: list[set[str]] = [
    {"email", "mail", "e_mail", "email_address"},
    {"username", "user", "login", "account", "user_name"},
    {"password", "pass", "pwd", "passcode"},
    {"name", "full_name", "fullname", "first_name", "customer_name"},
    {"phone", "mobile", "tel", "telephone", "phone_number"},
    {"zip", "postal", "postcode", "zip_code", "postal_code"},
    {"address", "street", "addr", "street_address"},
    {"state", "province", "region"},
    {"country", "nation"},
    {"age", "years", "dob", "birth"},
    {"amount", "total", "price", "value", "coverage", "premium"},
    {"terms", "agree", "accept", "consent", "tos"},
]


def _tokens(text: str) -> set[str]:
    raw = {t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if t}
    expanded = set(raw)
    for t in raw:
        for grp in SYNONYM_GROUPS:
            if t in grp:
                expanded.update(grp)
    return expanded


def _is_truthy(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes", "on", "check", "checked")


def _is_falsy(val: Any) -> bool:
    if isinstance(val, bool):
        return not val
    return str(val).strip().lower() in ("false", "0", "no", "off", "uncheck", "unchecked")


def match_element_for_key(key: str, elements: list[DOMElement]) -> DOMElement | None:
    """Deterministically matches an input key to the best unfilled interactive DOMElement."""
    key_norm = re.sub(r"[^a-z0-9]", "", key.lower())
    key_toks = _tokens(key)
    best_el: DOMElement | None = None
    best_score = 0.0

    for el in elements:
        if el.disabled or el.tag in ("button", "a"):
            continue
        # Direct id/name exact normalized match
        el_id_norm = re.sub(r"[^a-z0-9]", "", el.id.lower())
        el_name_norm = re.sub(r"[^a-z0-9]", "", el.name.lower())
        if key_norm and (key_norm == el_id_norm or key_norm == el_name_norm):
            return el

        blob = f"{el.label} {el.name} {el.id} {el.placeholder} {el.type}"
        el_toks = _tokens(blob)
        if not el_toks or not key_toks:
            continue
        overlap = len(key_toks & el_toks) / max(1, len(key_toks))
        if overlap > best_score:
            best_score = overlap
            best_el = el

    return best_el if best_score >= 0.45 else None


def heuristic_next_action(
    obs: PageObservation,
    test_case: TestCase,
    attempted_vids: set[int] | None = None,
) -> AgentAction | None:
    """Zero-LLM deterministic heuristic tier.

    1. Dismisses blocking cookie/consent dialogs if present.
    2. Fills any unmatched/unfilled input from test_case.inputs.
    3. Clicks primary submit/next/continue button when all matched inputs are satisfied.
    Returns None if no confident deterministic action exists (escalates to LLM tier).
    """
    attempted = attempted_vids or set()

    # 1. Dismiss cookie/modal dialogs if a clear dismiss button exists
    if obs.dialogs:
        for el in obs.elements:
            if el.region == "dialog" and el.tag in ("button", "a") and el.vid not in attempted:
                lbl = (el.label or "").lower()
                if any(w in lbl for w in ("accept", "agree", "ok", "close", "dismiss", "got it", "continue")):
                    return AgentAction(action="click", vid=el.vid, reason=f"heuristic:dismiss_dialog:{el.label}")

    # 2. Match test_case.inputs to unfilled form controls
    for key, raw_val in test_case.inputs.items():
        if raw_val is None:
            continue
        val_str = str(raw_val)
        el = match_element_for_key(key, obs.elements)
        if el is None or el.vid in attempted:
            continue

        tag = el.tag.lower()
        itype = el.type.lower()

        if tag == "input" and itype in ("checkbox", "radio"):
            want_checked = _is_truthy(raw_val)
            if itype == "checkbox":
                if want_checked and not el.checked:
                    return AgentAction(action="check", vid=el.vid, value="true", reason=f"heuristic:check:{key}")
                if _is_falsy(raw_val) and el.checked:
                    return AgentAction(action="uncheck", vid=el.vid, value="false", reason=f"heuristic:uncheck:{key}")
            elif itype == "radio" and want_checked and not el.checked:
                return AgentAction(action="check", vid=el.vid, value=val_str, reason=f"heuristic:radio:{key}")
            continue

        if tag == "select":
            if el.value.strip().lower() != val_str.strip().lower():
                # Verify option exists or pick closest
                matched_opt = val_str
                for opt in el.options:
                    if opt.strip().lower() == val_str.strip().lower() or val_str.strip().lower() in opt.strip().lower():
                        matched_opt = opt
                        break
                return AgentAction(action="select", vid=el.vid, value=matched_opt, reason=f"heuristic:select:{key}")
            continue

        # Text/number/email/textarea
        if el.value != val_str:
            return AgentAction(action="fill", vid=el.vid, value=val_str, reason=f"heuristic:fill:{key}")

    # 3. All matched inputs are filled -> look for primary submit / next / continue button
    submit_candidates: list[tuple[int, DOMElement]] = []
    for el in obs.elements:
        if el.disabled or el.vid in attempted:
            continue
        if el.tag == "button" or (el.tag == "input" and el.type in ("submit", "button")) or el.role == "button":
            lbl = (el.label or el.value or "").lower()
            score = 0
            if el.type == "submit":
                score += 30
            if any(w in lbl for w in ("submit", "continue", "next", "calculate", "quote", "login", "sign in", "save", "checkout", "confirm")):
                score += 25
            if el.region == "main":
                score += 10
            if score > 0:
                submit_candidates.append((score, el))

    if submit_candidates:
        submit_candidates.sort(key=lambda x: -x[0])
        best_btn = submit_candidates[0][1]
        return AgentAction(action="click", vid=best_btn.vid, reason=f"heuristic:submit:{best_btn.label or best_btn.selector}")

    return None
