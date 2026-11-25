from __future__ import annotations

import json
import re
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse

from pydantic import BaseModel, Field

from vettor.core.schema import ActionType, DOMElement


def normalize_page_key(url: str) -> str:
    """Normalizes a URL by stripping volatile query params (sid, session, token, ts, timestamp, _)

    while retaining path, non-volatile query parameters, and fragments.
    Example: 'http://localhost:8000/form?sid=123&step=2' -> '/form?step=2'
    """
    if not url:
        return ""
    if "://" in url:
        parsed = urlparse(url)
    elif url.startswith("/"):
        parsed = urlparse(f"http://dummy{url}")
    elif url.startswith("?"):
        parsed = urlparse(f"http://dummy/{url}")
    else:
        parsed = urlparse(f"http://dummy/{url}")

    path = parsed.path or "/"
    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    volatile = {"sid", "session", "token", "ts", "timestamp", "_"}
    clean_pairs = sorted([(k, v) for k, v in query_pairs if k.lower() not in volatile])
    query_str = f"?{urlencode(clean_pairs)}" if clean_pairs else ""
    frag_str = f"#{parsed.fragment}" if parsed.fragment else ""
    return f"{path}{query_str}{frag_str}"


def _redact_recipe_value(val: str, input_key: str = "") -> str:
    """Redacts sensitive values before persisting to recipe.json (B-18)."""
    if not val:
        return ""
    key_lower = input_key.lower()
    if any(s in key_lower for s in ("pass", "secret", "token", "ssn", "card", "cvv", "pin", "auth")):
        return f"<placeholder:{input_key or 'secret'}>"
    if re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", val):
        return f"<placeholder:{input_key or 'email'}>"
    if re.search(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", val):
        return "<placeholder:jwt>"
    return val


class LocatorFingerprint(BaseModel):
    """Rich multi-attribute element fingerprint for drift detection and deterministic self-healing."""

    role: str = ""
    label: str = ""
    text: str = ""
    tag: str = ""
    type: str = ""
    name: str = ""
    id: str = ""
    placeholder: str = ""
    selector: str = ""
    region: str = ""
    neighbor_labels: list[str] = Field(default_factory=list)

    @classmethod
    def from_element(cls, el: DOMElement) -> LocatorFingerprint:
        return cls(
            role=el.role,
            label=el.label,
            text=el.label,
            tag=el.tag,
            type=el.type,
            name=el.name,
            id=el.id,
            placeholder=el.placeholder,
            selector=el.selector,
            region=el.region,
            neighbor_labels=list(el.neighbor_labels),
        )


def _norm_tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", (text or "").lower()) if t}


def resolve_fingerprint(
    fp: LocatorFingerprint,
    elements: list[DOMElement],
) -> tuple[DOMElement | None, str]:
    """Deterministic healing ladder for a drifted RecipeStep.

    Returns (matched_element, method_name) where method_name is one of:
    'exact_selector', 'id_or_name', 'role_and_label', 'label_similarity', or 'unresolved'.
    """
    if not elements:
        return None, "unresolved"

    # 1. Exact selector match
    if fp.selector:
        for el in elements:
            if el.selector == fp.selector:
                return el, "exact_selector"

    # 2. Stable id or name + tag match
    if fp.id:
        for el in elements:
            if el.id == fp.id and (not fp.tag or el.tag == fp.tag):
                return el, "id_or_name"
    if fp.name:
        for el in elements:
            if el.name == fp.name and (not fp.tag or el.tag == fp.tag):
                return el, "id_or_name"

    # 3. Role + exact label/placeholder match
    fp_label = (fp.label or fp.text or fp.placeholder).strip().lower()
    if fp_label:
        for el in elements:
            el_label = (el.label or el.placeholder).strip().lower()
            if el_label == fp_label and (not fp.tag or el.tag == fp.tag):
                return el, "role_and_label"

    # 4. Token overlap similarity across label/name/id/placeholder
    fp_tokens = _norm_tokens(f"{fp.label} {fp.name} {fp.id} {fp.placeholder} {fp.text}")
    if fp_tokens:
        best_score = 0.0
        best_el: DOMElement | None = None
        for el in elements:
            if fp.tag and el.tag != fp.tag:
                # Allow button <-> a[role=button] equivalence
                if not ({fp.tag, el.tag} <= {"button", "a"}):
                    continue
            el_tokens = _norm_tokens(f"{el.label} {el.name} {el.id} {el.placeholder}")
            if not el_tokens:
                continue
            overlap = len(fp_tokens & el_tokens) / max(1, len(fp_tokens | el_tokens))
            if fp.region and el.region and fp.region == el.region:
                overlap += 0.1
            if overlap > best_score:
                best_score = overlap
                best_el = el
        if best_el is not None and best_score >= 0.35:
            return best_el, "label_similarity"

    return None, "unresolved"


class RecipeStep(BaseModel):
    """A deterministic execution step on a page."""

    action: ActionType
    selector: str
    input_key: str = ""
    default_value: str = ""
    fingerprint: LocatorFingerprint | None = None


class PageRecipe(BaseModel):
    """Replay trajectory for a given normalized page key."""

    page_key: str
    intent: str = ""
    dom_signature: str = ""
    version: int = 1
    steps: list[RecipeStep] = Field(default_factory=list)
    next_selector: str = ""
    hits: int = 0
    heal_history: list[dict[str, Any]] = Field(default_factory=list)


class RecipeCache:
    """Thread-safe persistent recipe cache for deterministic replay and self-healing."""

    def __init__(self, path: str | Path = "artifacts/recipe.json"):
        self.path = Path(path)
        self._lock = Lock()
        self.recipes: dict[str, PageRecipe] = self._load()

    def _load(self) -> dict[str, PageRecipe]:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8"))
                return {
                    k: PageRecipe.model_validate(v) if isinstance(v, dict) else v
                    for k, v in raw.items()
                }
            except Exception:
                return {}
        return {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        dump = {k: v.model_dump() for k, v in self.recipes.items()}
        self.path.write_text(json.dumps(dump, indent=2), encoding="utf-8")

    @staticmethod
    def _compose_key(url: str, intent: str = "") -> str:
        base = normalize_page_key(url)
        clean_intent = re.sub(r"\s+", " ", (intent or "").strip().lower())[:80]
        return f"{base}::{clean_intent}" if clean_intent else base

    def get(self, url: str, intent: str = "", dom_signature: str = "") -> PageRecipe | None:
        """Retrieves a cached recipe matching (page_key, intent) or falling back to page_key."""
        base_key = normalize_page_key(url)
        intent_key = self._compose_key(url, intent) if intent else base_key
        with self._lock:
            recipe = self.recipes.get(intent_key) or self.recipes.get(base_key)
            if recipe is None and not intent:
                prefix = f"{base_key}::"
                for k, v in self.recipes.items():
                    if k.startswith(prefix):
                        recipe = v
                        break
            if recipe:
                recipe.hits += 1
                return recipe
            return None

    def set(self, url: str, recipe: PageRecipe, intent: str = "") -> None:
        """Saves a page recipe to the cache."""
        key = self._compose_key(url, intent or recipe.intent)
        recipe.page_key = normalize_page_key(url)
        with self._lock:
            self.recipes[key] = recipe
            self._save()

    def promote_pending(
        self,
        url: str,
        steps: list[RecipeStep],
        *,
        intent: str = "",
        dom_signature: str = "",
    ) -> None:
        """Promotes a verified pending trace to the persistent RecipeCache only after PASS (B-06, B-18)."""
        if not steps:
            return
        sanitized_steps: list[RecipeStep] = []
        for s in steps:
            s_copy = s.model_copy(deep=True)
            if s_copy.input_key:
                # Never persist raw PII/test data when input_key is known; replay uses test_case.inputs[input_key]
                s_copy.default_value = ""
            else:
                s_copy.default_value = _redact_recipe_value(s_copy.default_value)
            sanitized_steps.append(s_copy)

        key = self._compose_key(url, intent)
        with self._lock:
            existing = self.recipes.get(key)
            version = (existing.version + 1) if existing else 1
            self.recipes[key] = PageRecipe(
                page_key=normalize_page_key(url),
                intent=intent,
                dom_signature=dom_signature,
                version=version,
                steps=sanitized_steps,
            )
            self._save()

    def record_step(self, url: str, step: RecipeStep, intent: str = "") -> None:
        """Appends or updates a step in the recipe without duplicating the same selector."""
        key = self._compose_key(url, intent)
        s_copy = step.model_copy(deep=True)
        s_copy.default_value = _redact_recipe_value(s_copy.default_value, s_copy.input_key)
        with self._lock:
            if key not in self.recipes:
                self.recipes[key] = PageRecipe(page_key=normalize_page_key(url), intent=intent, steps=[s_copy])
            else:
                recipe = self.recipes[key]
                for i, existing in enumerate(recipe.steps):
                    if existing.selector == s_copy.selector:
                        recipe.steps[i] = s_copy
                        break
                else:
                    recipe.steps.append(s_copy)
            self._save()

    def clear(self) -> None:
        """Empties the cache and removes the persisted file."""
        with self._lock:
            self.recipes.clear()
            if self.path.exists():
                try:
                    self.path.unlink()
                except Exception:
                    pass
