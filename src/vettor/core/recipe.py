from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from urllib.parse import parse_qsl, urlencode, urlparse

from pydantic import BaseModel, Field

from vettor.core.schema import ActionType


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


class RecipeStep(BaseModel):
    """A deterministic execution step on a page."""

    action: ActionType
    selector: str
    input_key: str = ""
    default_value: str = ""


class PageRecipe(BaseModel):
    """Replay trajectory for a given normalized page key."""

    page_key: str
    steps: list[RecipeStep] = Field(default_factory=list)
    next_selector: str = ""
    hits: int = 0


class RecipeCache:
    """Thread-safe persistent recipe cache for deterministic replay and fast execution."""

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

    def get(self, url: str) -> PageRecipe | None:
        """Retrieves a cached recipe matching the normalized page key."""
        key = normalize_page_key(url)
        with self._lock:
            recipe = self.recipes.get(key)
            if recipe:
                recipe.hits += 1
                return recipe
            return None

    def set(self, url: str, recipe: PageRecipe) -> None:
        """Saves a page recipe to the cache."""
        key = normalize_page_key(url)
        recipe.page_key = key
        with self._lock:
            self.recipes[key] = recipe
            self._save()

    def record_step(self, url: str, step: RecipeStep) -> None:
        """Appends or updates a step in the recipe without duplicating the same selector."""
        key = normalize_page_key(url)
        with self._lock:
            if key not in self.recipes:
                self.recipes[key] = PageRecipe(page_key=key, steps=[step])
            else:
                recipe = self.recipes[key]
                for i, existing in enumerate(recipe.steps):
                    if existing.selector == step.selector:
                        recipe.steps[i] = step
                        break
                else:
                    recipe.steps.append(step)
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
