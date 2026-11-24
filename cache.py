import json
from pathlib import Path
from threading import Lock


class RecipeCache:
    def __init__(self, path: str = "form_recipe.json"):
        self.path = Path(path)
        self._lock = Lock()
        self.data: dict = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
        return {}

    def get(self, page_key: str) -> dict | None:
        with self._lock:
            return self.data.get(page_key)

    def set(self, page_key: str, plan: dict) -> None:
        with self._lock:
            self.data[page_key] = plan
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def clear(self) -> None:
        with self._lock:
            self.data = {}
            if self.path.exists():
                self.path.unlink()