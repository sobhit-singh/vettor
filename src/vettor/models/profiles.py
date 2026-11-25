from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

ProfileTierName = Literal["sub1b", "small", "medium", "frontier"]


@dataclass(frozen=True)
class ModelCapabilityProfile:
    """Profile tuning prompt budget, decoding format, and batching per model size tier."""

    tier: ProfileTierName
    max_elements: int
    body_chars: int
    history_turns: int
    prompt_style: Literal["line", "json", "json_schema"]
    batch_actions: bool
    system_prompt_max_tokens: int
    temperature: float = 0.1
    max_tokens: int = 256


MODEL_PROFILES: dict[ProfileTierName, ModelCapabilityProfile] = {
    "sub1b": ModelCapabilityProfile(
        tier="sub1b",
        max_elements=14,
        body_chars=120,
        history_turns=1,
        prompt_style="line",
        batch_actions=False,
        system_prompt_max_tokens=100,
        temperature=0.0,
        max_tokens=96,
    ),
    "small": ModelCapabilityProfile(
        tier="small",
        max_elements=25,
        body_chars=220,
        history_turns=2,
        prompt_style="json_schema",
        batch_actions=False,
        system_prompt_max_tokens=160,
        temperature=0.1,
        max_tokens=192,
    ),
    "medium": ModelCapabilityProfile(
        tier="medium",
        max_elements=45,
        body_chars=350,
        history_turns=5,
        prompt_style="json_schema",
        batch_actions=False,
        system_prompt_max_tokens=300,
        temperature=0.1,
        max_tokens=384,
    ),
    "frontier": ModelCapabilityProfile(
        tier="frontier",
        max_elements=75,
        body_chars=600,
        history_turns=10,
        prompt_style="json_schema",
        batch_actions=True,
        system_prompt_max_tokens=800,
        temperature=0.1,
        max_tokens=768,
    ),
}


def detect_detailed_profile(model_name: str, configured: str = "auto") -> ModelCapabilityProfile:
    """Detects granular capability profile (sub1b, small, medium, frontier) from model name."""
    if configured == "small":
        return MODEL_PROFILES["small"]
    if configured == "frontier":
        return MODEL_PROFILES["frontier"]

    name_lower = (model_name or "").lower()

    # Sub-1B explicit markers (e.g. 135m, 360m, 500m, 0.5b, 0.6b, 0.8b)
    if re.search(r"(?<![a-z0-9])(?:135|270|350|360|500|600|700|800)m(?![a-z0-9])", name_lower):
        return MODEL_PROFILES["sub1b"]

    m = re.search(r"(?<![0-9.])(\d+(?:\.\d+)?)\s*b(?![a-z0-9])", name_lower)
    if m:
        try:
            params_b = float(m.group(1))
            if params_b < 1.0:
                return MODEL_PROFILES["sub1b"]
            if params_b <= 4.5:
                return MODEL_PROFILES["small"]
            if params_b <= 15.0:
                return MODEL_PROFILES["medium"]
            return MODEL_PROFILES["frontier"]
        except ValueError:
            pass

    if any(k in name_lower for k in ("smollm", "tiny", "nano", "qwen0.5")):
        return MODEL_PROFILES["sub1b"]
    if any(k in name_lower for k in ("lfm", "mini", "local-model")):
        return MODEL_PROFILES["small"]

    return MODEL_PROFILES["frontier"]


def rank_elements_for_case(elements: list[Any], inputs: dict[str, Any], max_elements: int) -> list[Any]:
    """Ranks DOMElements by relevance to test_case.inputs + primary action buttons for small/sub-1B models."""
    if len(elements) <= max_elements:
        return elements

    input_tokens = set()
    for k in inputs:
        input_tokens.update(re.split(r"[^a-z0-9]+", k.lower()))
    input_tokens.discard("")

    def score(el: Any) -> tuple[int, int]:
        s = 0
        if getattr(el, "region", "") == "dialog":
            s += 50
        if getattr(el, "region", "") == "main":
            s += 20
        text_blob = f"{getattr(el, 'label', '')} {getattr(el, 'name', '')} {getattr(el, 'id', '')} {getattr(el, 'placeholder', '')}".lower()
        for tok in input_tokens:
            if tok in text_blob:
                s += 30
        if getattr(el, "tag", "") == "button" or getattr(el, "type", "") == "submit":
            s += 15
        if getattr(el, "required", False):
            s += 10
        return (-s, getattr(el, "vid", 0))

    ranked = sorted(elements, key=score)
    return ranked[:max_elements]
