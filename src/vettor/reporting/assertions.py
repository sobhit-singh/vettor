from __future__ import annotations

import re
from typing import Any

from vettor.core.schema import AssertionResult, PageObservation, TestCase


def _try_parse_float(val: Any) -> float | None:
    """Attempts to parse a float after stripping whitespace, currency symbols, and commas."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if not isinstance(val, str):
        return None
    val_str = val.strip()
    cleaned = re.sub(r"[$€£¥, ]", "", val_str)
    if re.fullmatch(r"[-+]?\d+(\.\d+)?", cleaned):
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def evaluate_assertions(
    test_case: TestCase,
    obs: PageObservation,
    extracted: dict[str, Any],
    float_tolerance: float = 0.01,
) -> list[AssertionResult]:
    """Domain-agnostic assertion engine evaluating expected outcomes against page state and extracted data."""
    results: list[AssertionResult] = []

    # If no explicit expectations, pass if no page-level errors exist
    if not test_case.expected:
        if obs.errors:
            results.append(
                AssertionResult(
                    passed=False,
                    message=f"Page validation errors present: {'; '.join(obs.errors)}",
                    actual=obs.errors,
                    expected=[],
                )
            )
        else:
            results.append(
                AssertionResult(
                    passed=True,
                    message="Workflow completed without page validation errors",
                    actual="No errors",
                    expected="No errors",
                )
            )
        return results

    for key, expected_val in test_case.expected.items():
        k_lower = key.lower()
        exp_str = str(expected_val).strip()

        # 1. URL substring match
        if k_lower == "url":
            actual_url = obs.url.lower()
            matched = exp_str.lower() in actual_url
            results.append(
                AssertionResult(
                    key=key,
                    passed=matched,
                    message=f"URL match for '{key}': expected '{exp_str}' in '{obs.url}'",
                    actual=obs.url,
                    expected=expected_val,
                )
            )
            continue

        # 2. Text / Message / Contains substring match against body text
        if k_lower in ("text", "message", "contains"):
            body_lower = obs.body_text.lower()
            matched = exp_str.lower() in body_lower
            results.append(
                AssertionResult(
                    key=key,
                    passed=matched,
                    message=f"Body text match for '{key}': expected '{exp_str}'",
                    actual=obs.body_text[:120],
                    expected=expected_val,
                )
            )
            continue

        # 3. Numeric comparison within tolerance
        exp_float = _try_parse_float(expected_val)
        if exp_float is not None:
            raw_actual = extracted.get(key)
            act_float = _try_parse_float(raw_actual)

            # Single-metric fallback if key not found directly in extracted
            if act_float is None and len(extracted) == 1:
                single_val = next(iter(extracted.values()))
                act_float = _try_parse_float(single_val)

            if act_float is not None:
                diff = abs(act_float - exp_float)
                is_pass = diff <= float_tolerance
                results.append(
                    AssertionResult(
                        key=key,
                        passed=is_pass,
                        message=f"Numeric assertion on '{key}': actual {act_float} vs expected {exp_float} (diff: {diff:.4f}, tol: {float_tolerance})",
                        actual=act_float,
                        expected=exp_float,
                    )
                )
            else:
                results.append(
                    AssertionResult(
                        key=key,
                        passed=False,
                        message=f"Numeric assertion on '{key}' failed: could not parse numeric actual value from extracted '{raw_actual}'",
                        actual=raw_actual,
                        expected=exp_float,
                    )
                )
            continue

        # 4. General string containment against extracted data or page body
        actual_val = str(extracted.get(key, ""))
        matched = (exp_str.lower() in actual_val.lower()) or (exp_str.lower() in obs.body_text.lower())
        results.append(
            AssertionResult(
                key=key,
                passed=matched,
                message=f"String containment on '{key}': expected '{exp_str}' in actual '{actual_val}' or page body",
                actual=actual_val or obs.body_text[:80],
                expected=expected_val,
            )
        )

    return results
