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
    tol = test_case.tolerance if test_case.tolerance is not None else float_tolerance

    # Negative / expect_error oracle
    if test_case.negative or test_case.expect_error:
        combined_errors = " | ".join(obs.errors)
        if test_case.expect_error:
            exp_err = test_case.expect_error.strip().lower()
            matched = (exp_err in combined_errors.lower()) or (exp_err in (obs.regions.get("alert", "") or obs.body_text).lower())
            results.append(
                AssertionResult(
                    key="expect_error",
                    passed=matched,
                    message=(
                        f"Expected validation/error '{test_case.expect_error}' found"
                        if matched
                        else f"Expected error '{test_case.expect_error}' not found (page errors: {combined_errors or 'none'})"
                    ),
                    actual=combined_errors or obs.body_text[:120],
                    expected=test_case.expect_error,
                    region="alert" if combined_errors else "body",
                )
            )
        else:
            has_err = bool(obs.errors)
            results.append(
                AssertionResult(
                    key="negative",
                    passed=has_err,
                    message=(
                        f"Negative test rejected input as expected: {combined_errors}"
                        if has_err
                        else "Negative test failed: no validation/error message was raised"
                    ),
                    actual=obs.errors,
                    expected="validation error",
                    region="alert",
                )
            )
        if not test_case.expected:
            return results

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

        # 2. Text / Message / Contains substring match against main region or body text
        if k_lower in ("text", "message", "contains"):
            main_region = obs.regions.get("main", "")
            search_space = main_region if main_region else obs.body_text
            matched = exp_str.lower() in search_space.lower()
            body_only_match = False
            if not matched and main_region and exp_str.lower() in obs.body_text.lower():
                matched = True
                body_only_match = True
            results.append(
                AssertionResult(
                    key=key,
                    passed=matched,
                    message=f"Body text match for '{key}': expected '{exp_str}'",
                    actual=search_space[:120],
                    expected=expected_val,
                    region="body" if body_only_match else ("main" if main_region else "body"),
                    warn=body_only_match,
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
                is_pass = diff <= tol
                results.append(
                    AssertionResult(
                        key=key,
                        passed=is_pass,
                        message=f"Numeric assertion on '{key}': actual {act_float} vs expected {exp_float} (diff: {diff:.4f}, tol: {tol})",
                        actual=act_float,
                        expected=exp_float,
                    )
                )
            else:
                results.append(
                    AssertionResult(
                        key=key,
                        passed=False,
                        message=f"Numeric assertion on '{key}' failed: not found or non-numeric actual value '{raw_actual}'",
                        actual=raw_actual,
                        expected=exp_float,
                    )
                )
            continue

        # 4. Scoped string containment against extracted data (B-22 fix: avoid false pass from unrelated footer text)
        actual_val = extracted.get(key)
        if actual_val is not None and str(actual_val).strip():
            matched = exp_str.lower() in str(actual_val).lower()
            results.append(
                AssertionResult(
                    key=key,
                    passed=matched,
                    message=f"Extracted field '{key}' match: expected '{exp_str}', got '{actual_val}'",
                    actual=str(actual_val),
                    expected=expected_val,
                    region="extracted",
                )
            )
        else:
            # Check main region if available, otherwise fail (never match unrelated footer/nav)
            main_text = obs.regions.get("main", "")
            matched_main = bool(main_text) and (exp_str.lower() in main_text.lower())
            results.append(
                AssertionResult(
                    key=key,
                    passed=matched_main,
                    message=(
                        f"Scoped main-region match for '{key}': expected '{exp_str}'"
                        if matched_main
                        else f"Field '{key}' not found in extracted values or main region (expected '{exp_str}')"
                    ),
                    actual=main_text[:80] if matched_main else None,
                    expected=expected_val,
                    region="main" if matched_main else "missing",
                )
            )

    return results
