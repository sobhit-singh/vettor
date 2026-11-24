import re
from typing import Any, Optional


class AssertionResult:
    def __init__(self, passed: bool, message: str = "", actual: Any = None, expected: Any = None):
        self.passed = passed
        self.message = message
        self.actual = actual
        self.expected = expected

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "message": self.message,
            "actual": self.actual,
            "expected": self.expected,
        }


class QAAssertions:
    """Configurable assertion engine for Web QA validation."""

    @staticmethod
    def evaluate_case_assertions(row: dict[str, Any], page_state: dict[str, Any], extracted: dict[str, Any], float_tolerance: float = 0.01) -> list[AssertionResult]:
        results: list[AssertionResult] = []
        body_text = page_state.get("bodyText", "")
        current_url = page_state.get("url", "")

        # 1. Check for expected numeric premium / total / price
        for num_col in ["expected_premium", "expected_total", "expected_price", "expected_amount", "expected_quote"]:
            if num_col in row and row[num_col] is not None and str(row[num_col]).strip() != "":
                raw_exp = str(row[num_col])
                clean_exp = re.sub(r"[^\d.]", "", raw_exp)
                try:
                    exp_val = float(clean_exp)
                except ValueError:
                    exp_val = 0.0

                target_key = num_col.replace("expected_", "")
                actual_val = extracted.get(
                    target_key,
                    extracted.get("premium", extracted.get("total", extracted.get("amount", 0.0))),
                )
                try:
                    act_float = float(actual_val)
                except (ValueError, TypeError):
                    act_float = 0.0

                is_pass = abs(act_float - exp_val) <= float_tolerance
                msg = f"Numeric match: actual={act_float} vs expected={exp_val}" if is_pass else f"Value mismatch: actual={act_float} != expected={exp_val} (diff={round(abs(act_float - exp_val), 4)})"
                results.append(AssertionResult(passed=is_pass, message=msg, actual=act_float, expected=exp_val))

        # 2. Check for expected text containment
        for text_col in ["expected_text", "expected_message", "expected_contains"]:
            if text_col in row and row[text_col]:
                exp_text = str(row[text_col]).strip()
                is_pass = exp_text.lower() in body_text.lower()
                msg = f"Text found: '{exp_text}'" if is_pass else f"Missing expected text: '{exp_text}'"
                results.append(AssertionResult(passed=is_pass, message=msg, actual="[Text present]" if is_pass else "[Text absent]", expected=exp_text))

        # 3. Check for expected URL / path
        if "expected_url" in row and row["expected_url"]:
            exp_url = str(row["expected_url"]).strip()
            is_pass = exp_url.lower() in current_url.lower()
            msg = f"URL matched: '{current_url}' contains '{exp_url}'" if is_pass else f"URL mismatch: '{current_url}' does not contain '{exp_url}'"
            results.append(AssertionResult(passed=is_pass, message=msg, actual=current_url, expected=exp_url))

        # 4. Check for expected reference / confirmation code
        if "expected_ref" in row and row["expected_ref"]:
            exp_ref = str(row["expected_ref"]).strip()
            act_ref = extracted.get("quote_id", extracted.get("order_id", extracted.get("ref_id", "")))
            is_pass = exp_ref.lower() in str(act_ref).lower() or exp_ref.lower() in body_text.lower()
            msg = f"Reference ID matched: '{act_ref}'" if is_pass else f"Reference ID mismatch: actual='{act_ref}' expected='{exp_ref}'"
            results.append(AssertionResult(passed=is_pass, message=msg, actual=act_ref, expected=exp_ref))

        # 5. Dynamic domain-agnostic assertion matching for ANY other expected_* columns
        handled_keys = {
            "expected_premium", "expected_total", "expected_price", "expected_amount", "expected_quote",
            "expected_text", "expected_message", "expected_contains", "expected_url", "expected_ref"
        }
        for col, exp_raw in row.items():
            if not col.startswith("expected_") or col in handled_keys or exp_raw is None or str(exp_raw).strip() == "":
                continue

            target_key = col.replace("expected_", "").strip()
            exp_str = str(exp_raw).strip()

            # Attempt numeric match first
            clean_num = re.sub(r"[^\d.]", "", exp_str)
            if clean_num and clean_num.replace(".", "", 1).isdigit():
                try:
                    exp_num = float(clean_num)
                    act_val = extracted.get(target_key, extracted.get(target_key.lower(), 0.0))
                    clean_act = re.sub(r"[^\d.]", "", str(act_val))
                    act_num = float(clean_act) if clean_act and clean_act.replace(".", "", 1).isdigit() else 0.0
                    is_pass = abs(act_num - exp_num) <= float_tolerance
                    msg = f"Metric match [{target_key}]: actual={act_num} vs expected={exp_num}" if is_pass else f"Metric mismatch [{target_key}]: actual={act_num} != expected={exp_num}"
                    results.append(AssertionResult(passed=is_pass, message=msg, actual=act_num, expected=exp_num))
                    continue
                except Exception:
                    pass

            # String / containment match
            act_val = str(extracted.get(target_key, extracted.get(target_key.lower(), ""))).strip()
            is_pass = (exp_str.lower() in act_val.lower()) or (exp_str.lower() in body_text.lower())
            msg = f"Match [{target_key}]: found '{exp_str}'" if is_pass else f"Mismatch [{target_key}]: expected '{exp_str}', found '{act_val}'"
            results.append(AssertionResult(passed=is_pass, message=msg, actual=act_val or "[page text]", expected=exp_str))

        # 6. Default fallback if no explicit assertion columns exist
        if not results:
            # Check if any errors are on page
            errors = page_state.get("errors", [])
            has_errors = bool(errors)
            msg = f"Page completed with errors: {errors}" if has_errors else "Page completed without uncaught validation blockers."
            results.append(AssertionResult(passed=not has_errors, message=msg, actual=errors, expected="no errors"))

        return results
