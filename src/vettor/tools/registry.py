from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Any, Literal

from vettor.core.schema import ActionType, AgentAction

PermissionMode = Literal["read_only", "safe", "full"]


@dataclass(frozen=True)
class ToolSpec:
    """Single source of truth for a Vettor browser/QA action tool."""

    name: ActionType
    description: str
    requires_vid: bool = False
    takes_value: bool = False
    mutating: bool = True
    destructive: bool = False


TOOL_REGISTRY: dict[ActionType, ToolSpec] = {
    "fill": ToolSpec("fill", "Type text into an input or textarea by vid.", requires_vid=True, takes_value=True),
    "select": ToolSpec("select", "Select an option in a <select> or combobox by vid.", requires_vid=True, takes_value=True),
    "check": ToolSpec("check", "Check a checkbox or radio button by vid.", requires_vid=True),
    "uncheck": ToolSpec("uncheck", "Uncheck a checkbox by vid.", requires_vid=True),
    "click": ToolSpec("click", "Click a button, link, tab, or interactive element by vid.", requires_vid=True, destructive=True),
    "press": ToolSpec("press", "Press a keyboard key (e.g. Enter, Tab, Escape) on vid.", requires_vid=True, takes_value=True),
    "scroll": ToolSpec("scroll", "Scroll viewport 'down' or 'up'.", requires_vid=False, takes_value=True, mutating=False),
    "goto": ToolSpec("goto", "Navigate to a URL.", requires_vid=False, takes_value=True, mutating=False),
    "wait": ToolSpec("wait", "Wait for milliseconds (default 500).", requires_vid=False, takes_value=True, mutating=False),
    "extract": ToolSpec("extract", "Extract visible text from element vid or page.", requires_vid=False, takes_value=True, mutating=False),
    "assert_oracle": ToolSpec("assert_oracle", "Verify an expected condition on the current page.", requires_vid=False, takes_value=True, mutating=False),
    "report_bug": ToolSpec("report_bug", "Log a detected application defect with description in value.", requires_vid=False, takes_value=True, mutating=False),
    "done": ToolSpec("done", "Signal workflow completion to trigger oracle verification.", requires_vid=False, mutating=False),
    "stuck": ToolSpec("stuck", "Signal blocker when unable to proceed.", requires_vid=False, takes_value=True, mutating=False),
}


def is_action_allowed(
    action: AgentAction,
    permission_mode: PermissionMode = "full",
    element_label: str = "",
) -> tuple[bool, str]:
    """Enforces permission modes (read_only, safe, full) and destructive-action guard."""
    spec = TOOL_REGISTRY.get(action.action)
    if spec is None:
        return False, f"Unknown tool/action: {action.action}"

    if permission_mode == "read_only" and spec.mutating:
        return False, f"Permission mode 'read_only' blocks mutating action '{action.action}'"

    if permission_mode == "safe" and action.action == "click":
        lbl = (element_label or "").lower()
        destructive_words = ("delete", "remove", "destroy", "drop", "submit", "pay", "purchase", "confirm order", "checkout")
        if any(w in lbl for w in destructive_words):
            return False, f"Permission mode 'safe' blocks potentially destructive click on '{element_label}'"

    return True, ""


def build_constrained_action_schema(valid_vids: list[int], include_reason: bool = True) -> dict[str, Any]:
    """Generates a tight JSON schema with valid vid enum injected for constrained decoding."""
    vid_schema: dict[str, Any]
    if valid_vids:
        vid_schema = {"anyOf": [{"type": "integer", "enum": valid_vids}, {"type": "null"}]}
    else:
        vid_schema = {"type": "null"}

    props: dict[str, Any] = {
        "action": {"type": "string", "enum": list(TOOL_REGISTRY.keys())},
        "vid": vid_schema,
        "value": {"type": "string"},
    }
    required = ["action", "vid", "value"]
    if include_reason:
        props["reason"] = {"type": "string"}
        required.append("reason")

    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


def build_gbnf_grammar(valid_vids: list[int]) -> str:
    """Generates a compact GBNF grammar for llama.cpp constrained decoding of AgentAction."""
    actions_rule = " | ".join(f'"\\"{name}\\""' for name in TOOL_REGISTRY)
    if valid_vids:
        vids_rule = " | ".join(f'"{v}"' for v in valid_vids) + ' | "null"'
    else:
        vids_rule = '"null"'

    return (
        'root ::= "{" ws "\\"action\\":" ws action "," ws "\\"vid\\":" ws vid "," ws "\\"value\\":" ws string ws "}"\n'
        f"action ::= {actions_rule}\n"
        f"vid ::= {vids_rule}\n"
        'string ::= "\\"" ([^"\\\\\\n] | "\\\\" ["\\\\/bfnrt])* "\\""\n'
        'ws ::= [ \\t\\n\\r]*\n'
    )


def parse_line_action(raw: str) -> AgentAction | None:
    """Parses a compact line-format action (e.g. `fill 3 "Aarav"`, `click 5`, `done`) for sub-1B/small SLMs."""
    text = (raw or "").strip()
    if not text:
        return None
    # Strip markdown fences if any
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text).strip()
    text = re.sub(r"\s*```$", "", text).strip()
    first_line = text.splitlines()[0].strip()

    try:
        tokens = shlex.split(first_line)
    except ValueError:
        tokens = first_line.split()

    if not tokens:
        return None

    act_name = tokens[0].lower()
    if act_name not in TOOL_REGISTRY:
        return None

    spec = TOOL_REGISTRY[act_name]  # type: ignore[index]
    vid: int | None = None
    value = ""

    if len(tokens) >= 2:
        if tokens[1].isdigit():
            vid = int(tokens[1])
            if len(tokens) >= 3:
                value = " ".join(tokens[2:])
        else:
            value = " ".join(tokens[1:])

    if spec.requires_vid and vid is None:
        return None

    return AgentAction(action=act_name, vid=vid, value=value, reason="line-format")  # type: ignore[arg-type]


def format_tool_help(compact: bool = False) -> str:
    """Renders tool descriptions from the registry for system prompts."""
    if compact:
        return ", ".join(TOOL_REGISTRY.keys())
    return "\n".join(f"- {spec.name}: {spec.description}" for spec in TOOL_REGISTRY.values())
