from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ActionType = Literal[
    "fill",
    "select",
    "check",
    "uncheck",
    "click",
    "press",
    "wait",
    "assert",
    "done",
    "stuck",
]


class AgentAction(BaseModel):
    """Flat action schema optimized for 1.7B small models and GBNF/json_schema decoding."""

    action: ActionType
    vid: int | None = None
    value: str = ""
    reason: str = ""


class ActionBatch(BaseModel):
    """Batch of actions used by frontier models that can emit multi-step plans."""

    actions: list[AgentAction] = Field(default_factory=list)


class DOMElement(BaseModel):
    """Interactive DOM element annotated with an integer visual ID (vid)."""

    vid: int
    tag: str
    type: str = ""
    label: str = ""
    name: str = ""
    id: str = ""
    selector: str = ""
    required: bool = False
    value: str = ""
    checked: bool = False
    options: list[str] = Field(default_factory=list)


class PageObservation(BaseModel):
    """Snapshot of current browser page state, interactive elements, and diagnostics."""

    url: str
    pathname: str = ""
    title: str = ""
    elements: list[DOMElement] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    dialogs: list[dict[str, Any]] = Field(default_factory=list)
    body_text: str = ""

    def to_compact_text(self, max_elements: int = 50) -> str:
        """Renders a token-minimal text view for small LLMs.

        Example:
            URL: /step3 | Title: Health
            [1] select "Do you smoke?" options=[no, yes] val="no"
            [2] input(number) "Age" req val=""
            [3] button "Next Step"
            Errors: none
        """
        path_or_url = self.pathname if self.pathname else self.url
        lines = [f"URL: {path_or_url} | Title: {self.title}"]

        for el in self.elements[:max_elements]:
            # Element tag / input type representation
            tag_lower = el.tag.lower()
            if tag_lower == "input" and el.type:
                tag_str = f"input({el.type})"
            else:
                tag_str = tag_lower

            # Friendly descriptor
            name_desc = el.label or el.name or el.id or el.value or el.selector
            parts = [f"[{el.vid}] {tag_str} \"{name_desc}\""]

            if el.required:
                parts.append("req")

            if el.options:
                opt_str = ", ".join(el.options)
                parts.append(f"options=[{opt_str}]")

            if el.value and tag_lower != "button":
                parts.append(f'val="{el.value}"')
            elif tag_lower in ("input", "select", "textarea"):
                parts.append('val=""')

            if el.checked:
                parts.append("checked")

            lines.append(" ".join(parts))

        if self.errors:
            lines.append(f"Errors: {'; '.join(self.errors)}")
        else:
            lines.append("Errors: none")

        return "\n".join(lines)

    def signature(self) -> tuple[str, tuple[tuple[str, str, str], ...]]:
        """Stable structural signature of the page for caching and loop detection."""
        return (self.pathname, tuple((e.selector, e.tag, e.type) for e in self.elements))


class TestCase(BaseModel):
    """Specification of a test case with inputs, expected outcomes, and goals."""

    __test__ = False

    case_id: str
    name: str = ""
    inputs: dict[str, Any] = Field(default_factory=dict)
    expected: dict[str, Any] = Field(default_factory=dict)
    goal: str = ""
    tags: list[str] = Field(default_factory=list)

    @classmethod
    def from_flat_row(cls, row: dict[str, Any], default_id: str = "1") -> TestCase:
        """Splits expected_* keys into expected (stripping the prefix), case identifiers, and inputs."""
        case_id = default_id
        name = ""
        goal = ""
        tags: list[str] = []
        inputs: dict[str, Any] = {}
        expected: dict[str, Any] = {}

        for k, v in row.items():
            if k in ("case_id", "id"):
                case_id = str(v)
            elif k == "name":
                name = str(v)
            elif k == "goal":
                goal = str(v)
            elif k == "tags":
                if isinstance(v, list):
                    tags = [str(item) for item in v]
                elif isinstance(v, str):
                    tags = [t.strip() for t in v.split(",") if t.strip()]
                else:
                    tags = [str(v)]
            elif k.startswith("expected_"):
                expected_key = k[len("expected_"):]
                expected[expected_key] = v
            else:
                inputs[k] = v

        return cls(
            case_id=case_id,
            name=name,
            inputs=inputs,
            expected=expected,
            goal=goal,
            tags=tags,
        )


class Finding(BaseModel):
    """A detected bug, visual glitch, UX friction, or accessibility finding."""

    kind: str
    severity: Literal["P0", "P1", "P2", "P3"] = "P2"
    page: str = ""
    detail: str
    element_vid: int | None = None
    suggested_fix: str = ""
    source: str = "probe"


class AssertionResult(BaseModel):
    """Result of an assertion evaluation against extracted page state."""

    passed: bool
    message: str = ""
    actual: Any = None
    expected: Any = None
    key: str = ""


class CaseResult(BaseModel):
    """Outcome and execution telemetry of a single TestCase."""

    case_id: str
    status: Literal["PASS", "FAIL", "ERROR"]
    duration_s: float = 0.0
    steps: int = 0
    cache_hits: int = 0
    heals: int = 0
    note: str = ""
    extracted: dict[str, Any] = Field(default_factory=dict)
    assertions: list[AssertionResult] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    screenshot: str | None = None
