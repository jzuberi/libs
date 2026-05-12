from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Literal
import json

from pydantic import BaseModel, Field, validator



# -------------------------------------------------------------------------
# Trace Levels
# -------------------------------------------------------------------------

class TraceLevel(str, Enum):
    INFO = "INFO"
    DEBUG = "DEBUG"
    ERROR = "ERROR"
    AUDIT = "AUDIT"


from typing import List, Union


class HandlerMessage(BaseModel):
    title: Optional[str] = None
    body: Optional[str] = None
    bullets: Optional[List[str]] = None
    footer: Optional[str] = None
    success: bool = True   # optional styling flag

    def render(self) -> str:
        parts = []

        # Optional success prefix
        if self.success:
            parts.append("✅")
        else:
            parts.append("⚠️")

        # Title
        if self.title:
            parts.append(f" **{self.title}**")
        parts.append("")  # newline

        # Body
        if self.body:
            parts.append(self.body)
            parts.append("")

        # Bullets
        if self.bullets:
            for b in self.bullets:
                parts.append(f"- {b}")
            parts.append("")

        # Footer
        if self.footer:
            parts.append(self.footer)

        return "\n".join(parts).strip()


MessageLike = Union[str, "HandlerMessage"]

def merge_messages(messages: List[MessageLike]) -> str:
    """
    Merge a list of strings and/or HandlerMessage objects into a single
    rendered string with clean spacing.
    """
    rendered_parts = []

    for msg in messages:
        if msg is None:
            continue

        if isinstance(msg, HandlerMessage):
            rendered_parts.append(msg.render())
        else:
            rendered_parts.append(str(msg).strip())

    # Join with two newlines for readability
    return "\n\n".join(part for part in rendered_parts if part)


# -------------------------------------------------------------------------
# WorkflowStatusV2
# -------------------------------------------------------------------------
class WorkflowStatus(BaseModel):
    """
    Workflow state for an item.
    Approval is now part of status (not the item root).
    """
    branch: str
    substate: str
    approved: bool = True
    requires_approval: bool = False   # <-- add this
    flags: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        validate_assignment = True

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "WorkflowStatus":
        return WorkflowStatus(**data)

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @staticmethod
    def from_json(data: str) -> "WorkflowStatus":
        return WorkflowStatus.model_validate_json(data)

# -------------------------------------------------------------------------
# Step Output Record
# -------------------------------------------------------------------------

class StepOutputRecord(BaseModel):
    """
    Tracks all artifact files for a step:
    - raw output
    - current (materialized) output
    - edit history
    - optional schema reference
    """
    raw: Any = None
    current: Any = None
    edits: List[str] = Field(default_factory=list)
    schema_name: Optional[str] = None

    class Config:
        validate_assignment = True


# -------------------------------------------------------------------------
# WorkflowItemV2
# -------------------------------------------------------------------------

class WorkflowItem(BaseModel):
    """
    Rich, domain‑agnostic item model.
    Supports:
    - metadata
    - style
    - assets
    - step output registry
    - workflow status
    - provenance
    """

    id: str
    description: str
    type: str

    parent_id: Optional[str] = None

    metadata: Dict[str, Any] = Field(default_factory=dict)
    label: Optional[str] = None
    style: Dict[str, Any] = Field(default_factory=dict)

    initial_input: dict | None = None

    # NEW: asset registry (markdown, images, attachments)
    assets: Dict[str, str] = Field(default_factory=dict)

    # NEW: step output registry
    step_outputs: Dict[str, StepOutputRecord] = Field(default_factory=dict)

    # workflow state
    status: WorkflowStatus = Field(
        default_factory=lambda: WorkflowStatus(branch="default", substate="start")
    )

    # provenance
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    exported_at: Optional[datetime] = None

    created_by: Optional[str] = None
    modified_by: Optional[str] = None

    class Config:
        validate_assignment = True

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @staticmethod
    def from_json(data: str) -> "WorkflowItem":
        return WorkflowItem.model_validate_json(data)


# -------------------------------------------------------------------------
# Step Input / Output
# -------------------------------------------------------------------------

class WorkflowStepInput(BaseModel):
    item: WorkflowItem
    engine: "BaseWorkflowEngine"
    context: Optional[Dict[str, Any]] = None
    step_name: Optional[str] = None
    parsed_input: Any = None  # ✅ NEW

    class Config:
        arbitrary_types_allowed = True



class WorkflowStepOutput(BaseModel):
    artifact: Any
    details: Dict[str, Any]
    summary: str
    next_substate: Optional[str] = None
    requires_approval: bool = False
    approved: Optional[bool] = None

    class Config:
        validate_assignment = True

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "WorkflowStepOutput":
        return WorkflowStepOutput(**data)


WorkflowStepFn = Callable[[WorkflowStepInput], WorkflowStepOutput]
ValidatorFn = Callable[[WorkflowItem, "BaseWorkflowEngine"], None]


# -------------------------------------------------------------------------
# Step Spec (now schema‑aware + agent‑aware)
# -------------------------------------------------------------------------
class WorkflowStepSpec(BaseModel):
    name: str
    fn: WorkflowStepFn

    # NEW: restrict kind to known values
    kind: Literal["function", "workflow", "review", "custom_data_edit"]

    # Optional Pydantic schema for validating step outputs
    output_schema: Optional[type[BaseModel]] = None

    # Optional human-readable metadata for agents and UIs
    description: Optional[str] = None
    human_name: Optional[str] = None
    consumes: Optional[List[str]] = None
    produces: Optional[List[str]] = None
    agent_hints: Optional[str] = None
    allowed_handlers: list[str] = []

    # Workflow / step type
    child_workflow_name: str | None = None

    # Optional input schema
    input_schema: Optional[Type[BaseModel]] = None

    # ⭐ Custom data edit hooks
    model: Optional[type] = None
    renderer: Optional[callable] = None
    interpreter: Optional[callable] = None
    handler: Optional[callable] = None
    validator: Optional[callable] = None

    # ⭐ NEW: context builder hook
    context_builder: Optional[Callable] = None

    class Config:
        arbitrary_types_allowed = True



# -------------------------------------------------------------------------
# Candidate Resolution
# -------------------------------------------------------------------------

class CandidateResolutionResult(BaseModel):
    candidates: List[Tuple[WorkflowItem, float]]
    chosen_id: Optional[str]

    class Config:
        arbitrary_types_allowed = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidates": [(item.id, score) for item, score in self.candidates],
            "chosen_id": self.chosen_id,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any], engine: "BaseWorkflowEngine") -> "CandidateResolutionResult":
        candidates = []
        for item_id, score in data.get("candidates", []):
            item = engine._items.get(item_id)
            if item:
                candidates.append((item, score))

        return CandidateResolutionResult(
            candidates=candidates,
            chosen_id=data.get("chosen_id"),
        )
