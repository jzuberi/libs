from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
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
    context_requirements: List[str] = Field(default_factory=list)

    # Optional Pydantic schema for validating step outputs
    output_schema: Optional[type[BaseModel]] = None

    # Optional human-readable metadata for agents and UIs
    description: Optional[str] = None          # What does this step do?
    human_name: Optional[str] = None           # Friendly name for UI/agent
    consumes: Optional[List[str]] = None       # Which steps must run before this?
    produces: Optional[List[str]] = None       # What conceptual outputs does it create?
    agent_hints: Optional[str] = None          # Optional natural-language hints for LLM agents

    child_workflow_name: str | None = None
    kind: str = "function"

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
