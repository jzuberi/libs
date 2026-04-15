from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
import uuid, json
from pathlib import Path

class BaseIdea(BaseModel):
    id: str
    description: str
    type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    style: Dict[str, Any] = Field(default_factory=dict)
    approved: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    exported_at: datetime | None = None


class BaseAsset(BaseModel):
    id: str
    idea_id: str
    type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    style: Dict[str, Any] = Field(default_factory=dict)
    approved: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class BaseFinalOutput(BaseModel):
    id: str
    idea_id: str
    asset_ids: List[str]
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class EngineState(BaseModel):
    branch: str
    substate: Optional[str] = None
    asset_workflow: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    progress: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class IdeaProgressStep(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    branch: str                     # maps directly to EngineState.branch
    substate: str                   # maps directly to EngineState.substate
    action: str                     # created | edited | approved | rejected | regenerated | ...
    actor: str                      # user | agent
    artifact: Optional[Dict[str, Any]] = None
    details: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class IdeaProgress(BaseModel):
    idea_id: str
    steps: List[IdeaProgressStep] = Field(default_factory=list)
