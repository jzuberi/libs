
from pydantic import BaseModel
from typing import List, Dict, Any, TypedDict


class NodeMetadata(BaseModel):
    created: str          # ISO8601 string, safe for JSON + Qdrant
    active: bool
    evidence: List[str]

    class Config:
        extra = "allow"   # allow layer-specific metadata fields


class LayerNode(TypedDict):
    id: str
    text: str
    metadata: Dict[str, Any]
