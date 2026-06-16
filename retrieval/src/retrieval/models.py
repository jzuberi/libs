# libs/retrieval/src/retrieval/models.py

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field



class IntentType(str, Enum):
    """
    The type of retrieval task being requested.
    This determines which grading logic the retrieval layer applies.
    """
    TOPIC_FILTER = "topic_filter"
    SUPPORT_STATEMENT = "support_statement"
    TRIPLE_SUPPORT = "triple_support" 
    # Future types can be added here:
    # CONTRADICT_STATEMENT = "contradict_statement"
    # DEFINE_CONCEPT = "define_concept"
    # SUMMARIZE_DOMAIN = "summarize_domain"


class Indicator(BaseModel):
    """
    A one-dimensional semantic constraint.
    Example:
        name="is_loc", value="America", negated=False
        name="is_loc", value="Japan", negated=True
    """
    name: str
    value: str
    negated: bool = False


class RetrievalIntent(BaseModel):
    """
    A structured representation of what the user wants.
    Produced by the LLM library, consumed by the retrieval layer.
    """
    type: IntentType
    guidance: Optional[str] = None
    indicators: List[Indicator] = Field(default_factory=list)
    grading: bool = True
    statement: Optional[str] = None

    # -----------------------------------------
    # NEW: Canonical time constraints (used by retrieval)
    # -----------------------------------------
    date_after: Optional[str] = None     # ISO date string: "YYYY-MM-DD"
    date_before: Optional[str] = None    # ISO date string: "YYYY-MM-DD"

    # -----------------------------------------
    # NEW: Raw time indicators (LLM output only)
    # These will be mapped into date_after/date_before in post-processing.
    # -----------------------------------------
    recency_days: Optional[int] = None
    time_bucket: Optional[str] = None



class GradeResult(BaseModel):
    passed: bool
    score: float | None = None
    reason: str | None = None
    raw: dict | None = None



class ChunkTrace(BaseModel):
    chunk_id: Any
    passed: bool
    grader_passed: Optional[bool] = None
    grader_reason: Optional[str] = None
    indicator_results: Dict[str, bool] = {}
    failure_stage: Optional[str] = None  # "grader", "indicator", None


class RetrievalTrace(BaseModel):
    query: str
    intent: Any
    mode: Optional[str] = None
    chunks: List[ChunkTrace] = []
    total_seen: int = 0
    total_kept: int = 0
