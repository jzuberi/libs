from typing import Dict, Any, Literal, TypedDict, Optional


# ------------------------------------------------------------
# STATUS ENUM
# ------------------------------------------------------------
TaskStatus = Literal[
    "pending",   # requires approval
    "auto",      # auto-approved (no manual approval)
    "exists",    # already ingested → skip
    "error",     # failed before ingestion → skip
]


# ------------------------------------------------------------
# DESTINATION INFO
# ------------------------------------------------------------
class DestinationInfo(TypedDict):
    """
    Information needed by the destination to place the file.
    Model B: source_path is where the file already exists.
    """
    source_path: str
    subfolder: str
    filename: str


# ------------------------------------------------------------
# INGESTION TASK
# ------------------------------------------------------------
class IngestionTask(TypedDict, total=False):
    """
    A single ingestion unit handled by the base class.
    Each task goes through its own lifecycle:
        - approval (pending or auto)
        - apply
        - finalize
    """
    status: TaskStatus
    metadata: Dict[str, Any]
    destination_info: DestinationInfo
    error: Optional[str]  # only meaningful when status == "error"
