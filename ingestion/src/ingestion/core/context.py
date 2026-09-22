from enum import Enum, auto
from typing import Any, Dict, Optional


class ApprovalState(Enum):
    """
    Lifecycle states for an ingestion task.
    """
    INIT = auto()
    PENDING = auto()
    APPROVED = auto()
    REJECTED = auto()
    APPLIED = auto()


class UpdateContext:
    """
    Per-task ingestion context.

    Holds:
        - metadata: dict describing the asset
        - destination_info: dict describing where/how to place the asset
        - approval_state: current lifecycle state
        - error: optional error message

    This context is created fresh for each task in DataSourceBase._run_task()
    and cleared after finalize().
    """

    def __init__(self):
        self.metadata: Dict[str, Any] = {}
        self.destination_info: Dict[str, Any] = {}
        self.approval_state: ApprovalState = ApprovalState.INIT
        self.error: Optional[str] = None

    def transition(self, new_state: ApprovalState):
        """
        Transition to a new approval state.
        """
        self.approval_state = new_state

    def clear(self):
        """
        Reset the context after a task is fully processed.
        """
        self.metadata = {}
        self.destination_info = {}
        self.approval_state = ApprovalState.INIT
        self.error = None
