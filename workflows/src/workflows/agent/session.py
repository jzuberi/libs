from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

# Only import WorkflowIntent for type checking, not at runtime
if TYPE_CHECKING:
    from .intent_parser import WorkflowIntent


@dataclass
class SessionState:
    last_item_id: Optional[str] = None
    last_intent: Optional[str] = None

    # Pending-intent support
    pending_intent: Optional["WorkflowIntent"] = None
    pending_resolution: Optional[str] = None
