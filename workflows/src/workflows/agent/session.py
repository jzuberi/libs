from typing import Optional, TYPE_CHECKING
from dataclasses import dataclass, field

@dataclass
class PendingIntent:
    intent: str
    parameters: dict
    missing: list[str]
    original_message: str


@dataclass
class SessionState:
    last_item_id: Optional[str] = None
    last_intent: Optional[str] = None
    context: dict = field(default_factory=dict)

    pending_intent: Optional[PendingIntent] = None   # NEW
    last_listed_items: list[str] = field(default_factory=list)
    active_item_id: Optional[str] = None
