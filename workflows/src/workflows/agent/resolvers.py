from __future__ import annotations

from typing import Optional, Tuple

from ..engine.base_workflow_engine import BaseWorkflowEngine
from .session import SessionState


def resolve_item_reference(
    engine: BaseWorkflowEngine,
    reference: Optional[str],
    session: SessionState,
) -> Tuple[Optional[str], str]:
    """
    Returns (item_id, message).
    message is a human-readable explanation or error.
    """
    # If no reference, fall back to session
    if not reference and session.last_item_id:
        return session.last_item_id, "Using the last referenced item."

    if not reference:
        item_labels = [item.label for item in engine.list_items()]
        if not item_labels:
            return None, "There are no items yet."
        return None, f"Which item are you referring to? Valid items are: {', '.join(item_labels)}"

    # Try deterministic resolution
    result = engine.resolve_candidates(reference)

    label = engine.get_item_label(result.chosen_id)
    
    if result.chosen_id is None:
        item_labels = [item.label for item in engine.list_items()]
        return None, f"I don’t recognize the item '{reference}'. Valid items are: {', '.join(item_labels)}"

    return result.chosen_id, f"Resolved '{reference}' to item {label}."
