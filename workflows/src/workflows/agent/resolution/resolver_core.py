from __future__ import annotations

from typing import Optional, Tuple

from ...engine.base_workflow_engine import BaseWorkflowEngine
from ..session import SessionState

def resolve_item_reference(
    engine: BaseWorkflowEngine,
    reference: Optional[str],
    session: SessionState,
) -> Tuple[Optional[str], str]:
    """
    Resolve an item reference using:
    1. last_item_id fallback
    2. ordinal resolution
    3. short-id resolution
    4. exact label match
    5. timestamp match
    6. engine.resolve_candidates (deterministic)
    7. LLM fallback (guarded)
    """

    # ------------------------------------------------------------
    # 0. No reference → fallback to last_item_id
    # ------------------------------------------------------------
    if not reference:
        if session.last_item_id:
            return session.last_item_id, "Using the last referenced item."
        labels = [item.label for item in engine.list_items()]
        if not labels:
            return None, "There are no items yet."
        return None, f"Which item are you referring to? Valid items are: {', '.join(labels)}"

    ref = reference.lower().strip()
    items = engine.list_items()

    # ------------------------------------------------------------
    # 1. Ordinal resolution (first, second, third, last)
    # ------------------------------------------------------------
    if session.last_listed_items:
        ordinals = {
            "first": 0, "1st": 0,
            "second": 1, "2nd": 1,
            "third": 2, "3rd": 2,
            "last": len(session.last_listed_items) - 1,
        }
        for word, idx in ordinals.items():
            if word in ref and 0 <= idx < len(session.last_listed_items):
                chosen = session.last_listed_items[idx]
                label = engine.get_item_label(chosen)
                return chosen, f"Resolved ordinal '{reference}' to item {label}."

    # ------------------------------------------------------------
    # 2. Short-ID resolution
    # ------------------------------------------------------------
    if session.last_listed_items:
        for item_id in session.last_listed_items:
            short = item_id[:6].lower()
            if short in ref or ref.endswith(short):
                label = engine.get_item_label(item_id)
                return item_id, f"Resolved short-id '{reference}' to item {label}."

    # ------------------------------------------------------------
    # 3. Exact label match
    # ------------------------------------------------------------
    for item in items:
        if reference.lower() == item.label.lower():
            return item.id, f"Resolved by exact label match: {item.label}"

    # ------------------------------------------------------------
    # 4. Timestamp match (if label contains a timestamp)
    # ------------------------------------------------------------
    for item in items:
        if item.label.lower().endswith(ref) or ref in item.label.lower():
            return item.id, f"Resolved by timestamp/label match: {item.label}"

    # ------------------------------------------------------------
    # 5. Engine's deterministic resolver
    # ------------------------------------------------------------
    result = engine.resolve_candidates(reference)
    if result.chosen_id:
        label = engine.get_item_label(result.chosen_id)
        return result.chosen_id, f"Resolved '{reference}' to item {label}."

    # ------------------------------------------------------------
    # 6. LLM fallback (guarded)
    # ------------------------------------------------------------
    if hasattr(engine, "agent_llm") and engine.agent_llm:
        labels = [item.label for item in items]
        prompt = f"""
        You are resolving a user reference to an item.

        User reference: "{reference}"

        Valid items:
        {labels}

        Respond ONLY with JSON:
        {{
        "item_id": "<one of these valid item IDs>",
        "confidence": <0 to 1>
        }}
        """

        print('resolver prompt:')
        print(prompt)
        llm_result = engine.agent_llm.metadata(prompt).dict()
        print(llm_result)

        if isinstance(llm_result, dict):
            candidate = llm_result.get("item_id")
            if candidate in [item.id for item in items]:
                label = engine.get_item_label(candidate)
                return candidate, f"Resolved via LLM fallback to item {label}."

    # ------------------------------------------------------------
    # 7. Total failure
    # ------------------------------------------------------------
    labels = [item.label for item in items]
    return None, f"I don’t recognize the item '{reference}'. Valid items are: {', '.join(labels)}"
