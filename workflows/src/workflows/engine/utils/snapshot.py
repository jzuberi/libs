from __future__ import annotations

import json
from typing import Any, Dict, Optional, TYPE_CHECKING

from ..models import WorkflowItem
from ..models import StepOutputRecord

# Avoid circular import at runtime
if TYPE_CHECKING:
    from ..base_workflow_engine import BaseWorkflowEngine
    



def _serialize_item_summary(engine: BaseWorkflowEngine, item: WorkflowItem, small: bool) -> Dict[str, Any]:
    summary = engine.summarize_item_structured(item, small=small)
    return {
        "id": item.id,
        "type": item.type,
        "description": item.description,
        "status": {
            "branch": item.status.branch,
            "substate": item.status.substate,
            "flags": item.status.flags,
            "approved": item.status.approved,
            "exported_at": item.exported_at.isoformat() if item.exported_at else None,
        },
        "summary": summary,
    }


def _snapshot_tier_full(
    engine: BaseWorkflowEngine,
    focused_item_id: Optional[str],
) -> Dict[str, Any]:
    items = list(engine._items.values())
    focused = engine._items.get(focused_item_id) if focused_item_id else None

    return {
        "engine": {
            "name": engine.__class__.__name__,
            "workflow_paths": engine.definition.workflow_paths,
        },
        "items": [_serialize_item_summary(engine, item, small=False) for item in items],
        "focused_item": _serialize_item_summary(engine, focused, small=False) if focused else None,
        "tier": "full",
    }


def _snapshot_tier_small(
    engine: BaseWorkflowEngine,
    focused_item_id: Optional[str],
) -> Dict[str, Any]:
    items = list(engine._items.values())
    focused = engine._items.get(focused_item_id) if focused_item_id else None

    return {
        "engine": {
            "name": engine.__class__.__name__,
            "workflow_paths": engine.definition.workflow_paths,
        },
        "items": [_serialize_item_summary(engine, item, small=True) for item in items],
        "focused_item": _serialize_item_summary(engine, focused, small=False) if focused else None,
        "tier": "small_items_full_focus",
    }


def _snapshot_tier_minimal(
    engine: BaseWorkflowEngine,
    focused_item_id: Optional[str],
) -> Dict[str, Any]:
    items = list(engine._items.values())
    focused = engine._items.get(focused_item_id) if focused_item_id else None

    return {
        "engine": {
            "name": engine.__class__.__name__,
        },
        "items": [
            {
                "id": item.id,
                "type": item.type,
                "status": {
                    "branch": item.status.branch,
                    "substate": item.status.substate,
                    "approved": item.status.approved,
                },
            }
            for item in items
        ],
        "focused_item": _serialize_item_summary(engine, focused, small=False) if focused else None,
        "tier": "minimal_items_full_focus",
    }


def _snapshot_tier_focus_only(
    engine: BaseWorkflowEngine,
    focused_item_id: Optional[str],
) -> Dict[str, Any]:
    focused = engine._items.get(focused_item_id) if focused_item_id else None

    return {
        "engine": {
            "name": engine.__class__.__name__,
        },
        "items": [],
        "focused_item": _serialize_item_summary(engine, focused, small=False) if focused else None,
        "tier": "focus_only",
    }



def _serialize(obj):
    """
    Recursively convert engine objects into JSON-serializable structures.
    """
    # StepOutputRecord → dict
    if isinstance(obj, StepOutputRecord):
        return obj.model_dump()

    # Pydantic models → dict
    if hasattr(obj, "model_dump"):
        return obj.model_dump()

    # WorkflowItem, WorkflowStatus, etc. → use __dict__ safely
    if hasattr(obj, "__dict__"):
        return {k: _serialize(v) for k, v in obj.__dict__.items()}

    # dict → recurse
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}

    # list/tuple → recurse
    if isinstance(obj, (list, tuple)):
        return [_serialize(v) for v in obj]

    # Path → str
    try:
        from pathlib import Path
        if isinstance(obj, Path):
            return str(obj)
    except:
        pass

    # datetime → isoformat
    try:
        import datetime
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
    except:
        pass

    # fallback: return as-is (must be JSON-serializable)
    return obj


def build_context_snapshot(
    engine,
    focused_item_id: Optional[str] = None,
    max_chars: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Deterministic, JSON-serializable, bounded snapshot of engine state.

    Fallback tiers (in order):
      1. full summaries for all items + focused
      2. small summaries for items + full focused
      3. minimal items + full focused
      4. focused only
    """
    tiers = [
        _snapshot_tier_full,
        _snapshot_tier_small,
        _snapshot_tier_minimal,
        _snapshot_tier_focus_only,
    ]

    if max_chars is None or max_chars <= 0:
        snapshot = tiers[0](engine, focused_item_id)
        return _serialize(snapshot)

    for tier_fn in tiers:
        snapshot = tier_fn(engine, focused_item_id)
        serial = _serialize(snapshot)
        encoded = json.dumps(serial, separators=(",", ":"), ensure_ascii=False)
        if len(encoded) <= max_chars:
            return serial

    # If even the smallest tier exceeds max_chars, return the smallest anyway.
    snapshot = tiers[-1](engine, focused_item_id)
    return _serialize(snapshot)
