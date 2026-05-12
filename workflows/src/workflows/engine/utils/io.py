from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from ..models import WorkflowItem, WorkflowStatus


class EngineStorageMixin:
    """
    Mixin providing per-item JSON persistence:
      - item.json (canonical item, full Pydantic model)
      - status.json (mutable workflow status)
      - progress.jsonl (append-only ledger)
    """

    base_dir: Path  # subclasses or engine initializer must set this

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------
    def _item_dir(self, item_id: str) -> Path:

        # Use in-memory registry only
        if item_id not in self._items:
            print(f"[ITEM_DIR] ERROR: item_id {item_id!r} not in self._items")
            raise KeyError(f"Item {item_id} not found in engine._items")

        item = self._items[item_id]

        # ROOT ITEM (no parent)
        if not item.parent_id:
            d = self.base_dir / 'items' / item_id
            d.mkdir(parents=True, exist_ok=True)
            return d

        # CHILD ITEM (recursive)
        parent_dir = self._item_dir(item.parent_id)

        d = parent_dir / "derived_items" / item_id
        d.mkdir(parents=True, exist_ok=True)
        return d


    # ------------------------------------------------------------------
    # Status persistence (Pydantic-aware)
    # ------------------------------------------------------------------

    def _write_status(self, item: WorkflowItem) -> None:
        """
        Persist only the workflow status portion of the item.
        Approval now lives inside item.status.approved.
        """
        status_path = self._item_dir(item.id) / "status.json"

        data = {
            "branch": item.status.branch,
            "substate": item.status.substate,
            "approved": item.status.approved,
            "requires_approval": item.status.requires_approval,
            "flags": item.status.flags,
            "updated_at": item.updated_at.isoformat(),
            "exported_at": item.exported_at.isoformat() if item.exported_at else None,
        }

        status_path.write_text(json.dumps(data, indent=2))

    def _read_status(self, item_id: str) -> WorkflowStatus:
        """
        Load workflow status from status.json.
        """
        status_path = self._item_dir(item_id) / "status.json"
        if not status_path.exists():
            raise FileNotFoundError(f"No status.json found for item {item_id}")

        data = json.loads(status_path.read_text())
        return WorkflowStatus(**data)

    # ------------------------------------------------------------------
    # Progress ledger (append-only)
    # ------------------------------------------------------------------

    def _append_progress(self, item_id: str, entry: Dict[str, Any]) -> None:
        progress_path = self._item_dir(item_id) / "progress.jsonl"
        with progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    # ------------------------------------------------------------------
    # Canonical item persistence (Pydantic-native)
    # ------------------------------------------------------------------

    def _write_item(self, item: WorkflowItem) -> None:
        """
        Persist the full WorkflowItem using Pydantic's JSON serialization.
        """
        item_path = self._item_dir(item.id) / "item.json"

        item_path.write_text(item.model_dump_json(indent=2))

    def _read_item(self, item_id: str) -> WorkflowItem:
        """
        Load the full WorkflowItem using Pydantic's JSON validation.
        """
        item_path = self._item_dir(item_id) / "item.json"
        return WorkflowItem.model_validate_json(item_path.read_text())
