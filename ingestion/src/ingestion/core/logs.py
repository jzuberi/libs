from typing import Any, Dict
from datetime import datetime


def make_log_entry(source_name: str, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a structured log entry for ingestion events.

    Fields:
        - ts: ISO timestamp
        - source: name of the DataSource
        - event: event type (approval_requested, approved, applied, error, etc.)
        - payload: arbitrary event data
    """
    return {
        "ts": datetime.utcnow().isoformat() + "Z",
        "source": source_name,
        "event": event_type,
        "payload": payload or {},
    }
