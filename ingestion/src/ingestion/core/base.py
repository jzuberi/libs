from typing import Any, Dict, Optional, List

from .context import ApprovalState, UpdateContext
from .logs import make_log_entry
from .types import IngestionTask
from pathlib import Path
from file_ops import FileOps, jsonable

home_dir = './'
path_home_dir = Path(home_dir)
fs = FileOps(path_home_dir)

import hashlib

def dense16(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


class DataSourceBase:
    """
    Base class for ingestion sources using Model B.

    Enhancements:
    - fetch() must be PURE (no logs, no writes)
    - unified log buffer (subclass defines schema)
    - ingest_log() writes all logs once
    - retrieve_logs() validates canonical fields
    """

    def __init__(
        self,
        name: str,
        log_path: str,                     # <-- still required
        config: Optional[Dict[str, Any]] = None,
    ):
        if not log_path:
            raise ValueError("log_path is required for DataSourceBase")

        self.name = name
        self.log_path = log_path
        self.config = config or {}
        self.destination = self.config.get("destination")

        # Unified log buffer (lifecycle + subclass logs)
        self._logs: List[Dict[str, Any]] = []

    # ------------------------------------------------------------
    # FETCH PHASE (must be PURE)
    # ------------------------------------------------------------
    def fetch(self, **kwargs) -> List[IngestionTask]:
        """
        Subclasses must override this.
        IMPORTANT: fetch() must be PURE — no logging, no file writes.
        """
        raise NotImplementedError

    # ------------------------------------------------------------
    # MULTI-TASK ORCHESTRATOR
    # ------------------------------------------------------------
    def run(self, **fetch_kwargs) -> List[dict]:
        before_logs = len(self._logs)
        tasks = self.fetch(**fetch_kwargs)

        # Backward compatible: detect if subclass wrote logs inside fetch()
        if len(self._logs) != before_logs:
            self._append_log(make_log_entry(
                self.name,
                "warning_fetch_side_effect",
                {"message": "fetch() wrote logs; this is discouraged"}
            ))

        if isinstance(tasks, dict):
            tasks = [tasks]

        results = []
        for task in tasks:
            # Soft validation (backward compatible)
            self._validate_task_schema(task)

            result = self._run_task(task)
            results.append(result)

        return results

    # ------------------------------------------------------------
    # Soft ingestion task schema validation
    # ------------------------------------------------------------
    def _validate_task_schema(self, task: dict):
        """
        Soft validation — warns but does not fail.
        Required fields:
            - status
            - metadata
            - destination_info
        """
        required = ["status", "metadata", "destination_info"]
        for field in required:
            if field not in task:
                self._append_log(make_log_entry(
                    self.name,
                    "warning_task_missing_field",
                    {"field": field, "task": task}
                ))

    # ------------------------------------------------------------
    # PER-TASK LIFECYCLE
    # ------------------------------------------------------------
    def _run_task(self, task: IngestionTask) -> dict:
        status = task.get("status", "pending")

        if status == "exists":
            self._append_log(make_log_entry(self.name, "skip_exists", task))
            return {"status": "skipped", "task": task}

        if status == "error":
            self._append_log(make_log_entry(self.name, "error", task))
            return {"status": "error", "task": task}

        # Create context before approval
        self.context = UpdateContext()
        self.context.metadata = task.get("metadata", {})
        self.context.destination_info = task.get("destination_info", {})

        if status == "pending":
            self.request_approval(self.context.metadata, self.context.destination_info)
            self.approve()

        elif status == "auto":
            self.context.transition(ApprovalState.APPROVED)
            self._append_log(make_log_entry(self.name, "auto_approved", self.context.metadata))

        self.apply_update()
        self.finalize()

        self._append_log(make_log_entry(self.name, "completed", task))
        return {"status": "completed", "task": task}

    # ------------------------------------------------------------
    # APPROVAL PHASE
    # ------------------------------------------------------------
    def request_approval(self, metadata: Dict[str, Any], destination_info: Dict[str, Any]):
        self.context.metadata = metadata or {}
        self.context.destination_info = destination_info or {}
        self.context.transition(ApprovalState.PENDING)
        self.log_event("approval_requested", metadata=metadata)

    def approve(self):
        self.context.transition(ApprovalState.APPROVED)
        self.log_event("approved", metadata=self.context.metadata)

    def reject(self, reason: str = ""):
        self.context.transition(ApprovalState.REJECTED)
        self.context.error = reason
        self.log_event("rejected", reason=reason)
        self.context.clear()

    # ------------------------------------------------------------
    # APPLY PHASE
    # ------------------------------------------------------------
    def apply_update(self):
        return self.context.metadata

    # ------------------------------------------------------------
    # FINALIZE PHASE
    # ------------------------------------------------------------
    def finalize(self):
        self.context.transition(ApprovalState.APPLIED)
        self.log_event("applied", metadata=self.context.metadata)

        if self.destination:
            self.destination.write(
                metadata=self.context.metadata,
                destination_info=self.context.destination_info,
            )

        # Unified log persistence
        self.ingest_log()
        self.context.clear()

    # ------------------------------------------------------------
    # LOGGING
    # ------------------------------------------------------------
    def log_event(self, event_type: str, **payload):
        # Inject destination_info if available
        if self.context and self.context.destination_info:
            payload["destination_info"] = self.context.destination_info

        entry = make_log_entry(self.name, event_type, payload)
        self._append_log(entry)


    def _append_log(self, entry: Dict[str, Any]):
        self._logs.append(entry)

    def add_log(self, entry: Dict[str, Any]):
        """
        Subclasses call this to add unified log entries.
        """
        self._logs.append(entry)

    def ingest_log(self):
        """
        Write all logs accumulated in self._logs.
        Subclasses may override this for custom behavior.
        """

        if self._logs:

            valid_logs = [
                rec for rec in self._logs if isinstance(rec, dict) and "id" in rec
            ]

            if valid_logs:

                fs.upsert_json_records(
                    self.log_path,
                    valid_logs,
                    key="id"
                )

            else:

                print('no valid log.')

    def get_logs(self):
        return self._logs

    # ------------------------------------------------------------
    # REQUIRED: retrieve_logs() with validation
    # ------------------------------------------------------------
    def retrieve_logs(self) -> List[Dict[str, Any]]:
        """
        Load logs from self.log_path using fs.read_json() and validate
        that each record contains the required fields:
            - id
            - name
            - text
            - metadata
        """
        records = fs.read_json(self.log_path)

        required_fields = ["id", "name", "text", "metadata"]

        for rec in records:
            for field in required_fields:
                if field not in rec:
                    raise ValueError(
                        f"Log record missing required field '{field}': {rec}"
                    )

        return records
