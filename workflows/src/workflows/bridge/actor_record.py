from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Literal
import json
import uuid
import portalocker


ActorType = Literal["human", "agent", "system"]


class ActorRecord:
    """
    Unified per-actor interface for:
      - logging events globally (chat, steps, errors)
      - querying recent events (global or filtered)
      - inferring workflow + item context
      - handling concurrency via file locks
      - supporting log rotation (cold storage)

    """

    # ------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------
    def __init__(
        self,
        agent,
        actor_id: str,
        actor_type: ActorType,
        log_path: Path,
        source: str = "cli",
        session_id: Optional[str] = None,
        workflow_name: str = "",
    ):
        self.agent = agent
        self.workflow_name = workflow_name
        self.actor_id = actor_id
        self.actor_type = actor_type
        self.log_path = log_path
        self.source = source
        self.session_id = session_id or self._generate_session_id()

        # Track last known workflow + item context
        self.last_item_id: Optional[str] = None
        self.last_workflow_name: Optional[str] = None

        # Track last completed step to detect transitions
        self.last_completed_step: Optional[str] = None

    # ------------------------------------------------------------
    # Core event writer (GLOBAL JSONL)
    # ------------------------------------------------------------
    def log_event(
        self,
        event_type: str,
        workflow_name: Optional[str] = None,
        item_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ):
        """
        Append a single structured event to the global JSONL log.
        This is the ONLY write path.
        """
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "source": self.source,
            "session_id": self.session_id,
            "workflow_name": workflow_name or self._infer_workflow_name(),
            "item_id": item_id or self._infer_item_id(),
            "event_type": event_type,
            "payload": payload or {},
        }

        # Ensure parent directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic append with file lock
        lock_path = self.log_path.with_suffix(self.log_path.suffix + ".lock")
        with portalocker.Lock(str(lock_path), timeout=5):
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")

        return(event)

    # ------------------------------------------------------------
    # Chat wrapper (logs user + assistant + step transitions)
    # ------------------------------------------------------------
    def chat(self, user_message: str):
        """
        Wrap agent.handle_message:
        - log chat_user
        - call agent
        - log chat_assistant
        - capture step transitions
        - return BOTH the assistant message AND the assistant log event
        """

        # Capture identifiers
        actor_id = self.actor_id
        session_id = self.session_id
        workflow_name = self.workflow_name
        item_id = self._infer_item_id()

        # Capture last assistant event timestamp BEFORE chat
        """
        before = ActorRecord.recent_for_actor(actor_id, limit=1)
        before_ts = before[0]["timestamp"] if before else None
        """

        # 1. Log user message
        self.log_event(
            event_type="chat_user",
            payload={"message": user_message},
        )

        # 2. Send to agent
        handler_message = self.agent.handle_message(user_message)

        # 3. Log assistant response AND capture the event object
        assistant_event = self.log_event(
            event_type="chat_assistant",
            payload={
                "message": handler_message,
                "handler_name": self.agent.session.last_handler_name,
            },
        )

        # 4. Capture workflow step transitions
        self._capture_step_event_if_needed()

        # 5. Return BOTH the message and the assistant event
        return {
            "message": handler_message,
            "assistant_event": assistant_event,
            "actor_id": actor_id,
            "session_id": session_id,
            "workflow_name": workflow_name,
            "item_id": item_id,
        }


    # ------------------------------------------------------------
    # Step transition detection
    # ------------------------------------------------------------
    def _capture_step_event_if_needed(self):
        """
        Detect workflow substate changes and log step_completed events.
        """
        engine = self.agent.engine
        item_id = self.agent.session.last_item_id
        if not item_id:
            return

        item = engine.load_item(item_id)
        current_step = item.status.substate

        if current_step == self.last_completed_step:
            return

        step_outputs = item.step_outputs

        raw_output = None
        if current_step in step_outputs:
            raw_output = step_outputs[current_step].model_dump().get("raw")

        prev_output = None
        if self.last_completed_step in step_outputs:
            prev_output = step_outputs[self.last_completed_step].model_dump().get("current")

        self.log_event(
            event_type="step_completed",
            item_id=item_id,
            payload={
                "step_completed": current_step,
                "handler_name": self.agent.session.last_handler_name,
                "raw_output_of_completed_step": raw_output,
                "current_output_of_previous_step": prev_output,
            },
        )

        self.last_completed_step = current_step

    # ------------------------------------------------------------
    # Global query API
    # ------------------------------------------------------------
    @classmethod
    def recent_actions(
        cls,
        log_path: Path,
        *,
        limit: int = 50,
        actor_id: Optional[str] = None,
        workflow_name: Optional[str] = None,
        item_id: Optional[str] = None,
        event_type: Optional[str] = None,
        since: Optional[datetime] = None,
        include_history: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Read the global log (and optionally history logs) newest → oldest,
        apply filters, return most recent N events.
        """

        def iter_lines_reverse(path: Path):
            """Yield lines from a file in reverse order."""
            if not path.exists():
                return
            with path.open("r", encoding="utf-8") as f:
                f.seek(0, 2)  # go to end
                position = f.tell()
                buffer = ""

                while position >= 0:
                    f.seek(position)
                    char = f.read(1)
                    if char == "\n":
                        if buffer:
                            yield buffer[::-1]
                            buffer = ""
                    else:
                        buffer += char
                    position -= 1

                if buffer:
                    yield buffer[::-1]

        # ------------------------------------------------------------
        # Collect log files (active + optional history)
        # ------------------------------------------------------------
        files = [log_path]

        if include_history:
            history_dir = log_path.parent / "history"
            if history_dir.exists():
                history_files = sorted(
                    history_dir.glob("event_log_*.jsonl"),
                    reverse=True
                )
                files.extend(history_files)

        # ------------------------------------------------------------
        # Scan newest → oldest across all files
        # ------------------------------------------------------------
        results = []

        for file in files:
            for line in iter_lines_reverse(file):
                try:
                    event = json.loads(line)
                except Exception:
                    continue  # skip malformed lines

                # -------------------------
                # Apply filters
                # -------------------------
                if actor_id and event.get("actor_id") != actor_id:
                    continue

                if workflow_name and event.get("workflow_name") != workflow_name:
                    continue

                if item_id and event.get("item_id") != item_id:
                    continue

                if event_type and event.get("event_type") != event_type:
                    continue

                if since:
                    ts = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00"))
                    if ts < since:
                        continue

                results.append(event)

                if len(results) >= limit:
                    return results

        return results


    @classmethod
    def recent_for_actor(cls, log_path: Path, actor_id: str, limit: int = 50):
        return cls.recent_actions(log_path, limit=limit, actor_id=actor_id)

    @classmethod
    def recent_for_workflow(cls, log_path: Path, workflow_name: str, limit: int = 50):
        return cls.recent_actions(log_path, limit=limit, workflow_name=workflow_name)

    @classmethod
    def recent_for_item(cls, log_path: Path, item_id: str, limit: int = 50):
        return cls.recent_actions(log_path, limit=limit, item_id=item_id)

    # ------------------------------------------------------------
    # Log rotation (cold storage)
    # ------------------------------------------------------------
    @classmethod
    def rotate_logs(cls, log_path: Path, max_size_mb: int = 50):
        """
        If the active log exceeds max_size_mb, move it to history and create a new one.
        """
        if not log_path.exists():
            return

        size_mb = log_path.stat().st_size / (1024 * 1024)
        if size_mb < max_size_mb:
            return

        history_dir = log_path.parent / "history"
        history_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).isoformat().replace(":", "-")
        archive_path = history_dir / f"event_log_{timestamp}.jsonl"

        # Move the active log → history
        log_path.rename(archive_path)

        # Create a new empty active log
        log_path.touch()


    # ------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------
    def _generate_session_id(self) -> str:
        return f"sess_{uuid.uuid4().hex[:8]}"

    def _infer_workflow_name(self) -> Optional[str]:
        return self.workflow_name


    def _infer_item_id(self) -> Optional[str]:
        return self.agent.session.last_item_id
