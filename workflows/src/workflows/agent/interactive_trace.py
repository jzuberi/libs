from datetime import datetime, timezone
from typing import Literal, Optional, Any, Dict, List

import json
from pathlib import Path

TurnRole = Literal["user", "assistant", "system"]
ActorType = Literal["human", "agent", "system"]


class InteractiveTrace:
    
    def __init__(self, agent, actor_id: str = "human", actor_type: ActorType = "human"):
        self.agent = agent

        # identity of the actor driving this session
        self.actor_id: str = actor_id
        self.actor_type: ActorType = actor_type

        # conversational history (Anthropic-style)
        self.turns: List[Dict[str, Any]] = []

        # workflow progression events
        self.step_events: List[Dict[str, Any]] = []

        # workflow state tracking
        self.last_completed_step: Optional[str] = None
        self.last_handler_name: Optional[str] = None

        # optional: item + intent context
        self.context: Dict[str, Any] = {}
        self.last_item_id: Optional[str] = None
        self.last_intent: Optional[str] = None

    # ------------------------------------------------------------
    # Create a new item
    # ------------------------------------------------------------
    def create_item(self, *args, **kwargs):
        result = self.agent.run(*args, **kwargs)

        # reset trace
        self.turns = []
        self.step_events = []
        self.last_completed_step = None
        self.last_handler_name = None

        item_id = result.get("item_id")
        self.last_item_id = item_id
        self.context["current_item_id"] = item_id

        self.save()
        return item_id

    # ------------------------------------------------------------
    # Chat API
    # ------------------------------------------------------------
    def chat(self, user_message: str):
        now = datetime.now(timezone.utc).isoformat()

        # 1. Record user turn
        self.turns.append({
            "role": "user",
            "content": user_message,
            "timestamp": now,
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
        })

        # 2. Send to agent
        handler_message = self.agent.handle_message(user_message)

        # 3. Record assistant turn
        self.turns.append({
            "role": "assistant",
            "content": handler_message,
            "handler_name": self.agent.session.last_handler_name,
            "timestamp": datetime.utcnow().isoformat(),
            "actor_id": None,
            "actor_type": "agent",
        })

        # 4. Capture workflow step completion
        self._capture_step_event_if_needed()

        # 5. Save incrementally
        self.save()

        return handler_message

    # ------------------------------------------------------------
    # Step event capture
    # ------------------------------------------------------------
    def _capture_step_event_if_needed(self):
        
        engine = self.agent.engine
        item_id = self.agent.session.last_item_id

        if(item_id is None):
            return(None)
        
        item = engine.load_item(item_id)

        current_step = item.status.substate

        # If no change, nothing to record
        if current_step == self.last_completed_step:
            return

        step_outputs = item.step_outputs

        # Raw output of the completed step
        raw_output = None
        if current_step in step_outputs:
            record = step_outputs[current_step]
            raw_output = record.model_dump().get("raw")

        # Current output of the previous step
        current_output_prev = None
        if self.last_completed_step and self.last_completed_step in step_outputs:
            prev_record = step_outputs[self.last_completed_step]
            current_output_prev = prev_record.model_dump().get("current")

        event = {
            "step_completed": current_step,
            "timestamp": datetime.utcnow().isoformat(),
            "actor_id": self.actor_id,
            "actor_type": self.actor_type,
            "handler_name": self.agent.session.last_handler_name,
            "raw_output_of_completed_step": raw_output,
            "current_output_of_previous_step": current_output_prev,
        }

        self.step_events.append(event)
        self.last_completed_step = current_step

    # ------------------------------------------------------------
    # Export trace
    # ------------------------------------------------------------
    def history(self):
        return list(self.turns)

    def export_trace(self):
        return {
            "turns": self.history(),
            "step_events": list(self.step_events),
        }

    # ------------------------------------------------------------
    # Incremental save
    # ------------------------------------------------------------
    def save(self):
        item_id = self.agent.session.last_item_id
        if not item_id:
            return
        
        item_dir = self.agent.engine._item_dir(item_id)

        path = item_dir / "interactive_trace.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing trace
        if path.exists():
            existing = json.loads(path.read_text())
        else:
            existing = {"turns": [], "step_events": []}

        # Export current trace
        new = self.export_trace()

        # Merge incrementally
        merged = {
            "turns": existing.get("turns", []) + new.get("turns", []),
            "step_events": existing.get("step_events", []) + new.get("step_events", []),
        }

        path.write_text(json.dumps(merged, indent=2))

    # ------------------------------------------------------------
    # Load
    # ------------------------------------------------------------
    @classmethod
    def load(cls, agent, item_id):

        item_dir = agent.engine._item_dir(item_id)
        
        path = item_dir / "interactive_trace.json"

        session = cls(agent)
        session.last_item_id = item_id

        if path.exists():
            data = json.loads(path.read_text())
            session.turns = data.get("turns", [])
            session.step_events = data.get("step_events", [])

        return session
