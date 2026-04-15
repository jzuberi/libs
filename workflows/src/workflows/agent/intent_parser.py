from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .session import SessionState
from .contract.loader import load_agent_contract

CONTRACT = load_agent_contract()


# ---------------------------------------------------------
# Intent Model
# ---------------------------------------------------------

@dataclass
class WorkflowIntent:
    intent: str
    item_id: Optional[str] = None
    step_name: Optional[str] = None
    edit_text: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    reasoning: Optional[str] = None

    def to_dict(self):
        return {
            "intent": self.intent,
            "item_id": self.item_id,
            "step_name": self.step_name,
            "edit_text": self.edit_text,
            "parameters": self.parameters,
            "reasoning": self.reasoning,
        }




# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def extract_step_name(text: str) -> Optional[str]:
    parts = text.split()
    if len(parts) < 3:
        return None
    return parts[-1]


def extract_edit_text(text: str) -> Optional[str]:
    if ":" in text:
        return text.split(":", 1)[1].strip()
    return None


# ---------------------------------------------------------
# Contract-aware rule-based parser
# ---------------------------------------------------------

def parse_intent(message: str, session: SessionState) -> WorkflowIntent:
    text = message.strip().lower()

    # ---------------------------------------------------------
    # StepContext-powered commands
    # ---------------------------------------------------------

    if text.startswith("show step"):
        step = extract_step_name(text)
        return WorkflowIntent(
            intent="show_step_output",
            item_id=session.last_item_id,
            step_name=step,
        )

    if "list" in text and "step" in text and "output" in text:
        return WorkflowIntent(
            intent="list_step_outputs",
            item_id=session.last_item_id,
        )

    if text.startswith("edit step"):
        step = extract_step_name(text)
        edit = extract_edit_text(message)
        return WorkflowIntent(
            intent="edit_step_output",
            item_id=session.last_item_id,
            step_name=step,
            edit_text=edit,
        )

    if text.startswith("show schema"):
        step = extract_step_name(text)
        return WorkflowIntent(
            intent="show_schema",
            item_id=session.last_item_id,
            step_name=step,
        )

    if text.startswith("explain schema"):
        step = extract_step_name(text)
        return WorkflowIntent(
            intent="explain_schema",
            item_id=session.last_item_id,
            step_name=step,
        )

    # ---------------------------------------------------------
    # Existing commands (contract-aware)
    # ---------------------------------------------------------

    if text.startswith("run next") or "run the next step" in text:
        return WorkflowIntent(intent="run_next_step", item_id=session.last_item_id)

    if text.startswith("approve"):
        return WorkflowIntent(intent="approve_substate", item_id=session.last_item_id)

    if text.startswith("export"):
        return WorkflowIntent(intent="export", item_id=session.last_item_id)

    if "status" in text or "what's going on" in text:
        return WorkflowIntent(intent="query_item", item_id=session.last_item_id)

    if "list" in text and "items" in text:
        return WorkflowIntent(intent="query_workflow")

    if "describe" in text and "workflow" in text:
        return WorkflowIntent(intent="describe_workflow")

    if "list step outputs" in text:
        return WorkflowIntent(intent="list_step_outputs")


    # ---------------------------------------------------------
    # Contract-driven fallback
    # ---------------------------------------------------------

    # Try to match any contract-defined intent by keyword
    for intent_name, meta in CONTRACT["intents"].items():
        if intent_name in text:
            return WorkflowIntent(intent=intent_name)

    # No match
    return WorkflowIntent(intent="idle", reasoning="No clear intent matched.")
