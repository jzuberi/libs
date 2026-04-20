from __future__ import annotations

import json
import textwrap
from typing import Dict, Any, Optional

from .intent_parser import WorkflowIntent
from .contract.loader import load_agent_contract

CONTRACT = load_agent_contract()


def parse_intent_with_llm(
    engine,
    user_message: str,
    pending_context: Optional[WorkflowIntent] = None,
) -> WorkflowIntent:
    """
    Use engine.agent_llm to interpret user intent.
    Falls back to rule-based parsing if LLM fails.

    If pending_context is provided, the LLM should consider whether the user
    is responding to a previous clarification question and may return one of:
      - confirm_pending
      - reject_pending
      - cancel_pending
      - provide_item_id

    Users may refer to items by their human-readable labels.
When the user mentions an item by label, map it to the correct item_id.

    """

    snapshot = engine.build_context_snapshot(max_chars=1000)

    # Build intent list from contract
    intent_list = "\n".join(
        f'- "{name}" — {meta.get("description", "")}'
        for name, meta in CONTRACT["intents"].items()
    )

    # Optional pending context section
    if pending_context is not None:
        pending_block = textwrap.dedent(f"""
            Pending context:
            The agent previously asked a clarification question about this intent:
            {json.dumps(pending_context.to_dict(), indent=2)}

            If the user is responding to that clarification, classify their message as one of:
              - "confirm_pending"
              - "reject_pending"
              - "cancel_pending"
              - "provide_item_id" (when they specify an item_id to use)

            Users may refer to items by their human-readable labels.
            When the user mentions an item by label, map it to the correct item_id.


            Otherwise, treat it as a normal intent from the list.
        """)
    else:
        pending_block = "No pending clarification context.\n"

    prompt = textwrap.dedent(f"""
        You are a workflow concierge agent. Interpret the user's request into a structured intent.

        User message:
        {user_message}

        {pending_block}

        Context - current items (JSON):
        {json.dumps(snapshot, indent=2)}

        Valid values for "intent" (with descriptions):
        {intent_list}

        Return ONLY a JSON object matching this schema:

        {{
          "intent": string,
          "item_id": string | null,
          "step_name": string | null,
          "edit_text": string | null,
          "reasoning": string | null
        }}

        Respond ONLY with valid JSON. Do not include commentary.
    """)

    try:
        raw = engine.agent_llm.metadata(prompt).dict()
        intent_data = raw

        print(intent_data)

        if not isinstance(intent_data, dict):
            return WorkflowIntent(intent="idle")

        intent = intent_data.get("intent", "idle")

        # Validate against contract
        if intent not in CONTRACT["intents"]:
            intent = "idle"

        return WorkflowIntent(
            intent=intent,
            item_id=intent_data.get("item_id"),
            step_name=intent_data.get("step_name"),
            edit_text=intent_data.get("edit_text"),
            reasoning=intent_data.get("reasoning"),
        )

    except Exception:
        return WorkflowIntent(intent="idle")
