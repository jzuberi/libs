from __future__ import annotations

import json
import textwrap
from typing import Dict, Any, Optional

from ..parsing.intent_parser import WorkflowIntent

def parse_intent_with_llm(
    engine,
    contract,
    user_message: str,
    workflow_description: str,
    pending_context=None,
) -> WorkflowIntent:
    """
    Stage 1: Pure intent classification.
    Extracts only:
      - intent
      - reasoning
    """

    snapshot = engine.build_context_snapshot(max_chars=2000)

    # --------------------------------------------------------------
    # Build intent list (add clarify_pending)
    # --------------------------------------------------------------
    intent_list = "\n".join(
        f'- "{name}" — {meta.get("description", "")}'
        for name, meta in contract["intents"].items()
    )
    intent_list += '\n- "clarify_pending" — The user is providing missing information for a pending intent.'

    # --------------------------------------------------------------
    # Pending context block
    # --------------------------------------------------------------
    if pending_context is not None:
        pending_block = textwrap.dedent(f"""
            There is a pending intent that requires clarification.

            Pending intent:
            {pending_context.intent}

            Missing fields:
            {pending_context.missing}

            If the user is providing information to fill these missing fields,
            classify the message as "clarify_pending".

            If the user is issuing a new command unrelated to the pending intent,
            classify normally.
        """)
    else:
        pending_block = "There is no pending intent.\n"

    # --------------------------------------------------------------
    # Prompt
    # --------------------------------------------------------------
    prompt = textwrap.dedent(f"""
        You are a workflow concierge agent. Interpret the user's request into a structured intent.

        User message:
        {user_message}

        {pending_block}

        Current workflow description:
        {workflow_description}

        Valid values for "intent" (with descriptions):
        {intent_list}

        Return ONLY a JSON object matching this schema:

        {{
          "intent": string,
          "reasoning": string | null
        }}

        Respond ONLY with valid JSON. Do not include commentary.
    """)

    print(prompt)

    # --------------------------------------------------------------
    # LLM call
    # --------------------------------------------------------------
    try:
        raw = engine.agent_llm.metadata(prompt).dict()
        intent_data = raw

        print(intent_data)

        if not isinstance(intent_data, dict):
            return WorkflowIntent(intent="idle")

        intent_name = intent_data.get("intent", "idle")

        # Allow clarify_pending even though it's not in contract
        if intent_name != "clarify_pending" and intent_name not in contract["intents"]:
            intent_name = "idle"

        return WorkflowIntent(
            intent=intent_name,
            reasoning=intent_data.get("reasoning"),
        )

    except Exception:
        return WorkflowIntent(intent="idle")
