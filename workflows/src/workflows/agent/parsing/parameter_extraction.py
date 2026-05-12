import json
import textwrap

from datetime import datetime
from pathlib import Path
from enum import Enum
from pydantic import BaseModel
from typing import Any, Dict, Optional


def jsonable(obj):
    """
    Convert a Pydantic model (or any nested structure containing them)
    into a JSON‑serializable Python object.
    """

    # Pydantic model → dict
    if isinstance(obj, BaseModel):
        return {k: jsonable(v) for k, v in obj.model_dump().items()}

    # Datetime → ISO string
    if isinstance(obj, datetime):
        return obj.isoformat()

    # Enum → value
    if isinstance(obj, Enum):
        return obj.value

    # Path → string
    if isinstance(obj, Path):
        return str(obj)

    # Dict → recursively convert values
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}

    # List / tuple → recursively convert items
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]

    # Primitive → return as‑is
    return obj


def build_relevant_context_for_intent(agent, intent):
    """
    Build relevant context for an intent based on ontology metadata
    and the intent's declared uses_ontology list.

    Returns a dict like:
    {
        "ideas": [...],
        "asset_definitions": [...],
    }
    or None if nothing applies.
    """

    # 1. Read intent metadata
    meta = agent.contract["intents"].get(intent.intent, {})
    type_names = meta.get("uses_ontology")
    if not type_names:
        return None

    session = agent.session.context
    ontology = agent.workflow_ontology

    out = {}

    # 2. For each ontology type the intent uses
    for type_name in type_names:
        ot = ontology.get(type_name)
        if not ot:
            continue

        # 3. Where do instances of this type live in session?
        session_key = ot.metadata.get("session_key")
        if not session_key:
            continue

        objects = session.get(session_key) or []
        if not objects:
            continue

        # 4. How should this appear in relevant_context?
        context_key = ot.metadata.get("context_key", type_name.lower())

        # 5. Serialize objects using ontology rules
        serialized = [ot.serialize(obj) for obj in objects]

        # 6. Filter to ontology-declared fields (if provided)
        fields = ot.fields or None
        if fields:
            filtered = [
                {k: v for k, v in item.items() if k in fields}
                for item in serialized
            ]
        else:
            filtered = serialized

        out[context_key] = filtered

    return out or None



def extract_parameters_with_llm(
    engine,
    contract,
    user_message,
    intent,
    session,
    workflow_description: str,
    relevant_context=None,
):
    meta = contract["intents"].get(intent.intent, {})
    param_specs = meta.get("parameters", {})

    if not param_specs:
        return {}

    extraction_instructions = meta.get("extraction_instructions", "").strip()

    # Build schema block
    schema_lines = []
    for name, spec in param_specs.items():
        required = spec.get("required", False)
        ptype = spec.get("type", "string")
        schema_lines.append(f'- "{name}": type={ptype}, required={required}')
    schema_block = "\n".join(schema_lines)

    # Minimal session context
    current_item_id = session.last_item_id
    current_step = engine.get_current_step(current_item_id).name
    minimal_context = {
        "current_step_name": current_step,
        "current_item_id": current_item_id,
    }
    context_block = json.dumps(jsonable(minimal_context), indent=2)

    # ------------------------------------------------------------
    # Relevant context block (ontology-driven)
    # ------------------------------------------------------------
    relevant_block = ""
    if relevant_context:
        lines = ["Relevant context:"]
        for key, items in relevant_context.items():
            lines.append(f"{key}:")
            for item in items:
                if isinstance(item, dict):
                    # Serialize dicts as "k: v, k2: v2"
                    serialized = ", ".join(f"{k}: {v}" for k, v in item.items())
                    lines.append(f"  - {serialized}")
                else:
                    # Fallback for non-dict items
                    lines.append(f"  - {item}")
        relevant_block = "\n".join(lines)

    # ------------------------------------------------------------
    # Build prompt
    # ------------------------------------------------------------
    prompt = textwrap.dedent(f"""
        You are a workflow assistant. Extract parameters for the intent "{intent.intent}".

        User message:
        {user_message}

        Parameter schema for this intent:
        {schema_block}

        Extraction instructions for this intent:
        {extraction_instructions}

    """)

    if relevant_block:
        prompt += f"\n\n{relevant_block}\n"

    prompt += textwrap.dedent("""
        Follow these rules:
        - Extract parameters strictly according to the schema.
        - Apply the extraction instructions above.
        - Use context only when the user refers indirectly (e.g., "it", "that one").
        - If a parameter cannot be extracted, set it to null.
        - Return ONLY valid JSON. No commentary.

        Return JSON in this format:
        {
            "parameters": {
                <param_name>: <value or null>
            },
            "reasoning": "<brief explanation>"
        }
    """)

    print('parameter extraction prompt: ')
    print(prompt)

    # ------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------
    raw = engine.agent_llm.metadata(prompt).dict()
    if not isinstance(raw, dict):
        return {}

    return raw.get("parameters", {})
