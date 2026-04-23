import json
import textwrap

from datetime import datetime
from pathlib import Path
from enum import Enum
from pydantic import BaseModel

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


def extract_parameters_with_llm(
    engine,
    contract,
    user_message: str,
    intent,
    session,
    workflow_description: str,
):
    """
    Stage 2: LLM-powered parameter extraction.
    Contract-aware, context-aware, workflow-aware.
    Returns a dict of extracted parameters.
    """

    # Get parameter schema for this intent
    meta = contract["intents"].get(intent.intent, {})
    param_specs = meta.get("parameters", {})

    # Build JSON schema for the LLM
    schema_lines = []
    for name, spec in param_specs.items():
        required = spec.get("required", False)
        ptype = spec.get("type", "string")
        schema_lines.append(f'- "{name}": type={ptype}, required={required}')

    schema_block = "\n".join(schema_lines)

    # Build context block
    safe_context = jsonable(session.context)

    context_block = json.dumps(safe_context, indent=2)

    prompt = textwrap.dedent(f"""
        You are a workflow assistant. Extract parameters for the intent "{intent.intent}".

        User message:
        {user_message}

        Workflow description:
        {workflow_description}

        Session context (may contain case_id, item_id, step_name, etc.):
        {context_block}

        Parameter schema for this intent:
        {schema_block}

        Extract parameters according to the schema.
        Use context when the user refers indirectly (e.g., "it", "that case").
        Use workflow labels to resolve references.
        If a parameter cannot be extracted, set it to null.

        Return ONLY a JSON object with:
        {{
            "parameters": {{
                <param_name>: <value or null>
            }},
            "reasoning": "<brief explanation>"
        }}

        Respond ONLY with valid JSON. No commentary.
    """)

    raw = engine.agent_llm.metadata(prompt).dict()

    if not isinstance(raw, dict):
        return {}

    params = raw.get("parameters", {})
    return params
