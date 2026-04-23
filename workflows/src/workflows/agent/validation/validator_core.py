from typing import List, Tuple
from ..parsing.intent_parser import WorkflowIntent


def validate_intent(engine, intent: WorkflowIntent, contract) -> Tuple[bool, List[str], List[str]]:
    """
    Validate the intent against:
      - workflow items
      - workflow steps
      - contract-defined parameters

    Returns:
      (ok, missing_fields, errors)
    """

    missing = []
    errors = []

    params = intent.parameters or {}

    # --------------------------------------------------------------
    # 1. Validate item_id (if present)
    # --------------------------------------------------------------
    item_id = params.get("item_id")
    if item_id:
        if item_id not in engine._items:
            errors.append(f"Item '{item_id}' does not exist.")

    # --------------------------------------------------------------
    # 2. Validate step_name (if present)
    # --------------------------------------------------------------
    step_name = params.get("step_name")
    if step_name:
        if step_name not in engine.definition.step_specs:
            errors.append(f"Step '{step_name}' does not exist.")

    # --------------------------------------------------------------
    # 3. Validate contract-defined parameters
    # --------------------------------------------------------------
    meta = contract["intents"].get(intent.intent, {})
    param_specs = meta.get("parameters", {})

    for param_name, spec in param_specs.items():
        required = spec.get("required", False)
        value = params.get(param_name)

        if required and (value is None or value == ""):
            missing.append(param_name)

    # --------------------------------------------------------------
    # 4. Determine final result
    # --------------------------------------------------------------
    ok = (len(missing) == 0 and len(errors) == 0)
    return ok, missing, errors
