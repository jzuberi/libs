
from .intent_parser import WorkflowIntent


def validate_intent(engine, intent: WorkflowIntent) -> WorkflowIntent:
    """
    Ensures the intent is valid for the current workflow.
    """

    # Validate item_id
    if intent.item_id and intent.item_id not in engine._items:
        intent.reasoning = f"Item {intent.item_id} does not exist."
        intent.item_id = None

    # Validate step_name
    if intent.step_name:
        if intent.step_name not in engine.definition.step_specs:
            intent.reasoning = f"Step '{intent.step_name}' does not exist."
            intent.step_name = None

    return intent
