# workflows/agent/dispatch/dispatcher.py

from ..resolution.resolver_core import resolve_item_reference
from ..validation.validator_core import validate_intent
from ..session import PendingIntent


def dispatch_intent(agent, intent, trace=None):

    print("dispatch_intent start")
    print("parsed intent:")
    print(intent)

    # --------------------------------------------------------------
    # 1. Resolve item reference
    # --------------------------------------------------------------
    raw_ref = intent.parameters.get("item_id")

    item_id, resolution_msg = resolve_item_reference(
        agent.engine,
        raw_ref,
        agent.session,
    )

    if item_id is not None:
        intent.parameters["item_id"] = item_id

    print("resolved intent:")
    print(intent)

    # --------------------------------------------------------------
    # 2. Validate against workflow contract
    # --------------------------------------------------------------
    ok, missing, errors = validate_intent(agent.engine, intent, agent.contract)

    if errors:
        return " ".join(errors)

    if missing:
        agent.session.pending_intent = PendingIntent(
            intent=intent.intent,
            parameters=dict(intent.parameters),
            missing=missing,
            original_message=intent.reasoning or "",
        )
        missing_str = ", ".join(missing)
        return f"I need {missing_str} before I can continue."

    print("validated intent:")
    print(intent)

    if trace:
        trace.record_final_intent(intent.intent)

    agent.session.last_intent = intent.intent

    if trace:
        label = agent.engine.get_item_label(item_id) if item_id else None
        trace.record_item_resolution(
            {
                "item_id": item_id,
                "item_label": label,
                "resolution_msg": resolution_msg,
            }
        )

    # --------------------------------------------------------------
    # 3. Update session with current item
    # --------------------------------------------------------------
    if item_id:
        agent.session.last_item_id = item_id
        agent.session.context["current_item_id"] = item_id

    # --------------------------------------------------------------
    # 4. Dispatch to handler (NEW: handler(agent, ...))
    # --------------------------------------------------------------
    handler = agent._get_handler(intent.intent)

    raw_objects, user_output = handler(agent, intent, item_id, resolution_msg)

    if trace:
        trace.record_agent_response(user_output)

    return user_output
