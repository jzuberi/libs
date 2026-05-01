# workflows/agent/dispatch/dispatcher.py

from ..resolution.resolver_core import resolve_item_reference
from ..validation.validator_core import validate_intent
from ..session import PendingIntent

import time

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
    # 4. Dispatch to handler
    # --------------------------------------------------------------
    handler = agent._get_handler(intent.intent)
    raw_objects, user_output = handler(agent, intent, item_id, resolution_msg)

    if trace:
        trace.record_agent_response(user_output)

    # --------------------------------------------------------------
    # 5. AUTO‑ADVANCE WORKFLOW (incremental)
    # --------------------------------------------------------------
    # Only advance if the handler approved the substate
    # (engine tracks this internally)
    while True:

        # Capture old substate
        item = agent.engine.load_item(item_id)
        old_substate = item.status.substate

        if item.status.requires_approval and not item.status.approved:
            break

        # Run exactly one step

        step_result = None

        try:
            step_result = agent.engine.run_next_step(item_id)
        except:
            pass

        # If nothing ran, break
        if not step_result:
            break

        # Update session context from step output
        agent._update_session_context_from_step_output(old_substate)

        # Reload item to check new substate
        item = agent.engine.load_item(item_id)
        new_substate = item.status.substate

        # If substate didn't change, stop auto‑advancing
        if new_substate == old_substate:
            break
        else:
            print('successfully ran ' + str(old_substate))
            user_output += '\n successfully ran step: ' + str(old_substate)

            record = item.step_outputs[old_substate]
            if record and record.current:
                user_output += "\n\n" + "current output:\n" + str(record.current)


        # Otherwise continue the loop (run next step)

        time.sleep(1)

    return user_output

