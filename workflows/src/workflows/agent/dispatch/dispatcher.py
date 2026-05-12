# workflows/agent/dispatch/dispatcher.py

from ..resolution.resolver_core import resolve_item_reference
from ..validation.validator_core import validate_intent
from ..session import PendingIntent
from ...engine.models import HandlerMessage, merge_messages

import time

def render_user_output(user_output):
    if isinstance(user_output, HandlerMessage):
        return user_output.render()
    return str(user_output)

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

    # --------------------------------------------------------------
    # 4. Dispatch to handler
    # --------------------------------------------------------------
    handler = agent._get_handler(intent.intent)
    raw_objects, user_output = handler(
        agent, 
        intent, 
        item_id, 
        resolution_msg
    )

    handler_name = handler.__name__
    agent.session.last_handler_name = handler_name

    # Special case: create_item updates item_id
    if handler_name == 'handle_create_item':
        item_id = raw_objects['current_item_id']
        agent.session.last_item_id = item_id
        agent.engine._load_existing_items()

    # Normalize handler output
    messages = [user_output]

    # --------------------------------------------------------------
    # BLOCKED HANDLER OUTPUT (e.g., approval failed)
    # --------------------------------------------------------------
    if isinstance(raw_objects, dict) and raw_objects.get("blocked") is True:
        agent.save()
        return merge_messages(messages)

    if trace:
        trace.record_agent_response(user_output)

    if item_id is None:
        messages.append("item_id is None")
        return merge_messages(messages)

    # --------------------------------------------------------------
    # 5. AUTO‑ADVANCE WORKFLOW (incremental)
    # --------------------------------------------------------------
    while True:

        item = agent.engine.load_item(item_id)
        old_substate = item.status.substate

        # Stop if approval is required
        if item.status.requires_approval and not item.status.approved:
            break

        # Run next step
        try:
            step_result = agent.engine.run_next_step(item_id)
        except Exception as e:
            # Hard engine error (should be rare)
            messages.append(f"⚠️ Engine exception: {str(e)}")
            return merge_messages(messages)

        if not step_result:
            break

        # ----------------------------------------------------------
        # NEW: Surface workflow step errors
        # ----------------------------------------------------------
        if hasattr(step_result, "details") and step_result.details:
            if step_result.details.get("error"):
                messages.append(
                    f"⚠️ Workflow error in step '{old_substate}': {step_result.details['error']}"
                )
                agent.save()
                return merge_messages(messages)

        # Update session context
        agent._update_session_context_from_step_output(old_substate)

        # Reload item to check new substate
        item = agent.engine.load_item(item_id)
        new_substate = item.status.substate

        # If step didn't advance, append summary and stop
        if new_substate == old_substate:
            messages.append(step_result.summary)
            break

        # ----------------------------------------------------------
        # Auto‑advance message
        # ----------------------------------------------------------
        step_record = item.step_outputs.get(old_substate)
        bullets = []

        if step_record and step_record.current:
            for k, v in step_record.current.items():
                bullets.append(f"**{k}**: {v}")

        auto_msg = HandlerMessage(
            title=f"Step Completed: {old_substate}",
            body="The workflow automatically advanced.",
            bullets=bullets,
        )

        messages.append(auto_msg)

        time.sleep(1)

    agent.save()
    return merge_messages(messages)
