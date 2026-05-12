# workflows/agent/handlers/handlers.py

from ..context.decorators import updates_context
from ..context.normalization import normalize_item
from ...engine.utils.tracing import agent_trace

def base_handler(fn):
    fn._is_base_handler = True
    return fn





# ----------------------------------------------------------------------
# List Workflow Items
# ----------------------------------------------------------------------
@agent_trace("handle_list_workflow_items")
@updates_context(
    mapping={
        "items": ("items", normalize_item)
    },
    set_current_item=False
)
@base_handler
def handle_list_workflow_items(agent, intent, item_id, resolution_msg):
    items = agent.engine.list_items()
    if not items:
        agent.session.last_listed_items = []
        return {"items": []}, "There are no items in this workflow yet."

    items = sorted(items, key=lambda it: it.created_at, reverse=True)
    limit = getattr(agent, "DEFAULT_RECENT_LIMIT", 5)
    items = items[:limit]

    agent.session.last_listed_items = [it.id for it in items]

    lines = ["Here are the most recent workflow runs:\n"]
    for idx, item in enumerate(items, start=1):
        label = item.label or agent.engine.get_item_label(item.id)
        status_str = (
            f"{item.status.branch}/{item.status.substate} | "
            f"approved={item.status.approved}"
        )
        lines.append(f"{idx}. {label} — {status_str}")

    return {"items": items}, "\n".join(lines)

# ----------------------------------------------------------------------
# Unknown Intent
# ----------------------------------------------------------------------
@agent_trace("handle_unknown_intent")
@updates_context(mapping={})
@base_handler
def handle_unknown_intent(agent, intent, item_id, resolution_msg):
    return {}, (
        "I couldn’t map that to a workflow action. "
        "Try asking me to list items, run the next step, approve, export, "
        "show step outputs, or show schemas."
    )


# ----------------------------------------------------------------------
# Describe Workflow
# ----------------------------------------------------------------------
@agent_trace("handle_describe_workflow")
@updates_context(mapping={})
@base_handler
def handle_describe_workflow(agent, intent, item_id, resolution_msg):
    return {}, agent._describe_workflow()



@agent_trace("handle_create_item")
@updates_context(
    mapping={},
    set_current_item=True
)
@base_handler
def handle_create_item(agent, intent, item_id, resolution_msg):
    """
    Create a brand new workflow item and reset the agent trace + context.
    """
    
    # 1. Create the item via the engine
    result = agent.engine.create_item(
         description="content-production",
         type="content_production",
         initial_substate="load_ideas",
    )
    
    new_item_id = result.id

    # 2. Reset interactive trace
    agent.session.turns = []
    agent.session.step_events = []
    agent.session.last_completed_step = None
    agent.session.last_handler_name = None

    # 3. Reset item-scoped context
    agent.session.context["items"] = []

    # Reset updated_turn markers
    agent.session.context["_updated_turn"] = {
        "items": agent.session.turn_index,
        "steps": agent.session.turn_index,
        "current_item_id": agent.session.turn_index,
    }

    # 4. Update the single source of truth
    agent.session.last_item_id = new_item_id

    # 5. Optional: keep this convenience field in sync
    agent.session.context["current_item_id"] = new_item_id

    # 6. Persist
    agent.save()

    # 7. Return handler output
    return (
        {"current_item_id": new_item_id},
        f"Created a new workflow item: {new_item_id}"
    )

