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
         initial_substate="start",
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

@agent_trace("handle_show_current_item")
@updates_context(
    mapping={},
    set_current_item=False
)
@base_handler
def handle_show_current_item(agent, intent, item_id, resolution_msg):
    """
    Return a detailed, human-readable summary of the current workflow item.
    Does NOT persist anything into context.
    """

    # 1. Determine the current item
    current_id = item_id or agent.session.last_item_id
    if not current_id:
        return (
            {"current_item": None},
            "There is no active workflow item yet."
        )

    # 2. Fetch the item from the engine
    try:
        item = agent.engine.get_item(current_id)
    except Exception as e:
        return (
            {"current_item": None},
            f"Could not load item {current_id}: {e}"
        )

    # 3. Normalize into a serializable dict

    item_dict = {
        "id": item.id,
        "label": item.label,
        "type": item.type,
        "created_at": item.created_at.isoformat() if item.created_at else None,
        "updated_at": item.updated_at.isoformat() if item.updated_at else None,
        "status": {
            "branch": item.status.branch,
            "substate": item.status.substate,
            "approved": item.status.approved,
        },
        "metadata": item.metadata or {},
    }

    # 4. Build a rich human-readable summary
    summary_lines = []

    summary_lines.append(f"📌 **Item ID:** {item.id}")
    summary_lines.append(f"🏷️ **Label:** {item.label or '(none)'}")
    summary_lines.append(f"📂 **Type:** {item.type}")
    summary_lines.append("")
    summary_lines.append("### 🔧 Workflow Status")
    summary_lines.append(f"- Branch: **{item.status.branch}**")
    summary_lines.append(f"- Substate: **{item.status.substate}**")
    summary_lines.append(f"- Approved: **{item.status.approved}**")
    summary_lines.append("")
    summary_lines.append("### 🕒 Timestamps")
    summary_lines.append(f"- Created: {item_dict['created_at']}")
    summary_lines.append(f"- Updated: {item_dict['updated_at']}")
    summary_lines.append("")
    summary_lines.append("### 📝 Metadata")
    if item.metadata:
        for k, v in item.metadata.items():
            summary_lines.append(f"- **{k}:** {v}")
    else:
        summary_lines.append("- (none)")
    summary_lines.append("")
    summary_lines.append("### 🪜 Steps")

    human_summary = "\n".join(summary_lines)

    # 5. Return structured + human-readable output
    return (
        {"current_item": item_dict},
        human_summary
    )

@agent_trace("handle_load_recent")
@updates_context(
    mapping={},
    set_current_item=True
)
@base_handler
def handle_load_recent(agent, intent, item_id, resolution_msg):
    """
    Create a brand new workflow item and reset the agent trace + context.
    """

    item = agent.load()
    new_item_id = item.id

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
        f"loaded workflow item: {new_item_id}"
    )

@agent_trace("handle_rewind_substate")
@updates_context(
    mapping={},
    set_current_item=True
)
@base_handler
def handle_rewind_substate(agent, intent, item_id, resolution_msg):
    """
    Rewind the current workflow item to a previous substate.
    Deletes all step outputs for that substate and all later substates.
    """

    # 1. Determine current item
    current_id = item_id or agent.session.last_item_id
    if not current_id:
        return (
            {"rewound": False},
            "There is no active workflow item to rewind."
        )

    # 2. Extract target substate from the intent payload
    target = intent.parameters.get("target_substate").lower()
    if not target:
        return (
            {"rewound": False},
            "No target substate was provided. Please specify which substate to rewind to."
        )

    # 3. Attempt the rewind
    try:
        success = agent._rewind_substate(target_substate=target, item_id=current_id)
    except Exception as e:
        return (
            {"rewound": False, "error": str(e)},
            f"Could not rewind item {current_id} to '{target}': {e}"
        )

    # 4. If nothing changed (target was ahead or equal)
    if not success:
        return (
            {"rewound": False},
            f"Item {current_id} is already at or before substate '{target}'. No rewind performed."
        )

    # 5. Load updated item for summary
    item = agent.engine.load_item(current_id)

    # 6. Human‑readable summary
    summary = (
        f"🔄 Rewound item **{current_id}** to substate **{target}**.\n"
        f"- Branch: **{item.status.branch}**\n"
        f"- New substate: **{item.status.substate}**\n"
        f"- Approved: **{item.status.approved}**\n"
        f"- Step outputs for this and later substates were deleted."
    )

    # 7. Structured payload
    payload = {
        "rewound": True,
        "item_id": current_id,
        "target_substate": target,
        "new_substate": item.status.substate,
    }

    return payload, summary
