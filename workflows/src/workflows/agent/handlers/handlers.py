# workflows/agent/handlers/handlers.py

from ..context.decorators import updates_context
from ..context.normalization import normalize_item
from ...engine.utils.tracing import agent_trace


# ----------------------------------------------------------------------
# Query Current Item
# ----------------------------------------------------------------------
@agent_trace("handle_query_current_item")
@updates_context(mapping={}, set_current_item=True)
def handle_query_current_item(agent, intent, item_id, resolution_msg):
    if agent.session.last_item_id:
        label = agent.engine.get_item_label(agent.session.last_item_id)
        return {}, f"Your current item is {label}."
    return {}, "You don’t have a current item yet."


# ----------------------------------------------------------------------
# Query Item
# ----------------------------------------------------------------------
@agent_trace("handle_query_item")
@updates_context(mapping={}, set_current_item=True)
def handle_query_item(agent, intent, item_id, resolution_msg):
    if not item_id:
        return {}, resolution_msg

    item = agent.engine.load_item(item_id)
    summary = agent.engine.summarize_item_structured(item, small=False)

    return {}, f"{resolution_msg}\n\nCurrent status:\n{summary}"


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
# Run Next Step
# ----------------------------------------------------------------------
@agent_trace("handle_run_next_step")
@updates_context(mapping={}, set_current_item=True)
def handle_run_next_step(agent, intent, item_id, resolution_msg):
    if not item_id:
        return {}, resolution_msg

    output = agent.engine.run_next_step(item_id)
    item = agent.engine.load_item(item_id)
    label = agent.engine.get_item_label(item_id)

    msg = (
        f"{resolution_msg}\n\n"
        f"Ran step on item {label}.\n"
        f"Summary: {output.summary}\n"
        f"New state: {item.status.branch}/{item.status.substate}, "
        f"approved={item.status.approved}"
    )
    return {}, msg


# ----------------------------------------------------------------------
# Approve Substate
# ----------------------------------------------------------------------
@agent_trace("handle_approve_substate")
@updates_context(mapping={}, set_current_item=True)
def handle_approve_substate(agent, intent, item_id, resolution_msg):
    if not item_id:
        return {}, resolution_msg

    agent.engine.approve_substate(item_id)
    item = agent.engine.load_item(item_id)
    label = agent.engine.get_item_label(item_id)

    msg = (
        f"{resolution_msg}\n\n"
        f"Approved current substate for item {label}.\n"
        f"State: {item.status.branch}/{item.status.substate}, "
        f"approved={item.status.approved}"
    )
    return {}, msg


# ----------------------------------------------------------------------
# Export Item
# ----------------------------------------------------------------------
@agent_trace("handle_export")
@updates_context(mapping={}, set_current_item=True)
def handle_export(agent, intent, item_id, resolution_msg):
    if not item_id:
        return {}, resolution_msg

    agent.engine.export_item(item_id)
    item = agent.engine.load_item(item_id)
    label = agent.engine.get_item_label(item_id)

    msg = (
        f"{resolution_msg}\n\n"
        f"Exported item {label}.\n"
        f"Exported at: {item.exported_at.isoformat() if item.exported_at else 'unknown'}"
    )
    return {}, msg


# ----------------------------------------------------------------------
# Unknown Intent
# ----------------------------------------------------------------------
@agent_trace("handle_unknown_intent")
@updates_context(mapping={})
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
def handle_describe_workflow(agent, intent, item_id, resolution_msg):
    return {}, agent._describe_workflow()


# ----------------------------------------------------------------------
# Show Schema
# ----------------------------------------------------------------------
@agent_trace("handle_show_schema")
@updates_context(mapping={})
def handle_show_schema(agent, intent, item_id, resolution_msg):
    step = intent.step_name
    if step not in agent.engine.definition.step_specs:
        return {}, f"Unknown step: {step}"

    spec = agent.engine.definition.step_specs[step]
    schema = spec.output_schema.schema()

    lines = [f"Schema for {spec.human_name or step}:"]
    for name, field in schema["properties"].items():
        ftype = field.get("type", "unknown")
        lines.append(f"- {name}: {ftype}")

    return {}, "\n".join(lines)


# ----------------------------------------------------------------------
# Explain Schema
# ----------------------------------------------------------------------
@agent_trace("handle_explain_schema")
@updates_context(mapping={})
def handle_explain_schema(agent, intent, item_id, resolution_msg):
    step = intent.step_name
    if step not in agent.engine.definition.step_specs:
        return {}, f"Unknown step: {step}"

    spec = agent.engine.definition.step_specs[step]
    schema = spec.output_schema.schema()

    lines = [f"Explanation of schema for {spec.human_name or step}:"]
    for name, field in schema["properties"].items():
        ftype = field.get("type", "unknown")
        desc = field.get("description", "No description provided.")
        lines.append(f"- Field '{name}' is a(n) {ftype}. {desc}")

    return {}, "\n".join(lines)


# ----------------------------------------------------------------------
# Show Step Output
# ----------------------------------------------------------------------
@agent_trace("handle_show_step_output")
@updates_context(mapping={}, set_current_item=True)
def handle_show_step_output(agent, intent, item_id, resolution_msg):
    step = intent.step_name
    item = agent.engine.get_item(item_id)

    if step not in item.outputs:
        return {}, f"No output found for step {step}"

    output = item.outputs[step]

    lines = [f"Output for {step}:"]
    for k, v in output.items():
        lines.append(f"- {k}: {v}")

    return {}, "\n".join(lines)


# ----------------------------------------------------------------------
# List Step Outputs
# ----------------------------------------------------------------------
@agent_trace("handle_list_step_outputs")
@updates_context(mapping={}, set_current_item=True)
def handle_list_step_outputs(agent, intent, item_id, resolution_msg):
    item = agent.engine.get_item(item_id)
    if not item.outputs:
        return {}, "No step outputs yet."

    lines = ["Step outputs:"]
    for step, output in item.outputs.items():
        lines.append(f"- {step}: {output}")

    return {}, "\n".join(lines)
