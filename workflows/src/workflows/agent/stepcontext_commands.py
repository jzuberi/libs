from __future__ import annotations

from typing import Optional

from ..engine.models import WorkflowItem
from ..engine.workflow_definition import WorkflowDefinition


class StepContextAgentMixin:
    """
    Adds StepContext-aware commands to WorkflowAgent.
    """

    # ---------------------------------------------------------
    # Show step output
    # ---------------------------------------------------------

    def handle_show_step_output(self, intent, item_id, resolution_msg):
        step = intent.step_name
        item = self.engine.get_item(item_id)

        if step not in item.step_outputs:
            return f"No output found for step '{step}'."

        record = item.step_outputs[step]
        return (
            f"Step: {step}\n"
            f"Summary: {record.current or record.raw}\n"
            f"Edits: {len(record.edits)} edits\n"
        )

    # ---------------------------------------------------------
    # List all step outputs
    # ---------------------------------------------------------

    def handle_list_step_outputs(self, intent, item_id, resolution_msg):
        item = self.engine.get_item(item_id)

        if not item.step_outputs:
            return "This item has no step outputs yet."

        lines = []
        for step, record in item.step_outputs.items():
            lines.append(
                f"- {step}: current={bool(record.current)}, edits={len(record.edits)}"
            )

        return "Step outputs:\n" + "\n".join(lines)

    # ---------------------------------------------------------
    # Edit step output
    # ---------------------------------------------------------

    def handle_edit_step_output(self, intent, item_id, resolution_msg):
        step = intent.step_name
        new_text = intent.edit_text

        item = self.engine.get_item(item_id)

        if step not in item.step_outputs:
            return f"No output found for step '{step}'."

        record = item.step_outputs[step]
        record.edits.append(new_text)
        record.current = new_text

        self.engine.save_item(item)

        return f"Updated step '{step}' output."

    # ---------------------------------------------------------
    # Show schema
    # ---------------------------------------------------------

    def handle_show_schema(self, intent, item_id, resolution_msg):
        step = intent.step_name
        spec = self.engine.definition.step_specs.get(step)

        if not spec or not spec.output_schema:
            return f"No schema defined for step '{step}'."

        schema = spec.output_schema.model_json_schema()
        return f"Schema for {step}:\n{schema}"

    # ---------------------------------------------------------
    # Explain schema
    # ---------------------------------------------------------

    def handle_explain_schema(self, intent, item_id, resolution_msg):
        step = intent.step_name
        spec = self.engine.definition.step_specs.get(step)

        if not spec or not spec.output_schema:
            return f"No schema defined for step '{step}'."

        schema = spec.output_schema.model_json_schema()

        lines = [f"Schema explanation for {step}:"]
        for field, info in schema.get("properties", {}).items():
            ftype = info.get("type", "unknown")
            desc = info.get("description", "")
            lines.append(f"- {field}: {ftype} — {desc}")

        return "\n".join(lines)
