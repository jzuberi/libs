from __future__ import annotations

from typing import Any, Dict, Optional

from ..engine.base_workflow_engine import BaseWorkflowEngine
from ..engine.utils.tracing import agent_trace
from .intent_parser import WorkflowIntent, parse_intent
from .intent_validator import validate_intent

from .resolvers import resolve_item_reference
from .session import SessionState
from .stepcontext_commands import StepContextAgentMixin
from .llm_intent_parser import parse_intent_with_llm


    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------
class WorkflowAgent(StepContextAgentMixin):
    def __init__(self, engine: BaseWorkflowEngine):
        self.engine = engine
        self.session = SessionState()

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

    @agent_trace("handle_message")
    def handle_message(self, message: str, trace=None) -> str:
        
        if trace:
            trace.set_prompt(message)

        self.engine.refresh()

        # --------------------------------------------------------------
        # 1. If there is a pending intent, interpret this message
        #    as a possible response to that clarification.
        # --------------------------------------------------------------
        if self.session.pending_intent is not None and hasattr(self.engine, "agent_llm") and self.engine.agent_llm is not None:
            intent = parse_intent_with_llm(
                self.engine,
                message,
                pending_context=self.session.pending_intent,
            )
            if trace:
                trace.record_llm_intent(intent.to_dict())

            # Handle pending-related intents
            if intent.intent == "confirm_pending":
                return self._run_pending_with_last_item(trace=trace)

            if intent.intent == "reject_pending":
                # Ask user for item id next
                self.session.pending_resolution = "awaiting_item_id"
                return "Okay — which item should I use?"

            if intent.intent == "cancel_pending":
                self._clear_pending()
                return "Okay — canceled. What would you like to do next?"

            if intent.intent == "provide_item_id":
                return self._run_pending_with_item(intent.item_id, trace=trace)

            # If LLM decided this is a normal intent, clear pending and fall through
            self._clear_pending()
            # continue into normal flow below (no return)

        # --------------------------------------------------------------
        # 2. Normal intent parsing (LLM first, then rule-based fallback)
        # --------------------------------------------------------------
        if hasattr(self.engine, "agent_llm") and self.engine.agent_llm is not None:
            intent = parse_intent_with_llm(self.engine, message)
            if trace:
                trace.record_llm_intent(intent.to_dict())
        else:
            intent = parse_intent(message, self.session)

        if intent.intent == "idle":
            rb_intent = parse_intent(message, self.session)
            if trace:
                trace.record_rule_intent(rb_intent.to_dict())
            intent = rb_intent

        intent = validate_intent(self.engine, intent)

        if trace:
            trace.record_final_intent(intent.intent)

        self.session.last_intent = intent.intent

        # --------------------------------------------------------------
        # 3. Special intent: query_current_item
        # --------------------------------------------------------------
        if intent.intent == "query_current_item":
            if self.session.last_item_id:
                label = self.engine.get_item_label(self.session.last_item_id)
                
                return f"Your current item is {label}."
            return "You don’t have a current item yet."

        # --------------------------------------------------------------
        # 4. Idle handling
        # --------------------------------------------------------------
        if intent.intent == "idle":
            return (
                "I’m not sure what you want to do. "
                "You can ask me to list items, run the next step, approve, export, "
                "show step outputs, or show schemas."
            )

        # --------------------------------------------------------------
        # 5. Resolve item reference (if any)
        # --------------------------------------------------------------
        item_id, resolution_msg = resolve_item_reference(
            self.engine, intent.item_id, self.session
        )

        if item_id is not None:
            label = self.engine.get_item_label(item_id)
        else:
            label = None

        if trace:
            trace.record_item_resolution({
                "item_id": item_id,
                "item_label": label,
                "resolution_msg": resolution_msg
            })


        # --------------------------------------------------------------
        # 6. If item is required but missing, possibly enter pending mode
        # --------------------------------------------------------------
        if item_id is None and intent.intent not in (
            "query_workflow",
            "idle",
            "describe_workflow",
            "list_step_outputs",
        ):
            # If we have a last_item_id, ask if user meant that
            if self.session.last_item_id is not None:

                self.session.pending_intent = intent
                self.session.pending_resolution = None

                last_item = self.engine.get_item(self.session.last_item_id)

                return (
                    f"I couldn’t determine the item. "
                    f"Did you mean to use the last item ({last_item.label})?"
                )

            # No last item to fall back to — keep existing behavior
            return (
                resolution_msg
                or "You haven’t selected an item yet. Try: 'list items'."
            )

        # --------------------------------------------------------------
        # 7. Update session with current item
        # --------------------------------------------------------------
        if item_id:
            self.session.last_item_id = item_id

        # --------------------------------------------------------------
        # 8. Dispatch to handler
        # --------------------------------------------------------------
        handler = self._get_handler(intent.intent)
        response = handler(intent, item_id, resolution_msg)

        if trace:
            trace.record_agent_response(response)

        return response

    # ------------------------------------------------------------------
    # Pending helpers
    # ------------------------------------------------------------------

    def _clear_pending(self):
        self.session.pending_intent = None
        self.session.pending_resolution = None

    def _run_pending_with_last_item(self, trace=None) -> str:
        if not self.session.last_item_id or not self.session.pending_intent:
            self._clear_pending()
            return "I don’t have a previous item to use anymore."

        intent = self.session.pending_intent
        item_id = self.session.last_item_id
        self._clear_pending()

        handler = self._get_handler(intent.intent)
        response = handler(intent, item_id, None)

        if trace:
            trace.record_agent_response(response)

        return response

    def _run_pending_with_item(self, item_id: str | None, trace=None) -> str:
        if not item_id:
            return "I didn’t catch which item you meant. Please provide an item id."

        if not self.session.pending_intent:
            return "There is no pending action to apply this item to."

        intent = self.session.pending_intent
        self._clear_pending()

        handler = self._get_handler(intent.intent)
        response = handler(intent, item_id, None)

        if trace:
            trace.record_agent_response(response)

        # Also update last_item_id since user explicitly chose it
        self.session.last_item_id = item_id

        return response

    # ------------------------------------------------------------------
    # Handler dispatch
    # ------------------------------------------------------------------

    def _get_handler(self, intent_name: str):
        handlers = {
            "query_item": self.handle_query_item,
            "query_workflow": self.handle_query_workflow,
            "run_next_step": self.handle_run_next_step,
            "approve_substate": self.handle_approve_substate,
            "export": self.handle_export,
            "show_step_output": self.handle_show_step_output,
            "edit_step_output": self.handle_edit_step_output,
            "list_step_outputs": self.handle_list_step_outputs,
            "show_schema": self.handle_show_schema,
            "explain_schema": self.handle_explain_schema,
            "describe_workflow": self.handle_describe_workflow,
            "query_current_item": self.handle_query_current_item,
        }
        return handlers.get(intent_name, self.handle_unknown_intent)

    # ------------------------------------------------------------------
    # New handler
    # ------------------------------------------------------------------

    @agent_trace("handle_query_current_item")
    def handle_query_current_item(self, intent, item_id, resolution_msg):
        if self.session.last_item_id:
            label = self.engine.get_item_label(self.session.last_item_id)

            return f"Your current item is {label}."
        return "You don’t have a current item yet."

    @agent_trace("handle_query_item")
    def handle_query_item(self, intent, item_id, resolution_msg):
        if not item_id:
            return resolution_msg

        item = self.engine.load_item(item_id)
        summary = self.engine.summarize_item_structured(item, small=False)
        return f"{resolution_msg}\n\nCurrent status:\n{summary}"

    @agent_trace("handle_query_workflow")
    def handle_query_workflow(self, intent, item_id, resolution_msg):
        items = self.engine.list_items()
        if not items:
            return "There are no items in this workflow yet."

        lines = []
        for item in items:
            label = item.label or self.engine.get_item_label(item.id)
            lines.append(
                f"- {label}: {item.type} | "
                f"{item.status.branch}/{item.status.substate} | "
                f"approved={item.status.approved}"
            )

        return "Here are the current items:\n" + "\n".join(lines)

    @agent_trace("handle_run_next_step")
    def handle_run_next_step(self, intent, item_id, resolution_msg):
        if not item_id:
            return resolution_msg

        output = self.engine.run_next_step(item_id)
        item = self.engine.load_item(item_id)
        label = self.engine.get_item_label(item_id)

        return (
            f"{resolution_msg}\n\n"
            f"Ran step on item {label}.\n"
            f"Summary: {output.summary}\n"
            f"New state: {item.status.branch}/{item.status.substate}, "
            f"approved={item.status.approved}"
        )

    @agent_trace("handle_approve_substate")
    def handle_approve_substate(self, intent, item_id, resolution_msg):
        if not item_id:
            return resolution_msg

        self.engine.approve_substate(item_id)
        item = self.engine.load_item(item_id)
        label = self.engine.get_item_label(item_id)

        return (
            f"{resolution_msg}\n\n"
            f"Approved current substate for item {label}.\n"
            f"State: {item.status.branch}/{item.status.substate}, "
            f"approved={item.status.approved}"
        )

    @agent_trace("handle_export")
    def handle_export(self, intent, item_id, resolution_msg):
        if not item_id:
            return resolution_msg

        self.engine.export_item(item_id)
        item = self.engine.load_item(item_id)
        label = self.engine.get_item_label(item_id)
        return (
            f"{resolution_msg}\n\n"
            f"Exported item {label}.\n"
            f"Exported at: {item.exported_at.isoformat() if item.exported_at else 'unknown'}"
        )

    @agent_trace("handle_unknown_intent")
    def handle_unknown_intent(self, intent, item_id, resolution_msg):
        return (
            "I couldn’t map that to a workflow action. "
            "Try asking me to list items, run the next step, approve, export, "
            "show step outputs, or show schemas."
        )

    @agent_trace("handle_describe_workflow")
    def handle_describe_workflow(
        self,
        intent: WorkflowIntent,
        item_id: Optional[str],
        resolution_msg: str,
    ) -> str:
        return self._describe_workflow()

    def _describe_workflow(self) -> str:
        definition = self.engine.definition
        steps = definition.step_specs
        path = definition.workflow_paths.get("default", [])

        lines = []
        lines.append("Here’s an overview of the workflow:\n")

        for step_name in path:
            spec = steps[step_name]

            human = spec.human_name or step_name
            desc = spec.description or "No description provided."
            consumes = spec.consumes or []
            produces = spec.produces or []
            hints = spec.agent_hints or ""

            lines.append(f" {human} ('{step_name}')")
            lines.append(f"- What it does: {desc}")

            if consumes:
                lines.append(f"- Depends on: {', '.join(consumes)}")
            if produces:
                lines.append(f"- Produces: {', '.join(produces)}")
            if hints:
                lines.append(f"- Agent hints: {hints}")

            lines.append("")  # spacing

        return "\n".join(lines)

    @agent_trace("handle_show_schema")
    def handle_show_schema(self, intent, item_id, resolution_msg):
        step = intent.step_name
        if step not in self.engine.definition.step_specs:
            return f"Unknown step: {step}"

        spec = self.engine.definition.step_specs[step]
        schema = spec.output_schema.schema()

        lines = []
        lines.append(f"Schema for {spec.human_name or step}:")
        for name, field in schema["properties"].items():
            ftype = field.get("type", "unknown")
            lines.append(f"- {name}: {ftype}")

        return "\n".join(lines)


    @agent_trace("handle_explain_schema")
    def handle_explain_schema(self, intent, item_id, resolution_msg):
        step = intent.step_name
        if step not in self.engine.definition.step_specs:
            return f"Unknown step: {step}"

        spec = self.engine.definition.step_specs[step]
        schema = spec.output_schema.schema()

        lines = []
        lines.append(f"Explanation of schema for {spec.human_name or step}:")
        for name, field in schema["properties"].items():
            ftype = field.get("type", "unknown")
            desc = field.get("description", "No description provided.")
            lines.append(f"- Field '{name}' is a(n) {ftype}. {desc}")

        return "\n".join(lines)

    @agent_trace("handle_show_step_output")
    def handle_show_step_output(self, intent, item_id, resolution_msg):
        step = intent.step_name
        item = self.engine.get_item(item_id)
        
        

        if step not in item.outputs:
            return f"No output found for step {step}"

        output = item.outputs[step]

        lines = []
        lines.append(f"Output for {step}:")
        for k, v in output.items():
            lines.append(f"- {k}: {v}")

        return "\n".join(lines)


    @agent_trace("handle_list_step_outputs")
    def handle_list_step_outputs(self, intent, item_id, resolution_msg):
        item = self.engine.get_item(item_id)
        if not item.outputs:
            return "No step outputs yet."

        lines = ["Step outputs:"]
        for step, output in item.outputs.items():
            lines.append(f"- {step}: {output}")

        return "\n".join(lines)


