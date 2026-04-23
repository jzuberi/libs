from __future__ import annotations

from typing import Any, Dict, Optional

from ..engine.base_workflow_engine import BaseWorkflowEngine
from ..engine.utils.tracing import agent_trace
from .parsing.intent_parser import WorkflowIntent, parse_intent

from .validation.validator_core import validate_intent

from .resolution.resolver_core import resolve_item_reference
from .session import SessionState
from .stepcontext_commands import StepContextAgentMixin
from .parsing.general_parser import parse_intent_with_llm
from .parsing.parameter_extraction import extract_parameters_with_llm

from .contract.loader import load_agent_contract

from .dispatch.dispatcher import dispatch_intent

from .context.decorators import updates_context
from .context.normalization import normalize_step, normalize_item

# workflows/agent/agent.py

from .handlers.handlers import (
    handle_query_current_item,
    handle_query_item,
    handle_list_workflow_items,
    handle_run_next_step,
    handle_approve_substate,
    handle_export,
    handle_unknown_intent,
    handle_describe_workflow,
    handle_show_schema,
    handle_explain_schema,
    handle_show_step_output,
    handle_list_step_outputs,
)

class WorkflowAgent(StepContextAgentMixin):

    def __init__(self, engine: BaseWorkflowEngine):
        self.engine = engine
        self.session = SessionState()
        self.session.turn_index = 0

        self.contract = load_agent_contract()

        # Initialize context namespaces
        self.session.context = {
            "items": [],
            "steps": [],
            "current_item_id": None,
            "current_step_name": None,
            "_updated_turn": {
                "items": None,
                "steps": None,
            }
        }


        # --------------------------------------------------------------
        # Populate steps namespace using engine definition
        # --------------------------------------------------------------
        step_specs = self.engine.definition.step_specs.values()

        self.session.context["steps"] = [
            normalize_step(spec) for spec in step_specs
        ]

        self.session.context["_updated_turn"]["steps"] = self.session.turn_index

        # Instance-level handler map (safe for overrides)
        self.handler_map = {
            "query_current_item": handle_query_current_item,
            "query_item": handle_query_item,
            "list_workflow_items": handle_list_workflow_items,
            "run_next_step": handle_run_next_step,
            "approve_substate": handle_approve_substate,
            "export_item": handle_export,
            "unknown_intent": handle_unknown_intent,
            "describe_workflow": handle_describe_workflow,
            "show_schema": handle_show_schema,
            "explain_schema": handle_explain_schema,
            "show_step_output": handle_show_step_output,
            "list_step_outputs": handle_list_step_outputs,
        }

    def _get_handler(self, intent_name: str):
        """
        Resolve the handler function for a given intent name.
        Falls back to unknown_intent.
        """
        return self.handler_map.get(intent_name, handle_unknown_intent)



    def _idle_help(self):
        return (
            "I’m not sure what you want to do. "
            "You can ask me to list items, run the next step, approve, export, "
            "show step outputs, or show schemas."
        )

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------
    def _handle_pending_message(self, message: str, trace=None) -> str:
        """
        Deterministic pending-intent clarification.
        No LLM. No intent parsing.
        The user is answering a clarification question for a PendingIntent.
        """

        text = message.strip().lower()
        pending = self.session.pending_intent

        if pending is None:
            return "There is no pending action to clarify."

        # --------------------------------------------------------------
        # 1. Cancel keywords
        # --------------------------------------------------------------
        CANCEL_KEYWORDS = {"cancel", "never mind", "nevermind", "stop", "forget it"}
        if any(k in text for k in CANCEL_KEYWORDS):
            self._clear_pending()
            return "Okay — canceled. What would you like to do next?"

        # --------------------------------------------------------------
        # 2. Try to resolve each missing parameter deterministically
        # --------------------------------------------------------------
        resolved_params = {}

        for param in pending.missing:
            # Special-case: item_id resolution
            if param == "item_id":
                item_id, resolution_msg = resolve_item_reference(
                    self.engine,
                    message,
                    self.session,
                )
                if item_id is None:
                    # Still missing — return resolver's message
                    return resolution_msg
                resolved_params["item_id"] = item_id
                continue

            # Special-case: step_name resolution
            if param == "step_name":
                step_name, resolution_msg = resolve_step_reference(
                    self.engine,
                    message,
                    self.session,
                )
                if step_name is None:
                    return resolution_msg
                resolved_params["step_name"] = step_name
                continue

            # Generic parameter: treat message as the value
            resolved_params[param] = message.strip()

        # --------------------------------------------------------------
        # 3. Merge resolved parameters into pending intent
        # --------------------------------------------------------------
        for k, v in resolved_params.items():
            pending.parameters[k] = v

        # Remove resolved fields from missing list
        pending.missing = [m for m in pending.missing if m not in resolved_params]

        # --------------------------------------------------------------
        # 4. If still missing fields → ask again
        # --------------------------------------------------------------
        if pending.missing:
            missing_str = ", ".join(pending.missing)
            return f"I still need {missing_str}."

        # 5. All parameters resolved → dispatch the completed intent
        intent = WorkflowIntent(
            intent=pending.intent,
            parameters=pending.parameters
        )

        self._clear_pending()
        return dispatch_intent(self, intent, trace)


    @agent_trace("handle_message")
    def handle_message(self, message: str, trace=None) -> str:

        self.session.turn_index += 1

        if trace:
            trace.set_prompt(message)

        self.engine.refresh()

        print('handle message session:')
        print(self.session)

        # --------------------------------------------------------------
        # 1. Normal intent parsing (Stage 1 + Stage 2 when LLM is present)
        # --------------------------------------------------------------
        if hasattr(self.engine, "agent_llm") and self.engine.agent_llm is not None:

            workflow_desc = self._describe_workflow()

            # Stage 1: intent classification
            intent = parse_intent_with_llm(
                self.engine,
                self.contract,
                message,
                workflow_description=workflow_desc,
                pending_context=self.session.pending_intent,
            )

            # If this is a clarification for a pending intent → deterministic handler
            if intent.intent == "clarify_pending" and self.session.pending_intent is not None:
                return self._handle_pending_message(message, trace)

            # Stage 2: parameter extraction
            params = extract_parameters_with_llm(
                self.engine,
                self.contract,
                message,
                intent,
                self.session,
                workflow_desc,
            )
            intent = intent.with_parameters(params)

            if trace:
                trace.record_llm_intent(intent.to_dict())
        else:
            # Rule-based parsing (non-LLM path)
            intent = parse_intent(message, self.session, self.contract)

        print('intent')
        print(intent)

        # Rule-based fallback for idle (non-LLM safety net)
        if intent.intent == "idle":
            rb_intent = parse_intent(message, self.session, self.contract)
            if trace:
                trace.record_rule_intent(rb_intent.to_dict())
            intent = rb_intent

        # --------------------------------------------------------------
        # 2. Special intent: query_current_item
        # --------------------------------------------------------------
        if intent.intent == "query_current_item":
            if self.session.last_item_id:
                label = self.engine.get_item_label(self.session.last_item_id)
                return f"Your current item is {label}."
            return "You don’t have a current item yet."

        # --------------------------------------------------------------
        # 3. Idle handling
        # --------------------------------------------------------------
        if intent.intent == "idle":
            return self._idle_help()

        # --------------------------------------------------------------
        # 4. Dispatch the intent (resolve → maybe pending → handler)
        # --------------------------------------------------------------
        return dispatch_intent(self, intent, trace)



    # ------------------------------------------------------------------
    # Pending helpers
    # ------------------------------------------------------------------

    def _clear_pending(self):
        self.session.pending_intent = None

    def _describe_workflow(self) -> str:
            definition = self.engine.definition
            steps = definition.step_specs
            path = definition.workflow_paths.get("default", [])

            lines = []
            lines.append("Here's an overview of the workflow:\n")

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
