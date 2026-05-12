from __future__ import annotations

from typing import Any, Dict, Optional
from pathlib import Path
from ..engine.base_workflow_engine import BaseWorkflowEngine
from ..engine.utils.tracing import agent_trace
from .parsing.intent_parser import WorkflowIntent, parse_intent

from .validation.validator_core import validate_intent

from .resolution.resolver_core import resolve_item_reference
from .session import SessionState
from .stepcontext_commands import StepContextAgentMixin
from .parsing.general_parser import parse_intent_with_llm
from .parsing.parameter_extraction import build_relevant_context_for_intent, extract_parameters_with_llm

from .contract.loader import load_agent_contract
from .ontologies.ontology import WorkflowOntologyRegistry

from .dispatch.dispatcher import dispatch_intent

from .context.decorators import updates_context
from .context.normalization import normalize_step, normalize_item

import json
import datetime

# workflows/agent/agent.py

from .handlers.handlers import (
    handle_list_workflow_items,
    handle_describe_workflow,
    handle_unknown_intent,
    handle_create_item,
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
            "_updated_turn": {
                "items": None,
                "steps": None,
            }
        }

        self.workflow_ontology = WorkflowOntologyRegistry()

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
            "list_workflow_items": handle_list_workflow_items,
            "describe_workflow": handle_describe_workflow,
            "create_item": handle_create_item,
            
            "unknown_intent": handle_unknown_intent,
            
        }

    def save(self):
        """
        Save the entire agent session state into the current item's folder.
        This is future-proof: it serializes the entire session.context dict,
        converting ontology-backed objects into dicts automatically.
        """
        item_id = self.session.last_item_id
        if not item_id:
            return
        
        

        item_dir = self.engine._item_dir(item_id)
        item_dir.mkdir(parents=True, exist_ok=True)

        session_path = item_dir / "session.json"

        def serialize_value(v):
            # Ontology-backed Pydantic model
            if hasattr(v, "dict"):
                return serialize_value(v.dict())

            # datetime → ISO 8601 string
            if isinstance(v, datetime.datetime):
                return v.isoformat()

            # list → serialize each element
            if isinstance(v, list):
                return [serialize_value(x) for x in v]

            # dict → serialize each value
            if isinstance(v, dict):
                return {k: serialize_value(x) for k, x in v.items()}

            # primitive
            return v

        data = {
            "last_item_id": self.session.last_item_id,
            "last_intent": self.session.last_intent,
            "turn_index": self.session.turn_index,
            "context": serialize_value(self.session.context),
        }

        with open(session_path, "w") as f:
            json.dump(data, f, indent=2)

    def _rewind_substate(self, target_substate, item_id=None):
        """
        Rewind the workflow to `target_substate` and delete all step outputs
        for that substate and all substates that come after it.
        """

        # ---------------------------------------------------------
        # Load item
        # ---------------------------------------------------------
        if item_id is None:
            item_id = self.session.last_item_id

        item = self.engine.load_item(item_id)

        branch = item.status.branch
        current = item.status.substate

        # ---------------------------------------------------------
        # Validate workflow path
        # ---------------------------------------------------------
        path = self.engine.definition.workflow_paths.get(branch)
        if not path:
            raise RuntimeError(f"No workflow path defined for branch '{branch}'")

        if target_substate not in path:
            raise ValueError(f"Target substate '{target_substate}' is not valid for branch '{branch}'")

        current_idx = path.index(current)
        target_idx = path.index(target_substate)

        # ---------------------------------------------------------
        # Only rewind backward
        # ---------------------------------------------------------
        if target_idx >= current_idx:
            # Nothing to do (or invalid forward rewind)
            return False

        # ---------------------------------------------------------
        # ⭐ Delete step outputs for target and all future steps
        # ---------------------------------------------------------
        for sub in path[target_idx:]:
            if sub in item.step_outputs:
                del item.step_outputs[sub]

        # ---------------------------------------------------------
        # Perform the rewind
        # ---------------------------------------------------------
        item.status.substate = target_substate

        # Reset approval flags
        item.status.approved = False
        item.status.requires_approval = self.engine.substate_requires_approval(
            branch,
            target_substate
        )

        # Persist
        self.engine.save_item(item)

        return True

    def load(self, item_id=None):
        """
        Load a workflow item and restore the agent session state.
        Automatically rehydrates ontology-backed objects.
        """

        # 1. Determine which item to load
        if item_id is None:
            item_id = self._find_most_recent_item_id()
            if not item_id:
                raise ValueError("No workflow items exist to load.")

        # 2. Load the item from the engine
        item = self.engine.load_item(item_id)

        # 3. Load session.json if present
        session_path = Path(self.engine.base_dir) / item_id / "session.json"
        if not session_path.exists():
            self.session.last_item_id = item_id
            self.session.context = {}
            return item

        with open(session_path, "r") as f:
            data = json.load(f)

        self.session.last_item_id = data.get("last_item_id")
        self.session.last_intent = data.get("last_intent")
        self.session.turn_index = data.get("turn_index", 0)

        # ---------------------------------------------------------
        # 4. Rehydrate ontology-backed objects
        #    (with protection for internal namespaces)
        # ---------------------------------------------------------

        NON_MODEL_CONTEXT_KEYS = {"_updated_turn"}

        def rehydrate_value(key, v):
            # 0. Skip internal namespaces entirely
            if key in NON_MODEL_CONTEXT_KEYS:
                return v

            # 1. If this key corresponds to an ontology session_key
            ots = self.workflow_ontology.find_by_session_key(key)
            if ots:
                model = ots[0].model
                if isinstance(v, list):
                    return [model(**x) for x in v]
                if isinstance(v, dict):
                    return model(**v)
                return v  # primitive fallback

            # 2. Recurse into lists
            if isinstance(v, list):
                return [rehydrate_value(key, x) for x in v]

            # 3. Recurse into dicts
            if isinstance(v, dict):
                return {k: rehydrate_value(k, x) for k, x in v.items()}

            # 4. Primitive
            return v

        restored_context = {}
        for key, value in data.get("context", {}).items():
            restored_context[key] = rehydrate_value(key, value)

        self.session.context = restored_context

        return item

    def _find_most_recent_item_id(self):
        """
        Return the item_id of the most recently modified workflow item directory.
        Filters out system folders like __pycache__ and hidden directories.
        """
        base = Path(self.engine.base_dir)

        if not base.exists():
            return None

        item_dirs = [
            d for d in base.iterdir()
            if d.is_dir()
            and not d.name.startswith(".")          # hidden dirs
            and d.name != "__pycache__"             # Python cache
        ]

        if not item_dirs:
            return None

        item_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        return item_dirs[0].name

    def _get_handler(self, intent_name: str):
        """
        Resolve the handler function for a given intent name.
        Falls back to unknown_intent.
        """
        return self.handler_map.get(intent_name, handle_unknown_intent)

    def _get_base_handler_names(self):

        base_handlers = [
            fn.__name__
            for name, fn in self.handler_map.items()
            if getattr(fn, "_is_base_handler", False)
        ]
        return(base_handlers)

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

            current_step = None
            allowed_handlers = None

            if self.session.last_item_id:
                current_step = self.engine.get_current_step(self.session.last_item_id)

                step_handlers = current_step.allowed_handlers or []

                base_handlers = self._get_base_handler_names()

                allowed_handlers = step_handlers + base_handlers

            print('allowed_handlers')
            print(allowed_handlers)

            # Stage 1: intent classification
            intent = parse_intent_with_llm(
                self.engine,
                self.contract,
                self.handler_map,
                message,
                workflow_description=workflow_desc,
                pending_context=self.session.pending_intent,
                allowed_handlers = allowed_handlers,
            )

            # After Stage 1 classification
            if intent.intent == "idle":
                # No parameter extraction for idle
                if trace:
                    trace.record_llm_intent(intent.to_dict())
                return self._idle_help()


            if intent.intent == "clarify_pending" and self.session.pending_intent is not None:
                return self._handle_pending_message(message, trace)

            """

            # Stage 1.5: derive relevant_context for this intent
            relevant_context = build_relevant_context_for_intent(
                self,
                intent,
            )
            """

            relevant_context = build_relevant_context_for_intent(
                self,
                intent,
            )

            # Stage 2: parameter extraction
            params = extract_parameters_with_llm(
                self.engine,
                self.contract,
                message,
                intent,
                self.session,
                workflow_desc,
                relevant_context=relevant_context,
            )
            intent = intent.with_parameters(params)

            if trace:
                trace.record_llm_intent(intent.to_dict())
        else:
            # Rule-based parsing (non-LLM path)
            intent = parse_intent(message, self.session, self.contract)

        print('intent')
        print(intent)

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


    def _update_session_context_from_step_output(self, step_name):
        """
        After a step completes, load the current item and pull the step output
        directly from the engine's stored step_outputs, then update the agent
        session context using ontology metadata.

        This version implements Option A:
        - Convert dicts into ontology model instances before storing.
        """

        item_id = self.session.last_item_id
        if not item_id:
            return

        # Load the current workflow item
        item = self.engine.load_item(item_id)

        # Get the step output object
        step_output_obj = item.step_outputs.get(step_name)
        if not step_output_obj:
            return

        # The canonical output is in .current
        output = step_output_obj.current
        if not output:
            return

        # For each ontology type registered
        for ot in self.workflow_ontology.all_types():
            session_key = ot.metadata.get("session_key")
            if not session_key:
                continue

            # If this step produced objects for this ontology type
            if session_key in output:
                objects = output[session_key]

                # Normalize to list
                if not isinstance(objects, list):
                    objects = [objects]

                # Convert dicts → Pydantic model instances
                converted = []
                for obj in objects:
                    if isinstance(obj, ot.model):
                        converted.append(obj)
                    else:
                        # obj is a dict → convert to model
                        converted.append(ot.model(**obj))

                # Store in agent session context
                self.session.context[session_key] = converted



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
