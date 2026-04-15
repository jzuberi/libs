from __future__ import annotations

import uuid, json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional



from .models import (
    CandidateResolutionResult,
    TraceLevel,
    WorkflowItem,
    WorkflowStatus,
    WorkflowStepInput,
    WorkflowStepOutput,
)
from .workflow_definition import WorkflowDefinition
from .utils.io import EngineStorageMixin
from .utils.snapshot import build_context_snapshot


class BaseWorkflowEngine(EngineStorageMixin, ABC):
    """
    Domain-agnostic workflow engine with per-item JSON persistence.

    Subclasses:
      - Provide a WorkflowDefinition
      - Implement domain-specific summarization
      - Implement domain-specific export behavior
    """

    def __init__(
        self, 
        definition: WorkflowDefinition, 
        base_dir: str, 
        approval_requirements=None, 
        agent_llm=None, 
        label_fn=None
        ):

        self.definition = definition
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        

        self.agent_llm = agent_llm
        self.label_fn = label_fn or self.default_label_fn

        self.approval_requirements = approval_requirements or {}

        # In-memory index for quick access; canonical state lives on disk.
        self._items: Dict[str, WorkflowItem] = {}
        self._load_existing_items()

    def attach_trace(self, trace):
        self.trace = trace


    def get_item(self, item_id: str):
        """Return a WorkflowItem by ID. Used by agent StepContext commands."""
        return self.load_item(item_id)


    def load_typed_step_output(self, item_id: str, step_name: str):
        item = self.load_item(item_id)
        step_spec = self.definition.step_specs.get(step_name)

        artifact = self.load_step_output(item_id, step_name)

        if step_spec and step_spec.output_schema:
            return step_spec.output_schema.model_validate(artifact)

        return artifact

    def edit_step_output(self, item_id: str, step_name: str, edits: dict):
        """
        Apply user edits to a step's output with schema validation,
        versioned snapshots, and item.step_outputs registry updates.
        """
        # -------------------------------------------------------------
        # Load item + step spec
        # -------------------------------------------------------------
        item = self.load_item(item_id)
        step_spec = self.definition.step_specs.get(step_name)
        if step_spec is None:
            raise ValueError(f"No step spec found for '{step_name}'")

        item_dir = self.base_dir / item_id
        artifact_dir = item_dir / "artifacts"
        artifact_dir.mkdir(exist_ok=True)

        # -------------------------------------------------------------
        # 1. Locate raw artifact
        # -------------------------------------------------------------
        raw_path = artifact_dir / f"{step_name}_raw.json"
        if not raw_path.exists():
            raise FileNotFoundError(f"No raw artifact found for step '{step_name}'")

        # -------------------------------------------------------------
        # 2. Load current materialized state (raw or current)
        # -------------------------------------------------------------
        current_path = artifact_dir / f"{step_name}_current.json"
        if current_path.exists():
            current = self._read_json(current_path)
        else:
            current = self._read_json(raw_path)

        # -------------------------------------------------------------
        # 3. Merge edits into current state
        # -------------------------------------------------------------
        updated = {**current, **edits}

        # -------------------------------------------------------------
        # 4. Schema validation (if step defines a schema)
        # -------------------------------------------------------------
        if step_spec.output_schema:
            try:
                validated = step_spec.output_schema.model_validate(updated)
                updated = validated.model_dump()
            except Exception as e:
                raise ValueError(
                    f"Invalid edit for step '{step_name}': {e}"
                )

        # -------------------------------------------------------------
        # 5. Create versioned snapshot path
        # -------------------------------------------------------------
        i = 0
        while True:
            snapshot_path = artifact_dir / f"{step_name}_edit_{i}.json"
            if not snapshot_path.exists():
                break
            i += 1

        # -------------------------------------------------------------
        # 6. Write snapshot + updated current state
        # -------------------------------------------------------------
        self._write_json(snapshot_path, edits)
        self._write_json(current_path, updated)

        # -------------------------------------------------------------
        # 7. Update item.step_outputs registry
        # -------------------------------------------------------------
        record = item.step_outputs.get(step_name)
        if record is None:
            # Create a new record if missing (should not happen normally)
            record = StepOutputRecord(raw=str(raw_path))
            item.step_outputs[step_name] = record

        record.current = str(current_path)
        record.edits.append(str(snapshot_path))

        # -------------------------------------------------------------
        # 8. Update item metadata + save
        # -------------------------------------------------------------
        item.updated_at = datetime.utcnow()
        self.save_item(item)

        # -------------------------------------------------------------
        # 9. Log the edit
        # -------------------------------------------------------------
        self._log_trace(
            item_id=item_id,
            branch=item.status.branch,
            substate=item.status.substate,
            step_name=step_name,
            actor="user",
            artifact={
                "snapshot_path": str(snapshot_path),
                "current_path": str(current_path),
            },
            details={
                "event": "user_edit",
                "edits": edits,
                "snapshot_index": i,
            },
            trace_level=TraceLevel.INFO,
        )

        return updated



    # -------------------------------------------------------------------------
    # Item lifecycle
    # -------------------------------------------------------------------------

    def _advance_substate(self, item: WorkflowItem):
        branch = item.status.branch
        substate = item.status.substate

        # 1. Check approval requirement
        if self.substate_requires_approval(branch, substate):
            if not item.status.approved:
                return  # do not advance

        # 2. Normal advancement
        path = self.definition.workflow_paths[branch]
        idx = path.index(substate)

        if idx + 1 < len(path):
            next_sub = path[idx + 1]
            item.status.substate = next_sub
            item.status.approved = False
            self.save_item(item)


    def substate_requires_approval(self, branch: str, substate: str) -> bool:
        return (
            self.approval_requirements
            .get(branch, {})
            .get(substate, False)
        )


    def refresh(self):
        self._items.clear()
        self._load_existing_items()

    def get_item_label(self, item_id):
        item = self.load_item(item_id)
        return item.label


    def list_items_with_labels(self):
        items = self.list_items()
        return [(item.id, item.label) for item in items]

    def find_item_by_label(self, label: str):
        """
        Optional helper: exact match.
        You can extend this to fuzzy matching later.
        """
        for item in self.list_items():
            if item.label.lower() == label.lower():
                return item.id

        return None




    def default_label_fn(self, item: WorkflowItem) -> str:
        """
        Default human-readable label for items.
        Uses created_at timestamp in a friendly format.
        """
        dt = item.created_at
        return dt.strftime("Item from %b %d, %Y at %I:%M %p")




    def _load_existing_items(self):
        if not self.base_dir.exists():
            return

        for item_dir in self.base_dir.iterdir():
            if not item_dir.is_dir():
                continue

            item_json = item_dir / "item.json"
            status_json = item_dir / "status.json"

            if not item_json.exists() or not status_json.exists():
                continue

            item = WorkflowItem.from_json(item_json.read_text())
            status = WorkflowStatus.from_json(status_json.read_text())

            item.status = status
            self._items[item.id] = item


    def create_item(
        self,
        description: str,
        type: str,
        metadata: Optional[Dict[str, Any]] = None,
        style: Optional[Dict[str, Any]] = None,
        branch: str = "default",
        initial_substate: str = "start",
    ) -> WorkflowItem:

        item_id = str(uuid.uuid4())
        status = WorkflowStatus(branch=branch, substate=initial_substate)
        item = WorkflowItem(
            id=item_id,
            description=description,
            type=type,
            metadata=metadata or {},
            style=style or {},
            status=status,
        )

        item.label = self.label_fn(item)

        self._items[item_id] = item

        # Persist immediately
        self._write_item(item)
        self._write_status(item)

        # Log creation
        self._log_trace(
            item_id=item_id,
            branch=branch,
            substate=initial_substate,
            step_name="create_item",
            actor="engine",
            artifact=None,
            details={"description": description, "type": type},
            trace_level=TraceLevel.AUDIT,
        )

        return item

    def load_item(self, item_id: str) -> WorkflowItem:
        if hasattr(self, "trace") and self.trace:
            self.trace.record_engine_call("load_item", args={"item_id": item_id})
        return self._items[item_id]


    def save_item(self, item: WorkflowItem) -> None:
        item.updated_at = datetime.utcnow()
        self._items[item.id] = item
        self._write_item(item)
        self._write_status(item)

    def list_items(self, filters: Optional[Dict[str, Any]] = None) -> List[WorkflowItem]:
        items = list(self._items.values())
        if not filters:
            return items

        def matches(item: WorkflowItem) -> bool:
            for key, value in filters.items():
                if key == "branch" and item.status.branch != value:
                    return False
                if key == "substate" and item.status.substate != value:
                    return False
                if key == "approved" and item.status.approved != value:
                    return False
                if key == "exported" and ((item.exported_at is not None) != bool(value)):
                    return False
            return True

        return [i for i in items if matches(i)]

    # -------------------------------------------------------------------------
    # Workflow execution
    # -------------------------------------------------------------------------

    def next_action_for_item(self, item_id: str, small: bool = False) -> Dict[str, Any]:
        item = self.load_item(item_id)
        status = item.status

        step_spec = self.definition.step_specs.get(status.substate)
        ready = step_spec is not None

        return {
            "item_id": item.id,
            "branch": status.branch,
            "substate": status.substate,
            "approved": item.status.approved,
            "ready": ready,
            "needs_approval": not item.status.approved and ready,
            "summary": self.summarize_item_structured(item, small=small),
        }

    def load_step_output(self, item_id: str, step_name: str):
        """
        Load the materialized current output of a step.
        Falls back to raw output if no edits exist.
        """
        artifact_dir = self.base_dir / item_id / "artifacts"
        current_path = artifact_dir / f"{step_name}_current.json"
        raw_path = artifact_dir / f"{step_name}_raw.json"

        if current_path.exists():
            return self._read_json(current_path)
        if raw_path.exists():
            return self._read_json(raw_path)

        raise FileNotFoundError(f"No output found for step '{step_name}'")

    def run_next_step(self, item_id: str, context: Optional[Dict[str, Any]] = None) -> WorkflowStepOutput:
        # Load item and determine current step
        item = self.load_item(item_id)
        status = item.status
        step_spec = self.definition.step_specs.get(status.substate)

        if step_spec is None:
            raise ValueError(f"No step spec for substate '{status.substate}'")

        # PRE-STEP APPROVAL CHECK
        if self.substate_requires_approval(status.branch, status.substate):
            if not item.status.approved:
                raise RuntimeError(
                    f"Substate '{status.substate}' requires approval before running the next step."
                )

        step_input = WorkflowStepInput(item=item, engine=self, context=context or {})

        # STEP START TRACE
        self._log_trace(
            item_id=item.id,
            branch=status.branch,
            substate=status.substate,
            step_name=step_spec.name,
            actor="engine",
            artifact=None,
            details={"event": "step_start"},
            trace_level=TraceLevel.INFO,
        )

        # Execute step function
        output = step_spec.fn(step_input)

        # Validate output via schema
        validated_artifact = self._validate_step_output(step_spec, output)
        output.artifact = validated_artifact

        # ---------------------------------------------------------
        # STORE STEP OUTPUT (THIS IS THE CRITICAL PART)
        # ---------------------------------------------------------
        item.step_outputs[step_spec.name] = StepOutputRecord(
            raw=output.artifact,
            current=output.artifact,
            edits=[],
            schema_name=(
                step_spec.output_schema.__name__
                if step_spec.output_schema else None
            )
        )
        # ---------------------------------------------------------

        # Save raw artifact to disk
        artifact_dir = self.base_dir / item_id / "artifacts"
        artifact_dir.mkdir(exist_ok=True)
        raw_path = artifact_dir / f"{step_spec.name}_raw.json"
        self._write_json(raw_path, output.artifact)

        # Determine next substate
        next_substate = output.next_substate
        if next_substate is None:
            next_substate = self.definition.get_default_next_substate(
                status.branch,
                status.substate
            )

        # Move to next substate
        if next_substate is not None:
            item.status.substate = next_substate

        # Determine approval requirement
        requires_approval = self.substate_requires_approval(
            item.status.branch,
            item.status.substate
        )

        # Set approval flag
        item.status.approved = not requires_approval

        # Persist item
        self.save_item(item)

        # STEP END TRACE
        self._log_trace(
            item_id=item.id,
            branch=item.status.branch,
            substate=item.status.substate,
            step_name=step_spec.name,
            actor="engine",
            artifact=output.artifact,
            details={"event": "step_end", **(output.details or {})},
            trace_level=TraceLevel.INFO,
        )

        return output



    def approve_substate(self, item_id: str) -> None:
        item = self.load_item(item_id)
        item.status.approved = True
        self.save_item(item)

        self._log_trace(
            item_id=item.id,
            branch=item.status.branch,
            substate=item.status.substate,
            step_name="approve_substate",
            actor="user",
            artifact=None,
            details={"approved": True},
            trace_level=TraceLevel.AUDIT,
        )

    def export_item(self, item_id: str) -> None:
        item = self.load_item(item_id)
        self._export_item_impl(item)
        item.exported_at = datetime.utcnow()
        self.save_item(item)

        self._log_trace(
            item_id=item.id,
            branch=item.status.branch,
            substate=item.status.substate,
            step_name="export_item",
            actor="engine",
            artifact=None,
            details={"exported_at": item.exported_at.isoformat()},
            trace_level=TraceLevel.AUDIT,
        )

    # -------------------------------------------------------------------------
    # Abstracts for subclasses
    # -------------------------------------------------------------------------

    @abstractmethod
    def summarize_item_structured(self, item: WorkflowItem, small: bool = False) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def _export_item_impl(self, item: WorkflowItem) -> None:
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Candidate resolution (stub)
    # -------------------------------------------------------------------------

    def resolve_candidates(self, query: str) -> CandidateResolutionResult:
        candidates = []
        for item in self._items.values():
            score = 0.0
            if item.id == query:
                score = 1.0
            elif query.lower() in item.description.lower():
                score = 0.7
            if score > 0:
                candidates.append((item, score))

        candidates.sort(key=lambda t: t[1], reverse=True)
        chosen_id = candidates[0][0].id if candidates else None
        return CandidateResolutionResult(candidates=candidates, chosen_id=chosen_id)

    # -------------------------------------------------------------------------
    # Trace logging → JSONL
    # -------------------------------------------------------------------------
    def _log_trace(
        self,
        item_id: Optional[str],
        branch: Optional[str],
        substate: Optional[str],
        step_name: str,
        actor: str,
        artifact: Any,
        details: Dict[str, Any],
        trace_level: TraceLevel,
        trace_type: str = 'workflow'
    ) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "trace_type": trace_type,
            "item_id": item_id,
            "branch": branch,
            "substate": substate,
            "step_name": step_name,
            "actor": actor,
            "artifact": artifact,
            "details": details,
            "trace_level": trace_level.value,
        }

        # Agent-level events → global trace
        if trace_type == "agent":
            agent_dir = self.base_dir / "agent"
            agent_dir.mkdir(parents=True, exist_ok=True)

            global_path = agent_dir / "trace.jsonl"
            with global_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            return


        # Item-level events → per-item trace
        self._append_progress(item_id, entry)


    # -------------------------------------------------------------------------
    # Context snapshot
    # -------------------------------------------------------------------------

    def build_context_snapshot(
        self,
        focused_item_id: Optional[str] = None,
        max_chars: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Deterministic, JSON-serializable snapshot of engine state for LLM steps.
        """
        return build_context_snapshot(self, focused_item_id=focused_item_id, max_chars=max_chars)


    def list_step_outputs(self, item_id: str) -> Dict[str, Dict[str, Any]]:
        """
        Return a structured summary of all step outputs for an item.
        Includes raw, current, and edit snapshots for each step.
        """
        item_dir = self.base_dir / item_id
        artifact_dir = item_dir / "artifacts"
        if not artifact_dir.exists():
            return {}

        result = {}

        for path in artifact_dir.iterdir():
            if not path.is_file():
                continue

            name = path.name

            # Match patterns: step_raw.json, step_current.json, step_edit_X.json
            if name.endswith("_raw.json"):
                step = name[:-9]  # remove "_raw.json"
                result.setdefault(step, {"raw": None, "current": None, "edits": []})
                result[step]["raw"] = str(path)

            elif name.endswith("_current.json"):
                step = name[:-13]  # remove "_current.json"
                result.setdefault(step, {"raw": None, "current": None, "edits": []})
                result[step]["current"] = str(path)

            elif "_edit_" in name:
                step = name.split("_edit_")[0]
                result.setdefault(step, {"raw": None, "current": None, "edits": []})
                result[step]["edits"].append(str(path))

        return result

    def summarize_step_outputs(self, item_id: str) -> str:
        """
        Produce a clean, grouped, sorted summary of all step outputs
        for an item. Shows raw, current, and edit snapshots for each step.
        """
        outputs = self.list_step_outputs(item_id)
        if not outputs:
            return "No step outputs found."

        # Sort steps alphabetically for stable output
        steps = sorted(outputs.keys())

        lines = []
        for step in steps:
            info = outputs[step]

            lines.append(f"Step: {step}")
            lines.append(f"  raw:     {info['raw']}")
            lines.append(f"  current: {info['current']}")

            if info["edits"]:
                lines.append(f"  edits:")
                for e in sorted(info["edits"]):
                    lines.append(f"    - {e}")
            else:
                lines.append(f"  edits:   []")

            lines.append("")  # blank line between steps

        return "\n".join(lines)


    def load_raw_step_output(self, item_id: str, step_name: str):
        artifact_dir = self.base_dir / item_id / "artifacts"
        raw_path = artifact_dir / f"{step_name}_raw.json"
        if raw_path.exists():
            return self._read_json(raw_path)
        raise FileNotFoundError(f"No raw output for step '{step_name}'")

    def load_step_edit_history(self, item_id: str, step_name: str) -> List[Dict[str, Any]]:
        artifact_dir = self.base_dir / item_id / "artifacts"
        edits = []

        for path in artifact_dir.glob(f"{step_name}_edit_*.json"):
            edits.append(self._read_json(path))

        return edits


    def _write_json(self, path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _read_json(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _validate_step_output(self, step_spec: WorkflowStepSpec, output: WorkflowStepOutput):
        schema = step_spec.output_schema
        if schema is None:
            return output.artifact  # no validation needed

        try:
            validated = schema.model_validate(output.artifact)
            return validated.model_dump()
        except Exception as e:
            raise ValueError(
                f"Step '{step_spec.name}' produced invalid output: {e}"
            )


    def run_all_steps(self, item):
        """
        Run the workflow to completion by repeatedly calling run_next_step.
        """
        item_id = item.id

        while True:
            # Load current item state
            item = self.load_item(item_id)
            status = item.status

            # Determine if there is a next step
            next_substate = self.definition.get_default_next_substate(
                status.branch,
                status.substate
            )

            # No next substate → workflow is complete
            if next_substate is None:
                break

            # Run the next step
            self.run_next_step(item_id)





# Resolve forward references now that BaseWorkflowEngine is defined
from .models import WorkflowStepInput, WorkflowStepSpec, WorkflowStepOutput, WorkflowItem, StepOutputRecord

WorkflowStepInput.model_rebuild()
WorkflowStepSpec.model_rebuild()
WorkflowStepOutput.model_rebuild()
WorkflowItem.model_rebuild()
StepOutputRecord.model_rebuild()
