from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from .models import WorkflowStepInput, WorkflowStepOutput


class StepContext:
    """
    High-level, engine-agnostic interface for step functions.
    Steps should never touch engine internals directly.
    """

    def __init__(self, step_input: WorkflowStepInput):
        self.input = step_input
        self.item = step_input.item
        self.engine = step_input.engine
        self.context = step_input.context or {}

        # Output fields
        self._artifact: Optional[Dict[str, Any]] = None
        self._details: Dict[str, Any] = {}
        self._summary: str = ""
        self._next_substate: Optional[str] = None
        self._requires_approval: bool = False

    # ---------------------------------------------------------
    # NEW: Typed Input Model (Pattern A)
    # ---------------------------------------------------------

    @property
    def input_model(self):
        """
        Return the parsed, validated input schema instance for this step,
        or None if the step has no input_schema or no context was provided.
        """
        return self.input.parsed_input

    # ---------------------------------------------------------
    # Step Output
    # ---------------------------------------------------------

    def set_output(self, artifact: Dict[str, Any]):
        self._artifact = artifact

    def set_details(self, details: Dict[str, Any]):
        self._details = details

    def set_summary(self, summary: str):
        self._summary = summary

    def set_next_substate(self, substate: str):
        self._next_substate = substate

    def require_approval(self):
        self._requires_approval = True

    # ---------------------------------------------------------
    # Previous Step Outputs
    # ---------------------------------------------------------

    def get_output(self, step_name: str) -> Dict[str, Any]:
        return self.engine.load_step_output(self.item.id, step_name)

    def get_typed_output(self, step_name: str):
        return self.engine.load_typed_step_output(self.item.id, step_name)

    def get_self_output(self):
        """
        Return this step's previously materialized *current* output
        (raw or edited). If the step has never run before, return None
        so the step's own logic can run normally.
        """
        step_name = self.input.step_name
        if not step_name:
            return None

        record = self.item.step_outputs.get(step_name)
        if not record:
            return None

        # Only return if a current exists
        if record.current is not None:
            return record.current

        # If no current exists, treat as "never run"
        return None

    # ---------------------------------------------------------
    # Metadata / Style
    # ---------------------------------------------------------

    def get_metadata(self, key: str, default=None):
        return self.item.metadata.get(key, default)

    def get_style(self, key: str, default=None):
        return self.item.style.get(key, default)

    # ---------------------------------------------------------
    # Assets
    # ---------------------------------------------------------

    def save_asset(self, filename: str, content: str) -> str:
        item_dir = self.engine.base_dir / self.item.id / "assets"
        item_dir.mkdir(exist_ok=True)
        path = item_dir / filename
        path.write_text(content)
        self.item.assets[filename] = str(path)
        return str(path)

    def load_asset(self, filename: str) -> str:
        path = Path(self.item.assets[filename])
        return path.read_text()

    # ---------------------------------------------------------
    # Finalize
    # ---------------------------------------------------------

    def finalize(self) -> WorkflowStepOutput:
        
        if self._artifact is None:
            raise ValueError("Step did not set an artifact via ctx.set_output().")

        return WorkflowStepOutput(
            artifact=self._artifact,
            details=self._details,
            summary=self._summary,
            next_substate=self._next_substate,
            requires_approval=self._requires_approval,
        )

    def stay(self) -> WorkflowStepOutput:
        """
        Do not advance the workflow. Stay in the current step.
        """
        self._next_substate = self.item.status.substate

        if self._artifact is None:
            raise ValueError("ctx.stay() requires ctx.set_output() first.")

        return WorkflowStepOutput(
            artifact=self._artifact,
            details=self._details,
            summary=self._summary,
            next_substate=self._next_substate,
            requires_approval=self._requires_approval,
        )


