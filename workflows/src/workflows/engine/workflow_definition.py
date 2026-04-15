from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .models import WorkflowStepSpec, ValidatorFn


@dataclass
class WorkflowDefinition:
    # NEW: workflow name (required for loader + nested workflows)
    name: str = ""

    # branch -> ordered list of substates
    workflow_paths: Dict[str, List[str]] = field(default_factory=dict)

    # substate -> list of next substates (DAG)
    allowed_transitions: Dict[str, List[str]] = field(default_factory=dict)

    # substate -> step spec (function + context requirements)
    step_specs: Dict[str, WorkflowStepSpec] = field(default_factory=dict)

    # substate -> list of validators
    validators: Dict[str, List[ValidatorFn]] = field(default_factory=dict)

    # "function" or "workflow"
    type: str = "function"



    def get_default_next_substate(self, branch: str, current_substate: str) -> str | None:
        """Return the default next substate for a given branch and current substate."""
        path = self.workflow_paths.get(branch, [])
        if not path:
            return None
        try:
            idx = path.index(current_substate)
        except ValueError:
            return None
        if idx + 1 < len(path):
            return path[idx + 1]
        return None
