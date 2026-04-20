"""
Public API for the workflows package.

This module exposes the stable, user-facing surface for:
- defining workflows and steps
- running workflows
- building agents
- interacting with workflow engines

Internal engine modules remain private.
"""

# ---------------------------------------------------------
# Decorators (workflow authoring)
# ---------------------------------------------------------
from .engine.decorators import workflow, step, workflow_step

# ---------------------------------------------------------
# Step execution helpers
# ---------------------------------------------------------
from .engine.step_context import StepContext
from .engine.models import WorkflowStepOutput, WorkflowItem


# ---------------------------------------------------------
# Engine loading + workflow invocation
# ---------------------------------------------------------
from .engine.loader import WorkflowLoader

def load_engine(name: str):
    """Load a workflow engine by name."""
    return WorkflowLoader.load_engine(name)

from .engine.call_workflow import call_workflow

# Optional convenience wrapper
try:
    from .api import run_workflow
except ImportError:
    # run_workflow is optional; ignore if not present
    pass

# ---------------------------------------------------------
# Agent
# ---------------------------------------------------------
from .agent.workflow_agent import WorkflowAgent

# ---------------------------------------------------------
# Public API surface
# ---------------------------------------------------------
__all__ = [
    # Decorators
    "workflow",
    "step",
    "workflow_step",

    # Step helpers
    "StepContext",
    "WorkflowStepOutput",
    "WorkflowItem",

    # Engine access
    "load_engine",
    "call_workflow",

    # Agent
    "WorkflowAgent",
]

# Optional exports if run_workflow exists
if "run_workflow" in globals():
    __all__.append("run_workflow")
