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
from .engine.decorators import workflow, step, workflow_step, custom_data_edit_step

# ---------------------------------------------------------
# Step execution helpers
# ---------------------------------------------------------
from .engine.step_context import StepContext
from .engine.models import WorkflowStepOutput, WorkflowItem, HandlerMessage


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
from .agent.contract.loader import load_agent_contract
from .agent.context.decorators import updates_context

from .agent.utils.agentic_edit import (
    LocalFieldOntology, 
    LocalOntology, 
    EditResult, 
    resolve_local_edit, 
    ontology_from_model, 
    build_edits_from_edit_result, 
    register_validation_handler,
    )

from .agent.utils.handler_factory import (
    EditConfig, 
    ChoiceConfig,
    make_edit_handlers, 
    make_choice_handlers
    )

from .bridge.actor_record import (
    ActorRecord,
    )

from .bridge.background_agent.base import (
    BaseBackgroundAgent,
    )

# ---------------------------------------------------------
# Public API surface
# ---------------------------------------------------------
__all__ = [
    # Decorators
    "workflow",
    "step",
    "workflow_step",
    "custom_data_edit_step",

    # Step helpers
    "StepContext",
    "WorkflowStepOutput",
    "WorkflowItem",

    # Engine access
    "load_engine",
    "call_workflow",

    # Agent
    "WorkflowAgent",
    "HandlerMessage",
    "load_agent_contract",
    "updates_context",
    "LocalFieldOntology",
    "LocalOntology",
    "EditResult",
    "resolve_local_edit",
    "ontology_from_model",
    "build_edits_from_edit_result",
    "register_validation_handler",
    "EditConfig", 
    "ChoiceConfig",
    "make_edit_handlers", 
    "make_choice_handlers",

    # Bridge
    "ActorRecord",
    "BaseBackgroundAgent",
]

# Optional exports if run_workflow exists
if "run_workflow" in globals():
    __all__.append("run_workflow")
