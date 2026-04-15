"""
Public API for the Workflows framework.
"""

# Decorators
from .engine.decorators import workflow, step

# Engine loader
from .engine.loader import WorkflowLoader

def load_engine(name: str):
    return WorkflowLoader.load_engine(name)

# Nested workflow call primitive
from .engine.call_workflow import call_workflow

# Version (optional)
__all__ = [
    "workflow",
    "step",
    "load_engine",
    "call_workflow",
]
