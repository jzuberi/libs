from __future__ import annotations

from ...agent.workflow_agent import WorkflowAgent


class SimpleProjectAgent(WorkflowAgent):
    """
    Agent that wraps an already-instantiated workflow engine.
    """

    def __init__(self, engine):
        # Just pass the engine to the parent class
        super().__init__(engine)
