from __future__ import annotations

from workflows.engine.decorators import workflow, step
from .steps.call_simple import call_simple_step


@step("call_simple")
def call_simple(input):
    # call_workflow already returns a step function
    return call_simple_step(input)


@workflow("caller_project", steps=[call_simple])
class CallerProject:
    workflow_paths = {
        "default": ["call_simple"]
    }

    def summarize_item_structured(self, item, small: bool = False):
        return {
            "id": item.id,
            "type": item.type,
            "current_substate": item.status.substate,
            "outputs": item.step_outputs,
        }

    def _export_item_impl(self, item):
        return {"status": "export not implemented for caller workflow"}
