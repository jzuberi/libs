from __future__ import annotations

import functools
import traceback
from typing import Any, Callable, Dict

from ..models import TraceLevel, WorkflowStepInput, WorkflowStepOutput


# -------------------------------------------------------------------------
# Workflow step tracing
# -------------------------------------------------------------------------

def workflow_step_internal(step_name: str, trace_level: TraceLevel = TraceLevel.INFO):
    """
    Decorator for workflow step functions.

    Automatically logs:
      - step_start
      - step_end
      - artifacts
      - details
      - exceptions
    """

    def decorator(fn: Callable[[WorkflowStepInput], WorkflowStepOutput]):
        @functools.wraps(fn)
        def wrapper(input: WorkflowStepInput) -> WorkflowStepOutput:
            engine = input.engine
            item = input.item
            status = item.status

            # STEP START
            engine._log_trace(
                trace_type="workflow",
                item_id=item.id,
                branch=status.branch,
                substate=status.substate,
                step_name=step_name,
                actor="engine",
                artifact=None,
                details={"event": "step_start"},
                trace_level=trace_level,
            )

            try:
                output = fn(input)

            except Exception as e:
                # ERROR TRACE
                engine._log_trace(
                    trace_type="workflow",
                    item_id=item.id,
                    branch=status.branch,
                    substate=status.substate,
                    step_name=step_name,
                    actor="engine",
                    artifact=None,
                    details={
                        "event": "exception",
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    },
                    trace_level=TraceLevel.ERROR,
                )
                raise

            # STEP END
            engine._log_trace(
                trace_type="workflow",
                item_id=item.id,
                branch=item.status.branch,
                substate=item.status.substate,
                step_name=step_name,
                actor="engine",
                artifact=output.artifact,
                details={"event": "step_end", **(output.details or {})},
                trace_level=trace_level,
            )

            return output

        return wrapper

    return decorator


# -------------------------------------------------------------------------
# Agent method tracing
# -------------------------------------------------------------------------

def agent_trace(method_name: str, trace_level: TraceLevel = TraceLevel.INFO):
    """
    Decorator for WorkflowAgent methods.

    Logs:
      - agent_start
      - agent_end
      - arguments
      - results (summarized)
      - exceptions
    """

    def decorator(fn: Callable):
        @functools.wraps(fn)
        def wrapper(self, *args, **kwargs):
            engine = self.engine

            # START
            engine._log_trace(
                trace_type="agent",
                item_id=getattr(self.session, "last_item_id", None),
                branch=None,
                substate=None,
                step_name=method_name,
                actor="agent",
                artifact=None,
                details={"event": "agent_start", "args": str(args), "kwargs": str(kwargs)},
                trace_level=trace_level,
            )

            try:
                result = fn(self, *args, **kwargs)

            except Exception as e:
                engine._log_trace(
                    trace_type="agent",
                    item_id=getattr(self.session, "last_item_id", None),
                    branch=None,
                    substate=None,
                    step_name=method_name,
                    actor="agent",
                    artifact=None,
                    details={
                        "event": "exception",
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    },
                    trace_level=TraceLevel.ERROR,
                )
                raise

            # END
            engine._log_trace(
                trace_type="agent",
                item_id=getattr(self.session, "last_item_id", None),
                branch=None,
                substate=None,
                step_name=method_name,
                actor="agent",
                artifact=None,
                details={"event": "agent_end", "result": str(result)[:500]},
                trace_level=trace_level,
            )

            return result

        return wrapper

    return decorator
