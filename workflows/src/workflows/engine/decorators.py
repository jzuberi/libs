from functools import wraps
from .utils.tracing import workflow_step_internal
from .models import WorkflowStepSpec
from abc import ABCMeta

from .workflow_definition import WorkflowDefinition
from .base_workflow_engine import BaseWorkflowEngine
from .loader import WorkflowLoader
from .step_context import StepContext


class ChildWorkflowProxy:
    def __init__(self, engine, parent_item, step_spec):
        self.engine = engine
        self.parent_item = parent_item
        self.step_spec = step_spec

    def run(self, **kwargs):
        return self.engine._run_child_workflow_step(
            parent_item=self.parent_item,
            step_spec=self.step_spec,
            initial_input=kwargs,
        )


def step(
    name: str,
    input_schema=None,
    output_schema=None,
    human_name: str | None = None,
    description: str | None = None,
    consumes: list[str] | None = None,
    produces: list[str] | None = None,
    agent_hints: str | None = None,
    mode: str = "function",   # ⭐ NEW
):
    """
    Public ergonomic decorator for defining workflow steps.
    """
    def decorator(fn):

        # ⭐ NEW: review steps use a no-op function to satisfy validation
        if mode == "review":
            def noop(*args, **kwargs):
                # Engine will skip calling this because kind="review"
                return {}
            wrapped = noop
        else:
            wrapped = workflow_step_internal(name)(fn)

        kind = "review" if mode == "review" else "function"

        step_spec = WorkflowStepSpec(
            name=name,
            fn=wrapped,                     # ⭐ always callable now
            output_schema=output_schema,
            human_name=human_name,
            description=description,
            consumes=consumes or [],
            produces=produces or [],
            agent_hints=agent_hints,
            child_workflow_name=None,
            kind=kind,
            input_schema=input_schema,
        )

        # ⭐ Attach spec to the wrapped function
        wrapped._step_spec = step_spec
        return wrapped

    return decorator



def workflow_step(
    child_workflow_name: str,
    name: str | None = None,
    output_schema=None,
    human_name: str | None = None,
    description: str | None = None,
    consumes: list[str] | None = None,
    produces: list[str] | None = None,
    agent_hints: str | None = None,
    input_schema=None,
):
    """
    Public decorator for defining a step that runs a child workflow.
    """

    def decorator(fn):
        step_name = name or fn.__name__

        @wraps(fn)
        def wrapped(input):
            ctx = StepContext(input)

            child = ChildWorkflowProxy(
                engine=input.engine,
                parent_item=input.item,
                step_spec=wrapped._step_spec,
            )

            result = fn(ctx, child)
            return result if result is not None else {}

        wrapped._step_spec = WorkflowStepSpec(
            name=step_name,
            fn=wrapped,
            output_schema=output_schema,
            human_name=human_name,
            description=description,
            consumes=consumes or [],
            produces=produces or [],
            agent_hints=agent_hints,
            child_workflow_name=child_workflow_name,
            kind="workflow",
            input_schema=input_schema,
        )

        return wrapped

    return decorator



def workflow(
    name: str,
    steps: list = None,
    approval_requirements: dict | None = None,
    label_fn=None,
):
    def decorator(cls):
        step_specs = {}

        if steps:
            funcs = steps
        else:
            funcs = [
                getattr(cls, attr)
                for attr in dir(cls)
                if hasattr(getattr(cls, attr), "_step_spec")
            ]

        for fn in funcs:
            spec = fn._step_spec
            step_specs[spec.name] = spec

        definition = WorkflowDefinition(
            name=name,
            step_specs=step_specs,
            workflow_paths=cls.workflow_paths,
        )

        class GeneratedEngine(BaseWorkflowEngine):
            pass

        GeneratedEngine.definition = definition

        if "summarize_item_structured" in cls.__dict__:
            GeneratedEngine.summarize_item_structured = cls.__dict__["summarize_item_structured"]

        if "_export_item_impl" in cls.__dict__:
            GeneratedEngine._export_item_impl = cls.__dict__["_export_item_impl"]

        def __init__(self, base_dir, agent_llm=None):
            BaseWorkflowEngine.__init__(
                self,
                definition=definition,
                base_dir=base_dir,
                approval_requirements=approval_requirements,
                agent_llm=agent_llm,
                label_fn=label_fn,
            )

        GeneratedEngine.__init__ = __init__
        GeneratedEngine.__abstractmethods__ = frozenset()

        WorkflowLoader.register(name, GeneratedEngine)

        cls.Engine = GeneratedEngine
        return cls

    return decorator
