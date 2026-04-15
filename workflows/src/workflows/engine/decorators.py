from functools import wraps
from .utils.tracing import workflow_step
from .models import WorkflowStepSpec
from abc import ABCMeta

from .workflow_definition import WorkflowDefinition
from .base_workflow_engine import BaseWorkflowEngine
from .loader import WorkflowLoader


def step(
    name: str,
    output_schema=None,
    human_name: str | None = None,
    description: str | None = None,
    consumes: list[str] | None = None,
    produces: list[str] | None = None,
    agent_hints: str | None = None,
):
    """
    Public ergonomic decorator for defining workflow steps.
    """
    def decorator(fn):
        wrapped = workflow_step(name)(fn)

        wrapped._step_spec = WorkflowStepSpec(
            name=name,
            fn=wrapped,
            output_schema=output_schema,
            human_name=human_name,
            description=description,
            consumes=consumes or [],
            produces=produces or [],
            agent_hints=agent_hints,
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

        # Collect steps
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

        # Build WorkflowDefinition
        definition = WorkflowDefinition(
            name=name,
            step_specs=step_specs,
            workflow_paths=cls.workflow_paths,
        )

        # Create dynamic engine subclass
        class GeneratedEngine(BaseWorkflowEngine):
            pass

        # Attach definition
        GeneratedEngine.definition = definition

        # Copy engine methods from workflow class
        if "summarize_item_structured" in cls.__dict__:
            GeneratedEngine.summarize_item_structured = cls.__dict__["summarize_item_structured"]

        if "_export_item_impl" in cls.__dict__:
            GeneratedEngine._export_item_impl = cls.__dict__["_export_item_impl"]

        # Inject constructor
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

        # Clear abstract methods
        GeneratedEngine.__abstractmethods__ = frozenset()

        # Register engine class
        WorkflowLoader.register(name, GeneratedEngine)

        cls.Engine = GeneratedEngine
        return cls

    return decorator
