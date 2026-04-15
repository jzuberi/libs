from .loader import WorkflowLoader
from .models import WorkflowStepOutput

def call_workflow(workflow_name: str):
    def _run_nested_workflow(input):
        EngineClass = WorkflowLoader.load_engine(workflow_name)

        engine = EngineClass(
            base_dir=input.engine.base_dir,
            agent_llm=input.engine.agent_llm
        )

        first_step = engine.definition.workflow_paths["default"][0]

        item = engine.create_item(
            description=f"Nested workflow: {workflow_name}",
            type=workflow_name,
            initial_substate=first_step
        )

        engine.run_all_steps(item)

        # Convert StepOutputRecord → dict for JSON serialization
        artifact = {
            step_name: record.model_dump()
            for step_name, record in item.step_outputs.items()
        }

        return WorkflowStepOutput(
            artifact=artifact,
            next_substate=None,
            details={"nested_workflow": workflow_name},
            summary=f"Nested workflow '{workflow_name}' completed"
        )

    return _run_nested_workflow
