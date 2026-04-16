from .loader import WorkflowLoader
from .models import WorkflowStepOutput

def call_workflow(workflow_name: str):
    def _run_nested_workflow(step_input):
        print(f"\n>>> ENTERED nested workflow: {workflow_name}")
        parent_engine = step_input.engine

        # Load child engine
        EngineClass = WorkflowLoader.load_engine(workflow_name)
        engine = EngineClass(
            base_dir=parent_engine.base_dir,
            agent_llm=parent_engine.agent_llm,
        )

        # Share item registry
        engine._items = parent_engine._items

        # Create child item
        first_step = engine.definition.workflow_paths["default"][0]
        parent_id = step_input.item.id

        item = engine.create_item(
            description=f"Nested workflow: {workflow_name}",
            type=workflow_name,
            initial_substate=first_step,
            parent_id=parent_id,
        )

        print(f"[CALL_WORKFLOW] child item id = {item.id}")

        # Run child workflow
        result = engine.run_until_blocked(item.id)

        # Load child step outputs
        child_item = engine.load_item(item.id)

        print(">>> CALL_WORKFLOW: creating child id =", item.id)

        step_outputs = {
            step_name: record.model_dump()
            for step_name, record in child_item.step_outputs.items()
        }

        # Build a consistent artifact
        artifact = {
            "item_id": item.id,       # <-- ALWAYS PRESENT
            "steps": step_outputs,    # <-- child step outputs
            "result": result,         # <-- engine result dict
        }

        # Build consistent details
        details = {
            "nested_workflow": workflow_name,
            "blocked": result["blocked"],
            "reason": result["reason"],
            "substate": result["substate"],
        }

        # Return consistent WorkflowStepOutput
        return WorkflowStepOutput(
            artifact=artifact,
            next_substate=None,
            details=details,          # <-- REQUIRED BY MODEL
            summary=f"Nested workflow '{workflow_name}' executed",
            approved=not result["blocked"],
        )

    return _run_nested_workflow
