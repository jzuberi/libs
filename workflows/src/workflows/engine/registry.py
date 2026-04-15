workflow_registry = {}

def register_workflow(name: str, engine_path: str):
    """
    Register a workflow by name with its engine import path.
    engine_path example: "my_project.workflows.contract.engine.ContractEngine"
    """
    workflow_registry[name] = engine_path
