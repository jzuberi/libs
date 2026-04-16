# workflows/engine/loader.py

workflow_registry = {}

class WorkflowLoader:
    @staticmethod
    def register(name: str, engine_class):
        """
        Register a workflow engine class directly.
        """
        workflow_registry[name] = engine_class

    @staticmethod
    def load_engine(name: str):
        """
        Load a workflow engine class from the registry.
        """
        if name not in workflow_registry:
            print(f"Workflow '{name}' is not registered.")
            raise ValueError(f"Workflow '{name}' is not registered.")
        return workflow_registry[name]
