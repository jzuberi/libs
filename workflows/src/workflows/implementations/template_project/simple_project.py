from ...engine.decorators import workflow
from .steps.ingest import ingest
from .steps.transform import transform
from .steps.export import export

class SimpleProject:
    
    workflow_paths = {
        "default": ["ingest_data", "transform_data", "export_data"]
    }

    def summarize_item_structured(self, item, small: bool = False) -> str:
        return (
            f"Item {item.label}\n"
            f"Type: {item.type}\n"
            f"State: {item.status.branch}/{item.status.substate}\n"
            f"Approved: {item.status.approved}\n"
        )

    def _export_item_impl(self, item):
        export_dir = self.base_dir / item.id / "export"
        export_dir.mkdir(exist_ok=True)

        final_output = self.load_step_output(item.id, "export_data")
        out_path = export_dir / "export.json"
        self._write_json(out_path, final_output)

        return {"export_path": str(out_path)}

# ⭐ Apply decorator AFTER class creation
SimpleProject = workflow("simple_project", steps=[ingest, transform, export])(SimpleProject)
