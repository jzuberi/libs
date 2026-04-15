from __future__ import annotations

from ....engine.decorators import step
from ....engine.step_context import StepContext
from ..schemas.export_schema import ExportSchema


@step(
    "export_data",
    output_schema=ExportSchema,
    human_name="Export Data",
    description="Packages the transformed numbers into a final exportable format.",
    consumes=["transform_data"],
    produces=["exported artifact"],
    agent_hints="This is the final step. It exports the transformed numbers."
)
def export(input):
    ctx = StepContext(input)

    transform = ctx.get_typed_output("transform_data")

    ctx.set_output({"exported": {"transformed": transform.transformed}})
    ctx.set_summary("Exported transformed numbers.")

    return ctx.finalize()
