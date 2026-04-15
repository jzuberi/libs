from __future__ import annotations

from ....engine.decorators import step
from ....engine.step_context import StepContext
from ..schemas.transform_schema import TransformSchema


@step(
    "transform_data",
    output_schema=TransformSchema,
    human_name="Transform Data",
    description="Transforms the ingested numbers by multiplying them by 10.",
    consumes=["ingest_data"],
    produces=["transformed numbers"],
    agent_hints="This step depends on ingest_data. It produces a list of transformed numbers."
)
def transform(input):
    ctx = StepContext(input)

    ingest = ctx.get_typed_output("ingest_data")
    transformed = [n * 10 for n in ingest.numbers]

    ctx.set_output({"transformed": transformed})
    ctx.set_details({"count": len(transformed)})
    ctx.set_summary("Transformed numbers by multiplying by 10.")

    return ctx.finalize()
