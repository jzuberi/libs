from __future__ import annotations

from ....engine.decorators import step
from ....engine.step_context import StepContext
from ..schemas.ingest_schema import IngestSchema


@step(
    "ingest_data",
    output_schema=IngestSchema,
    human_name="Ingest Data",
    description="Loads the initial list of numbers into the workflow.",
    consumes=[],
    produces=["numbers"],
    agent_hints="This is the first step. It gathers raw numbers for later processing."
)
def ingest(input):
    ctx = StepContext(input)

    numbers = [1, 2, 3, 4]

    ctx.set_output({"numbers": numbers})
    ctx.set_details({"count": len(numbers)})
    ctx.set_summary("Ingested raw numbers.")

    return ctx.finalize()
