from workflows.agent.evals.base.runner.eval_result_recorder import EvalResultRecorder
from workflows.engine.base_workflow_engine import BaseWorkflowEngine
from workflows.engine.workflow_definition import WorkflowDefinition, WorkflowStepSpec
from pydantic import BaseModel

from workflows.agent.workflow_agent import WorkflowAgent
from workflows.agent.session import SessionState
from workflows.agent.evals.base.scoring.base_scorer import BaseScorer

from workflows.agent.evals.base.runner.trace import TraceCollector


import sys
sys.path.append('/Users/pense/projects/pensiveturtles_data/pfuncs/agentic')
import agentic_funcs as a_f

gAnswerer_qwen = a_f.get_llm(
    local_llm='qwen3.5-9b-claude-4.6-highiq-instruct-heretic-uncensored',
    local_timeout=250,
    retry=False,
)

from datetime import datetime
import json
from pathlib import Path

results = []


# --- Synthetic schemas ---
class IngestSchema(BaseModel):
    value: int


class TransformSchema(BaseModel):
    value: int


class ExportSchema(BaseModel):
    value: int


# --- Synthetic step functions ---
def ingest_fn(ctx):
    return {"value": 1}


def transform_fn(ctx):
    prev = ctx.get("ingest_data", {}).get("value", 1)
    return {"value": prev * 2}


def export_fn(ctx):
    prev = ctx.get("transform_data", {}).get("value", 2)
    return {"value": prev + 10}


# --- Synthetic engine for base evals ---
class SyntheticEvalEngine(BaseWorkflowEngine):
    def __init__(self, base_dir='.'):
        definition = WorkflowDefinition(
            workflow_paths={"default": ["ingest_data", "transform_data", "export_data"]},
            step_specs={
                "ingest_data": WorkflowStepSpec(
                    name="ingest_data",
                    human_name="Ingest",
                    fn=ingest_fn,
                    output_schema=IngestSchema,
                    description="Synthetic ingest step for base evals.",
                ),
                "transform_data": WorkflowStepSpec(
                    name="transform_data",
                    human_name="Transform",
                    fn=transform_fn,
                    output_schema=TransformSchema,
                    description="Synthetic transform step for base evals.",
                    consumes=["ingest_data"],
                ),
                "export_data": WorkflowStepSpec(
                    name="export_data",
                    human_name="Export",
                    fn=export_fn,
                    output_schema=ExportSchema,
                    description="Synthetic export step for base evals.",
                    consumes=["transform_data"],
                ),
            },
            validators={},
        )

        super().__init__(definition=definition, base_dir=base_dir, agent_llm=gAnswerer_qwen)

    # --- required abstract methods ---
    def _export_item_impl(self, item_id: str):
        # Minimal export behavior for evals
        return {"status": "exported", "item_id": item_id}

    def summarize_item_structured(self, item_id: str, small: bool = False):
        # Minimal structured summary for evals
        return {
            "item_id": item_id,
            "status": "ok",
            "steps": ["ingest_data", "transform_data", "export_data"],
        }

    def get_item(self, item_id: str):

        if hasattr(self, "trace") and self.trace:
            self.trace.record_engine_call("run_next_step", args={"item_id": item_id})

        """Agent-facing convenience wrapper used by StepContext commands."""
        return self.load_item(item_id)



def load_cases():
    from pathlib import Path
    import yaml

    cases_dir = Path(__file__).parent.parent / "cases"
    all_cases = []
    for path in cases_dir.glob("*.yaml"):
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        if data is None:
            continue  # skip empty files

        if isinstance(data, list):
            all_cases.extend(data)
        elif isinstance(data, dict):
            all_cases.append(data)
        else:
            raise ValueError(f"Unexpected YAML structure in {path}: {type(data)}")

    return all_cases


def run():

    # --- build base test agent ---
    engine = SyntheticEvalEngine(base_dir='.')
    agent = WorkflowAgent(engine=engine)
    scorer = BaseScorer(agent)
    cases = load_cases()

    # --- shared summary recorder ---
    out_dir = agent.engine.base_dir / "eval_results"
    recorder = EvalResultRecorder(out_dir)

    for case in cases:
        # --- trace setup ---
        trace = TraceCollector(case["name"], trace_dir=agent.engine.base_dir / 'eval_traces')
        agent.engine.attach_trace(trace)
        trace.set_prompt(case["prompt"])

        try:
            # --- run agent with tracing ---
            response = agent.handle_message(case["prompt"], trace=trace)
            trace.record_agent_response(response)

            # scorer does NOT take response
            score, reasons = scorer.score_case(case)

        except Exception as e:
            trace.record_error(e)
            score, reasons = 0.0, [str(e)]

        # --- save trace file ---
        trace.save()

        # --- print result ---
        status = "PASS" if score >= 0.99 else "FAIL"
        print(f"{status}  {case['name']}  (score={score:.2f})")
        for r in reasons:
            print(f"  - {r}")

        # --- record succinct result ---
        recorder.record_case(case["name"], status, score, reasons)

    # --- save summary file ---
    out_path = recorder.save()
    print(f"\nSaved eval summary → {out_path}")





if __name__ == "__main__":
    run()
