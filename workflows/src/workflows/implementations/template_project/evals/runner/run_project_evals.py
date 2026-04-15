# projects/simple_project/evals/runner/run_project_evals.py

from pathlib import Path
import yaml
from workflows.agent.workflow_agent import WorkflowAgent
from workflows.agent.evals.base.scoring.base_scorer import BaseScorer
from workflows.agent.evals.base.runner.trace import TraceCollector
from workflows.implementations.template_project.simple_project_engine import SimpleProjectEngine
from workflows.agent.evals.base.runner.eval_result_recorder import EvalResultRecorder


import sys
sys.path.append('/Users/pense/projects/pensiveturtles_data/pfuncs/agentic')
import agentic_funcs as a_f



def build_project_agent(base_dir):
    engine = SimpleProjectEngine(base_dir)
    return WorkflowAgent(engine)


def load_cases():
    cases_dir = Path(__file__).parent.parent / "cases"
    all_cases = []
    for path in cases_dir.glob("*.yaml"):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
            if data:
                all_cases.extend(data)
    return all_cases



def run():
    # --- build agent + engine ---
    gAnswerer_qwen = a_f.get_llm(
        local_llm='qwen3.5-9b-claude-4.6-highiq-instruct-heretic-uncensored',
        local_timeout=250,
        retry=False,
    )

    base_dir = '.'
    engine = SimpleProjectEngine(base_dir, agent_llm=gAnswerer_qwen)
    agent = WorkflowAgent(engine)

    scorer = BaseScorer(agent)
    cases = load_cases()

    # --- shared summary recorder ---
    out_dir = agent.engine.base_dir / "eval_results"
    recorder = EvalResultRecorder(out_dir)

    for case in cases:
        trace = TraceCollector(case["name"], trace_dir=engine.base_dir)
        agent.engine.attach_trace(trace)
        trace.set_prompt(case["prompt"])

        try:
            response = agent.handle_message(case["prompt"], trace=trace)
            trace.record_agent_response(response)
            score, reasons = scorer.score_case(case)
        except Exception as e:
            trace.record_error(e)
            score, reasons = 0.0, [str(e)]

        trace.save()

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
