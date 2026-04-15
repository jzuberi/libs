import json
from pathlib import Path
from datetime import datetime

class TraceCollector:
    def __init__(self, case_name: str, trace_dir: Path | None = None):
        self.case_name = case_name
        self.trace_dir = trace_dir  # <-- new
        self.data = {
            "case_name": case_name,
            "timestamp": datetime.utcnow().isoformat(),
            "prompt": None,
            "intent_llm": None,
            "intent_rule_based": None,
            "final_intent": None,
            "item_resolution": None,
            "engine_calls": [],
            "agent_response": None,
            "errors": []
        }

    def set_prompt(self, prompt):
        self.data["prompt"] = prompt

    def record_llm_intent(self, intent):
        self.data["intent_llm"] = intent

    def record_rule_intent(self, intent):
        self.data["intent_rule_based"] = intent

    def record_final_intent(self, intent):
        self.data["final_intent"] = intent

    def record_item_resolution(self, info):
        self.data["item_resolution"] = info

    def record_engine_call(self, name, args=None, result=None):
        self.data["engine_calls"].append({
            "call": name,
            "args": args,
            "result": result
        })

    def record_agent_response(self, response):
        self.data["agent_response"] = response

    def record_error(self, error):
        self.data["errors"].append(str(error))

    def save(self):
        # If caller passed a directory, use it
        if self.trace_dir is not None:
            traces_dir = Path(self.trace_dir)
        else:
            # fallback default (base evals)
            traces_dir = Path(__file__).parent / "eval_traces"

        traces_dir.mkdir(exist_ok=True, parents=True)

        out = traces_dir / f"{self.case_name.replace(' ', '_')}.json"
        with open(out, "w") as f:
            json.dump(self.data, f, indent=2)
