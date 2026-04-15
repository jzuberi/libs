import json
from pathlib import Path
from datetime import datetime


class EvalResultRecorder:
    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(exist_ok=True)
        self.results = []

    def record_case(self, name: str, status: str, score: float, reasons: list):
        self.results.append({
            "name": name,
            "status": status,
            "score": score,
            "reasons": reasons,
        })

    def save(self):
        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
        out_path = self.out_dir / f"eval_results_{timestamp}.json"

        summary = {
            "timestamp": timestamp,
            "summary": {
                "total": len(self.results),
                "passed": sum(1 for r in self.results if r["status"] == "PASS"),
                "failed": sum(1 for r in self.results if r["status"] == "FAIL"),
            },
            "cases": self.results,
        }

        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)

        return out_path
