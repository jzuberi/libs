# src/pensive/storage.py


import json
import os
from datetime import datetime


class SaveConfig:
    """Controls what parts of a turn get persisted."""

    def __init__(
        self,
        save_snapshot_metadata=True,
        save_strategy_outputs=True,
        save_criteria_results=True,
        save_review_rationale=True,
        save_final_decisions=True,
    ):
        self.save_snapshot_metadata = save_snapshot_metadata
        self.save_strategy_outputs = save_strategy_outputs
        self.save_criteria_results = save_criteria_results
        self.save_review_rationale = save_review_rationale
        self.save_final_decisions = save_final_decisions


class DecisionStore:
    def __init__(self, path="./decisions"):
        self.path = path
        os.makedirs(self.path, exist_ok=True)
        self.file = os.path.join(self.path, "decisions.json")

        # Initialize or upgrade file
        if not os.path.exists(self.file):
            with open(self.file, "w") as f:
                json.dump({"accepted": [], "rejected": [], "deferred": []}, f)
        else:
            # Upgrade existing file if missing keys
            with open(self.file, "r") as f:
                data = json.load(f)

            changed = False
            for key in ["accepted", "rejected", "deferred"]:
                if key not in data:
                    data[key] = []
                    changed = True

            if changed:
                with open(self.file, "w") as f:
                    json.dump(data, f, indent=2)



    def load(self):
        with open(self.file, "r") as f:
            return json.load(f)

    def save(self, decisions):
        with open(self.file, "w") as f:
            json.dump(decisions, f, indent=2)
            
    def add_raw(self, raw_decisions):
        """Append raw decision dicts to the correct buckets."""
        data = self.load()

        for d in raw_decisions:
            status = d["status"]
            if status not in data:
                raise ValueError(f"Unknown decision status: {status}")
            data[status].append(d)

        self.save(data)



class TurnStore:
    def __init__(self, path="./turns"):
        self.path = path
        os.makedirs(self.path, exist_ok=True)

    def save_turn(self, turn_data):
        turn_id = turn_data.get("turn_id") or datetime.now().isoformat().replace(":", "-")
        file = os.path.join(self.path, f"turn_{turn_id}.json")
        with open(file, "w") as f:
            json.dump(turn_data, f, indent=2)
