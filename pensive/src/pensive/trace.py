import json
import os


class TraceHelper:
    """Utility for navigating decisions and turns."""

    def __init__(self, instance_path):
        self.instance_path = instance_path
        self.decisions_file = os.path.join(instance_path, "decisions", "decisions.json")
        self.turns_path = os.path.join(instance_path, "turns")

    # -----------------------------
    # Load all decisions
    # -----------------------------
    def load_decisions(self):
        with open(self.decisions_file, "r") as f:
            return json.load(f)

    # -----------------------------
    # Load a specific turn by ID
    # -----------------------------
    def load_turn(self, turn_id):
        filename = f"turn_{turn_id}.json"
        path = os.path.join(self.turns_path, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No turn file found for turn_id={turn_id}")
        with open(path, "r") as f:
            return json.load(f)

    # -----------------------------
    # Find a decision by decision_id
    # -----------------------------
    def find_decision(self, decision_id):
        decisions = self.load_decisions()

        for status in ["accepted", "rejected", "deferred"]:
            for d in decisions.get(status, []):
                if d["decision_id"] == decision_id:
                    return d

        raise KeyError(f"Decision {decision_id} not found")

    # -----------------------------
    # Given a decision_id, load the turn that produced it
    # -----------------------------
    def trace_decision_to_turn(self, decision_id):
        decision = self.find_decision(decision_id)
        turn_id = decision["turn_id"]
        turn = self.load_turn(turn_id)
        return {
            "decision": decision,
            "turn": turn,
        }

    # -----------------------------
    # List all turns
    # -----------------------------
    def list_turns(self):
        turns = []

        for filename in os.listdir(self.turns_path):
            if not filename.startswith("turn_") or not filename.endswith(".json"):
                continue

            turn_id = filename.replace("turn_", "").replace(".json", "")
            path = os.path.join(self.turns_path, filename)

            # Load timestamp from inside the file
            with open(path, "r") as f:
                data = json.load(f)

            timestamp = data.get("timestamp", "")

            turns.append((timestamp, turn_id))

        # Sort by timestamp ascending
        turns.sort(key=lambda x: x[0])

        # Return only turn_ids
        return [tid for (_, tid) in turns]
