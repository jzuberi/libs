import json
import os
from typing import Dict, Any

class TurnDiff:
    """Compute differences between two turn files."""

    def __init__(self, instance_path):
        self.turns_path = os.path.join(instance_path, "turns")

    # -----------------------------
    # Load a turn by ID
    # -----------------------------
    def load_turn(self, turn_id: str) -> Dict[str, Any]:
        filename = f"turn_{turn_id}.json"
        path = os.path.join(self.turns_path, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No turn file found for turn_id={turn_id}")
        with open(path, "r") as f:
            return json.load(f)

    # -----------------------------
    # Compute diff between two turns
    # -----------------------------
    def diff(self, turn_a_id: str, turn_b_id: str) -> Dict[str, Any]:
        a = self.load_turn(turn_a_id)
        b = self.load_turn(turn_b_id)

        # ------------------------------------------------------------
        # 1. IDEA DIFFS (based on idea["key"])
        # ------------------------------------------------------------
        def extract_decision_ideas(turn):
            ideas = []
            for bucket in ("accepted", "rejected", "deferred"):
                for d in turn["decisions"][bucket]:
                    ideas.append(d["idea"])
            return ideas

        ideas_a = extract_decision_ideas(a)
        ideas_b = extract_decision_ideas(b)

        idea_map_a = {i["key"]: i for i in ideas_a}
        idea_map_b = {i["key"]: i for i in ideas_b}

        keys_a = set(idea_map_a.keys())
        keys_b = set(idea_map_b.keys())

        ideas_added = [idea_map_b[k] for k in (keys_b - keys_a)]
        ideas_removed = [idea_map_a[k] for k in (keys_a - keys_b)]

        # ------------------------------------------------------------
        # 2. STRATEGY OUTPUT DIFFS
        # ------------------------------------------------------------
        strategies_added = {
            k: v for k, v in b["strategy_outputs"].items()
            if k not in a["strategy_outputs"]
        }
        strategies_removed = {
            k: v for k, v in a["strategy_outputs"].items()
            if k not in b["strategy_outputs"]
        }

        # ------------------------------------------------------------
        # 3. CRITERIA RESULT DIFFS
        # ------------------------------------------------------------
        criteria_changes = {}
        all_keys = set(a["criteria_results"].keys()).union(b["criteria_results"].keys())

        for key in all_keys:
            a_res = a["criteria_results"].get(key, {})
            b_res = b["criteria_results"].get(key, {})
            if a_res != b_res:
                criteria_changes[key] = {"before": a_res, "after": b_res}

        # ------------------------------------------------------------
        # 4. DECISION DIFFS (bucketed)
        # ------------------------------------------------------------
        def bucket_diff(bucket):
            return {
                "added": [
                    d for d in b["decisions"][bucket]
                    if d not in a["decisions"][bucket]
                ],
                "removed": [
                    d for d in a["decisions"][bucket]
                    if d not in b["decisions"][bucket]
                ],
            }

        decisions_added = {
            "accepted": bucket_diff("accepted")["added"],
            "rejected": bucket_diff("rejected")["added"],
            "deferred": bucket_diff("deferred")["added"],
        }

        decisions_removed = {
            "accepted": bucket_diff("accepted")["removed"],
            "rejected": bucket_diff("rejected")["removed"],
            "deferred": bucket_diff("deferred")["removed"],
        }

        # ------------------------------------------------------------
        # 5. Return structured diff
        # ------------------------------------------------------------
        return {
            "turn_a": turn_a_id,
            "turn_b": turn_b_id,
            "ideas_added": ideas_added,
            "ideas_removed": ideas_removed,
            "strategies_added": strategies_added,
            "strategies_removed": strategies_removed,
            "criteria_changes": criteria_changes,
            "decisions_added": decisions_added,
            "decisions_removed": decisions_removed,
        }
