from abc import ABC, abstractmethod
from typing import Dict, Any

class DecisionPolicy(ABC):

    def decide(self, idea, criteria_results):
        """Optional: per‑idea decision."""
        raise NotImplementedError

    def decide_all(self, ideas, criteria_results):
        """
        Optional: per‑turn decision.
        Should return a list of:
        {
            "idea": idea,
            "status": "accepted" | "rejected" | "deferred",
            "rationale": str
        }
        """
        # Default fallback: call decide() on each idea
        return [
            {
                "idea": idea,
                "status": self.decide(idea, criteria_results[idea["key"]])["status"],
                "rationale": self.decide(idea, criteria_results[idea["key"]])["rationale"],
            }
            for idea in ideas
        ]
