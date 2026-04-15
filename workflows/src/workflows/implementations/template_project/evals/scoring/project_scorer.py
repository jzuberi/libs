# projects/simple_project/evals/scoring/project_scorer.py
from workflows.agent.evals.base.scoring.base_scorer import BaseScorer

class SimpleProjectScorer(BaseScorer):
    def score_case(self, case: dict):
        score, reasons = super().score_case(case)
        expected = case.get("expected", {})

        # example: project-specific check
        if expected.get("requires_item_resolution"):
            if self.agent.session.last_item_id is None:
                score -= 0.3
                reasons.append("item_id was not resolved")

        return max(score, 0.0), reasons
