import dataclasses

class BaseScorer:
    def __init__(self, agent):
        self.agent = agent

    def score_case(self, case: dict):
        prompt = case["prompt"]
        expected = case.get("expected", {})

        response = self.agent.handle_message(prompt)
        intent = getattr(self.agent.session, "last_intent", None)
        item_id = getattr(self.agent.session, "last_item_id", None)

        score = 1.0
        reasons = []

        # intent check
        if "intent" in expected and intent != expected["intent"]:
            score -= 0.4
            reasons.append(f"intent mismatch: {intent} != {expected['intent']}")

        # must_include
        for text in expected.get("must_include", []):
            if text.lower() not in response.lower():
                score -= 0.2
                reasons.append(f"missing text: {text}")

        # must_not_include
        for text in expected.get("must_not_include", []):
            if text.lower() in response.lower():
                score -= 0.2
                reasons.append(f"forbidden text present: {text}")

        return max(score, 0.0), reasons
