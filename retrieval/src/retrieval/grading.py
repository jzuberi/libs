import json
import re
from .models import GradeResult

class BooleanGrader:
    def __init__(self, llm_call, debug=False):
        self.llm_call = llm_call
        self.debug = debug

    def grade(self, prompt: str) -> GradeResult:
        raw = self.llm_call(prompt)

        # ---------------------------------------------------------
        # 1. Try JSON parsing first
        # ---------------------------------------------------------
        try:
            data = json.loads(raw)

            # Case A: JSON object with "relevant"
            if isinstance(data, dict) and "relevant" in data:
                passed = bool(data["relevant"])
                return GradeResult(
                    passed=passed,
                    score=1.0 if passed else 0.0,
                    reason=data.get("reason"),
                    raw=data,   # already a dict
                )

            # Case B: JSON boolean (true/false)
            if isinstance(data, bool):
                return GradeResult(
                    passed=data,
                    score=1.0 if data else 0.0,
                    reason=None,
                    raw={"raw": data},   # wrap in dict
                )

        except Exception:
            pass  # fall through to raw handling

        # ---------------------------------------------------------
        # 2. Handle raw "true"/"false" strings
        # ---------------------------------------------------------
        if isinstance(raw, str):
            raw_lower = raw.strip().lower()
            if raw_lower == "true":
                return GradeResult(
                    passed=True,
                    score=1.0,
                    reason=None,
                    raw={"raw": raw},   # wrap in dict
                )
            if raw_lower == "false":
                return GradeResult(
                    passed=False,
                    score=0.0,
                    reason=None,
                    raw={"raw": raw},   # wrap in dict
                )

        # ---------------------------------------------------------
        # 3. Handle raw Python booleans
        # ---------------------------------------------------------
        if isinstance(raw, bool):
            return GradeResult(
                passed=raw,
                score=1.0 if raw else 0.0,
                reason=None,
                raw={"raw": raw},   # wrap in dict
            )

        # ---------------------------------------------------------
        # 4. Final fallback — cannot interpret
        # ---------------------------------------------------------
        return GradeResult(
            passed=False,
            raw={"error": "json_parse_failed", "raw": raw},
        )



class ScoreGrader:
    def __init__(self, llm_call, threshold=0.5):
        self.llm_call = llm_call
        self.threshold = threshold

    def grade(self, prompt: str) -> GradeResult:
        raw = self.llm_call(prompt)
        data = json.loads(raw)

        score = float(data.get("score", 0.0))
        passed = score >= self.threshold

        return GradeResult(
            passed=passed,
            score=score,
            reason=data.get("reason"),
            raw=data,
        )
