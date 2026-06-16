# libs/retrieval/src/retrieval/intent_builder.py

from .models import RetrievalIntent, Indicator, IntentType
from .negation import TopicNegationDetector, NegationStripper
from .time_postprocessing import apply_time_postprocessing

import json


def safe_get(d, key, default=None):
    """Defensive dictionary access."""
    try:
        return d.get(key, default)
    except Exception:
        return default


def safe_int(x):
    """Convert to int if possible, else None."""
    try:
        return int(x)
    except Exception:
        return None


def safe_str(x):
    """Convert to str if possible, else None."""
    try:
        return str(x)
    except Exception:
        return None


def apply_negation(indicators, neg_result):
    negated_values = {t.value.lower() for t in neg_result.negated_topics}

    # 1. Flip existing indicators to negated if needed
    for ind in indicators:
        if ind.name == "is_topic" and ind.value.lower() in negated_values:
            ind.negated = True

    # 2. Add missing negated indicators
    existing_values = {ind.value.lower() for ind in indicators if ind.name == "is_topic"}

    for value in negated_values:
        if value not in existing_values:
            indicators.append(
                Indicator(name="is_topic", value=value, negated=True)
            )

    return indicators


class RetrievalIntentBuilder:
    def __init__(self, llm_call):
        self.llm_call = llm_call

    # ---------------------------------------------------------
    # NEW: Separate, defensive time-intent extraction
    # ---------------------------------------------------------
    def extract_time_intent(self, query: str) -> dict:
        time_prompt = f"""
You are a time-intent extraction system.

Extract ONLY time-related intent from the user's query.

Return valid JSON with EXACTLY this structure:

{{
  "recency_days": number or null,
  "time_bucket": string or null
}}

TIME INTENT RULES:
- If the user expresses recency intent ("recent", "latest", "new",
  "past week", "last 30 days", "in the last month"), extract recency_days.
- Represent recency as an integer number of days.
- If recency is implied but no number is given, default recency_days to 30.
- If the user refers to a calendar period ("this week", "last month",
  "this year", "last quarter"), extract a time_bucket string.
- Valid time_bucket values:
  "this_week", "last_week", "this_month", "last_month",
  "this_year", "last_year", "this_quarter", "last_quarter".
- If no time intent is present, set both fields to null.

User query:
{query}
"""

        raw = self.llm_call(time_prompt)

        # -------------------------------
        # Defensive JSON parsing
        # -------------------------------
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("Time intent must be a JSON object")
        except Exception:
            return {"recency_days": None, "time_bucket": None}

        # -------------------------------
        # Defensive extraction
        # -------------------------------
        recency = safe_get(data, "recency_days", None)
        bucket = safe_get(data, "time_bucket", None)

        # Validate recency_days
        recency = safe_int(recency)

        # Validate time_bucket
        if bucket is not None:
            bucket = safe_str(bucket)
            if bucket not in {
                "this_week", "last_week", "this_month", "last_month",
                "this_year", "last_year", "this_quarter", "last_quarter"
            }:
                bucket = None

        return {
            "recency_days": recency,
            "time_bucket": bucket,
        }

    # ---------------------------------------------------------
    # ORIGINAL INTENT BUILDER (semantic + indicators)
    # ---------------------------------------------------------
    def build(self, query: str) -> RetrievalIntent:

        # -------------------------------
        # 1. Extract semantic intent
        # -------------------------------
        semantic_prompt = f"""
You are an intent-analysis system that converts a raw user query into a structured
retrieval request.

You MUST return valid JSON with EXACTLY this structure:

{{
  "type": "topic_filter" or "support_statement",
  "guidance": "expanded natural language description of the user's query",
  "grading": true,
  "indicators": [
    {{
      "name": "is_loc" or "is_industry" or "is_topic",
      "value": "string",
      "negated": true or false
    }}
  ],
  "statement": "original query"
}}

Rules:
- The "topic" MUST NOT be a paraphrase of the user's query.
- The "topic" MUST be a broad subject‑matter category.
- Extract geographic regions, industries, and broad topics.
- If the user expresses a claim, set type="support_statement".
- Otherwise use type="topic_filter".
- Expand the query into a clearer guidance string.
- If no indicators apply, return an empty list.

User query:
{query}
"""

        raw = self.llm_call(semantic_prompt)

        # -------------------------------
        # Defensive JSON parsing
        # -------------------------------
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("Semantic intent must be a JSON object")
        except Exception:
            # Fail-safe: minimal intent
            return RetrievalIntent(
                type=IntentType.TOPIC_FILTER,
                guidance=query,
                grading=True,
                indicators=[],
                statement=query,
            )

        # -------------------------------
        # Defensive extraction of indicators
        # -------------------------------
        raw_inds = safe_get(data, "indicators", [])
        indicators = []
        if isinstance(raw_inds, list):
            for i in raw_inds:
                try:
                    indicators.append(
                        Indicator(
                            name=safe_str(i.get("name")),
                            value=safe_str(i.get("value")),
                            negated=bool(i.get("negated", False)),
                        )
                    )
                except Exception:
                    continue  # skip malformed indicator

        # -------------------------------
        # Negation detection
        # -------------------------------
        neg_detector = TopicNegationDetector(self.llm_call)
        neg_result = neg_detector.detect(query)
        indicators = apply_negation(indicators, neg_result)

        # -------------------------------
        # Clean guidance
        # -------------------------------
        raw_guidance = safe_get(data, "guidance", query)
        if neg_result.negated_topics:
            stripper = NegationStripper(self.llm_call)
            clean_guidance = stripper.strip(raw_guidance, neg_result)
        else:
            clean_guidance = raw_guidance

        # -------------------------------
        # Extract time intent separately
        # -------------------------------
        time_data = self.extract_time_intent(query)
        recency_days = time_data["recency_days"]
        time_bucket = time_data["time_bucket"]

        # -------------------------------
        # Build intent object
        # -------------------------------
        intent_type = IntentType(safe_get(data, "type", "topic_filter"))
        grading = bool(safe_get(data, "grading", True))

        intent = RetrievalIntent(
            type=intent_type,
            guidance=clean_guidance,
            grading=grading,
            indicators=indicators,
            statement=query,
            recency_days=recency_days,
            time_bucket=time_bucket,
        )

        # -------------------------------
        # Apply time post-processing
        # -------------------------------
        intent = apply_time_postprocessing(intent)

        return intent
