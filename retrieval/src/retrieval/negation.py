# retrieval/negation.py

from pydantic import BaseModel
from typing import List

class NegatedTopic(BaseModel):
    value: str

class TopicNegationResult(BaseModel):
    negated_topics: List[NegatedTopic] = []

NEGATION_PROMPT = """
You are a negation extractor.

Your job:
- Identify concepts the user explicitly excludes.
- Treat every negated concept as a "topic".
- Do NOT classify or label the concept type.
- Only extract explicit negations. Do NOT infer or guess.
- Return JSON only.

Negation patterns include:
- "not X"
- "not about X"
- "except X"
- "without X"
- "other than X"
- "but not X"
- "excluding X"

### POSITIVE EXAMPLES

User: "US–China relations not about trade"
Output:
{{
  "negated_topics": [
    {{"value": "trade"}}
  ]
}}

User: "Energy policy but not climate change"
Output:
{{
  "negated_topics": [
    {{"value": "climate change"}}
  ]
}}

User: "AI regulation except in Europe"
Output:
{{
  "negated_topics": [
    {{"value": "Europe"}}
  ]
}}

User: "Foreign policy without NATO or the EU"
Output:
{{
  "negated_topics": [
    {{"value": "NATO"}},
    {{"value": "EU"}}
  ]
}}

User: "Politics but not Biden"
Output:
{{
  "negated_topics": [
    {{"value": "Biden"}}
  ]
}}

### NEGATIVE EXAMPLES

User: "US–China relations"
Output:
{{
  "negated_topics": []
}}

User: "What happened in politics?"
Output:
{{
  "negated_topics": []
}}

User: "Trade and technology issues"
Output:
{{
  "negated_topics": []
}}

User: "AI regulation in Europe"
Output:
{{
  "negated_topics": []
}}

### NOW PROCESS THIS QUERY

Query: "{query}"

Return format:
{{
  "negated_topics": [
    {{"value": "..."}}
  ]
}}
"""


# retrieval/negation.py

class TopicNegationDetector:
    def __init__(self, llm):
        self.llm = llm

    def detect(self, query: str) -> TopicNegationResult:
        prompt = NEGATION_PROMPT.format(query=query)
        raw = self.llm(prompt)
        return TopicNegationResult.model_validate_json(raw)



STRIP_NEGATION_PROMPT = """
You are a query rewriting assistant.

Your job:
- Remove any concepts the user explicitly negated.
- Do NOT paraphrase or expand the query.
- Do NOT add new information.
- Only remove the negated concepts.
- Return ONLY the cleaned query without any negation concept as plain text.

Negated concepts:
{negated_list}

Original query:
"{query}"

Return the cleaned query:
"""


class NegationStripper:
    def __init__(self, llm_call):
        self.llm_call = llm_call

    def strip(self, query: str, neg_result) -> str:
        negated_list = "\n".join(f"- {t.value}" for t in neg_result.negated_topics)

        prompt = STRIP_NEGATION_PROMPT.format(
            query=query,
            negated_list=negated_list
        )

        cleaned = self.llm_call(prompt)
        return cleaned.strip()
