# libs/retrieval/src/retrieval/indicators.py

import json


class BaseIndicatorHandler:
    """
    Base class for all one-dimensional semantic validators.
    Each handler:
      - builds a JSON-based LLM prompt
      - parses {"match": bool, "reason": "..."}
      - returns True/False
    """

    label = "attribute"  # overridden by subclasses

    def build_prompt(self, text: str, value: str) -> str:
        return f"""
You are an attribute classifier. Determine whether the text is meaningfully related
to the target {self.label}, and explain your reasoning.

Target {self.label}:
{value}

Text to evaluate:
\"\"\"{text}\"\"\"

Guidelines:
- Consider semantic meaning, not exact keywords.
- If the text discusses events, actors, policies, organizations, or developments
  strongly associated with the target, it IS relevant.
- If the text only mentions the target in passing, it is NOT relevant.
- Ignore metadata, IDs, and technical wrappers. Focus only on the human‑readable content.

Respond in valid JSON with EXACTLY this structure:

{{
  "match": true or false,
  "reason": "very, very short explanation"
}}

No extra keys. No commentary outside the JSON.
"""

    def parse(self, raw: str) -> bool:
        """
        Parse JSON of the form:
        {"match": true, "reason": "..."}
        """
        try:
            data = json.loads(raw)
            return bool(data.get("match", False))
        except Exception:
            return False

    def validate(self, chunk, value: str) -> bool:
        """
        Extract text, build prompt, call LLM, parse JSON.
        """
        # Extract text from dict chunks
        if isinstance(chunk, dict) and "text" in chunk:
            text = chunk["text"]
        else:
            text = str(chunk)

        prompt = self.build_prompt(text, value)
        raw = self._call_llm(prompt)

        print('indicator response:')
        print(raw)

        # Debugging (optional)
        # print("\n[Indicator] PROMPT:\n", prompt)
        # print("[Indicator] RAW OUTPUT:\n", raw)

        return self.parse(raw)

    def _call_llm(self, prompt: str) -> str:
        raise NotImplementedError


class LocHandler(BaseIndicatorHandler):
    """
    Determines whether a chunk is about a given location.
    Example:
        value="China"
        value="United States"
    """
    label = "location"


class IndustryHandler(BaseIndicatorHandler):
    """
    Determines whether a chunk is about a given industry.
    Example:
        value="Automotive Industry"
        value="Semiconductor Industry"
    """
    label = "industry"


class TopicHandler(BaseIndicatorHandler):
    """
    Determines whether a chunk is about a given topic.
    Example:
        value="Technology"
        value="Environmental Risk"
    """
    label = "topic"
