import json
from .cleaners import strip_think_blocks

def extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found")
    return text[start:end+1]


class DefaultJSONParser:
    """
    JSON parser with graceful fallback behavior.
    Ensures raw input is always treated as a string.
    """

    def parse(self, raw, expected_keys=None) -> dict:
        # --- NEW: normalize backend output ---
        if not isinstance(raw, str):
            raw = str(raw)

        cleaned = strip_think_blocks(raw)

        # Try to extract JSON
        try:
            extracted = extract_json(cleaned)
            data = json.loads(extracted)
        except Exception:
            # Fallback: return raw text wrapped in JSON
            return {"error": "invalid_json", "raw": cleaned}

        # If no expected keys, return parsed JSON
        if not expected_keys:
            return data

        # Fill missing keys with None instead of raising
        for key in expected_keys:
            if key not in data:
                data[key] = None

        return data
