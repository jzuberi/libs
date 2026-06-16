# libs/retrieval/src/retrieval/utils.py

from .models import Indicator


def extract_text(chunk) -> str:
    """
    Normalize chunk into a plain text string.
    Supports dict chunks with a 'text' field.
    """
    if isinstance(chunk, dict) and "text" in chunk:
        return chunk["text"]
    return str(chunk)


def apply_indicators(chunk, intent, handlers) -> bool:
    """
    Apply all indicators in the intent to the chunk.
    Returns True if all indicators pass.
    """
    for ind in intent.indicators:
        handler = handlers[ind.name]
        ok = handler.validate(chunk, ind.value)

        if ind.negated:
            ok = not ok

        if not ok:
            return False

    return True


def add_indicators_to_intent(intent, indicators):
    """
    Convert indicator tuples into Indicator objects and attach them to the intent.
    """
    if not indicators:
        return intent

    for item in indicators:
        if len(item) == 2:
            name, value = item
            negated = False
        else:
            name, value, negated = item

        intent.indicators.append(
            Indicator(name=name, value=value, negated=negated)
        )

    return intent


def debug_banner(label: str, content):
    """
    Standardized debug banner for consistent logging.
    """
    print("\n==============================")
    print(label)
    print(content)
    print("==============================")
