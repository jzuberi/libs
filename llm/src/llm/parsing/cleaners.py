import re

def strip_think_blocks(text: str) -> str:
    """
    Remove <think>...</think> blocks.
    Directly adapted from your working code.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip()
