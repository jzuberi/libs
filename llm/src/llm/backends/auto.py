import requests

from .ollama_backend import OllamaBackend
from .lmstudio_backend import LMStudioBackend

LMSTUDIO_BASE_URL = "http://localhost:1234/v1"

def lmstudio_is_serving(model_name: str) -> bool:
    """
    Check if LM Studio is serving the given model.
    Adapted from your working code.
    """
    url = f"{LMSTUDIO_BASE_URL}/chat/completions"
    headers = {"Authorization": "Bearer lmstudio"}

    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Are you ready?"}],
        "temperature": 0.5,
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException:
        return False


def get_backend(model: str, timeout: int = 10):
    """
    Auto-select backend based on whether LM Studio is serving the model.
    """
    if lmstudio_is_serving(model):
        return LMStudioBackend(model=model, timeout=timeout)
    return OllamaBackend(model=model, timeout=timeout)
