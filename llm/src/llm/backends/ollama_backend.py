from langchain_ollama import ChatOllama
from .base import BaseBackend

class OllamaBackend(BaseBackend):
    """
    Ollama backend using your working ChatOllama configuration.
    """

    def __init__(self, model: str, timeout: int = 10):
        self.model = model
        self.timeout = timeout

        self.client = ChatOllama(
            model=model,
            format="json",
            temperature=0,
            stop=["<|eot_id|>"],
            timeout=timeout,
            num_predict=3000,
        )

    def generate(self, prompt: str) -> str:
        msg = self.client.invoke(prompt)

        # LangChain ChatOllama returns an AIMessage
        if hasattr(msg, "content"):
            return msg.content

        # Fallback: convert to string
        return str(msg)

