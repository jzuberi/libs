from langchain_community.chat_models import ChatOpenAI
from .base import BaseBackend

class LMStudioBackend(BaseBackend):
    """
    LM Studio backend using your working ChatOpenAI configuration.
    """

    def __init__(self, model: str, timeout: int = 10):
        self.model = model
        self.timeout = timeout

        self.client = ChatOpenAI(
            base_url="http://localhost:1234/v1",
            api_key="lmstudio",
            model=model,
            temperature=0.1,
            timeout=timeout,
        )
        
    def generate(self, prompt: str) -> str:
        msg = self.client.invoke(prompt)

        if hasattr(msg, "content"):
            return msg.content

        return str(msg)
