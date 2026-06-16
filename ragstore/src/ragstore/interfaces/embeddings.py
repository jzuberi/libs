from typing import List, Protocol
from .types import Vector


class EmbeddingService(Protocol):
    """
    Abstract interface for embedding providers.
    OllamaEmbeddingService, OpenAIEmbeddingService, etc. will implement this.
    """

    def embed_texts(self, texts: List[str]) -> List[Vector]:
        ...

    def embed_query(self, text: str) -> Vector:
        ...
