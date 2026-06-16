import requests
from typing import List
from ragstore.interfaces.embeddings import EmbeddingService
from ragstore.interfaces.types import Vector


class LMStudioEmbeddingService(EmbeddingService):
    """
    EmbeddingService wrapper around LM Studio's OpenAI-compatible
    /v1/embeddings endpoint.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text-v1.5",
        url: str = "http://localhost:1234/v1/embeddings",
    ):
        self.model = model
        self.url = url
        self.headers = {"Authorization": "Bearer lm-studio"}

    # ---------------------------------------------------------
    # Core embedding methods
    # ---------------------------------------------------------
    def embed_texts(self, texts: List[str]) -> List[Vector]:
        # LM Studio supports batching natively
        response = requests.post(
            self.url,
            headers=self.headers,
            json={"input": texts, "model": self.model},
        )
        response.raise_for_status()
        data = response.json()["data"]
        return [item["embedding"] for item in data]

    def embed_query(self, text: str) -> Vector:
        # Reuse the same endpoint for single text
        response = requests.post(
            self.url,
            headers=self.headers,
            json={"input": [text], "model": self.model},
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
