from typing import Any, Dict, List, Optional, Protocol
from .types import Vector, Metadata, QueryResult


class VectorBackend(Protocol):
    """
    Abstract interface for vector database backends.
    ChromaBackend and QdrantBackend will both implement this.
    """

    def upsert(
        self,
        ids: List[str],
        vectors: List[Vector],
        metadatas: List[Metadata],
        documents: Optional[List[str]] = None,
    ) -> None:
        ...

    def query(
        self,
        vector: Vector,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        ...

    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> None:
        ...

    def count(self, filter: Optional[Dict[str, Any]] = None) -> int:
        ...

    def stream_export(self, batch_size: int = 1000):
        """
        Yields batches of (ids, vectors, metadatas, documents).
        Used for backup/restore and migrations.
        """
        ...

    def close(self) -> None:
        ...
