from typing import Any, Dict, List, Optional
import chromadb

from ragstore.interfaces.backend import VectorBackend
from ragstore.interfaces.types import Vector, Metadata, QueryResult


class ChromaBackend(VectorBackend):
    """
    Chroma backend that mirrors the behavior of your existing Vector_db class.
    """

    def __init__(
        self,
        db_path: str,
        collection_name: str,
        embedding_function,
    ):
        """
        Equivalent to Vector_db.initialize_db()
        but wrapped in the backend interface.
        """
        self.client = chromadb.PersistentClient(path=db_path)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
        )

    # ---------------------------------------------------------
    # UPSERT (mirrors update_db)
    # ---------------------------------------------------------
    def upsert(
        self,
        ids: List[str],
        vectors: List[Vector],
        metadatas: List[Metadata],
        documents: Optional[List[str]] = None,
    ) -> None:
        """
        Direct upsert. Your Vector_db.update_db() logic will live
        in RAGStore, not here.
        """
        self.collection.upsert(
            ids=ids,
            embeddings=vectors if vectors else None,
            metadatas=metadatas,
            documents=documents,
        )

    # ---------------------------------------------------------
    # QUERY (mirrors query_vector_db)
    # ---------------------------------------------------------
    def query(
        self,
        vector: Vector,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:

        # DO NOT mutate filter here — pass it through exactly as given
        res = self.collection.query(
            query_embeddings=[vector],
            n_results=k,
            where=filter,
            include=["embeddings", "metadatas", "documents", "distances"],
        )

        return QueryResult(
            ids=res.get("ids", [[]])[0],
            vectors=res.get("embeddings", [[]])[0],
            metadatas=res.get("metadatas", [[]])[0],
            documents=res.get("documents", [[]])[0],
            distances=res.get("distances", [[]])[0],
        )


    # ---------------------------------------------------------
    # DELETE COLLECTION (mirrors delete_collection)
    # ---------------------------------------------------------
    def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> None:
        if ids:
            self.collection.delete(ids=ids)
        elif filter:
            self.collection.delete(where=filter)
        else:
            raise ValueError("delete() requires ids or filter")

    # ---------------------------------------------------------
    # COUNT (mirrors get_num_meta_db)
    # ---------------------------------------------------------
    def count(self, filter: Optional[Dict[str, Any]] = None) -> int:
        if filter is None:
            return self.collection.count()

        res = self.collection.get(where=filter)
        return len(res.get("ids", []))

    # ---------------------------------------------------------
    # STREAM EXPORT (mirrors stream_metadata_batches_by_dimension)
    # ---------------------------------------------------------
    def stream_export(self, batch_size: int = 1000):
        """
        Yields batches of (ids, vectors, metadatas, documents)
        similar to your existing streaming logic.
        """
        offset = 0

        while True:
            results = self.collection.get(
                include=["embeddings", "metadatas", "documents"],
                limit=batch_size,
                offset=offset,
            )

            ids = results.get("ids", [])
            if not ids:
                break

            yield (
                ids,
                results.get("embeddings", []),
                results.get("metadatas", []),
                results.get("documents", []),
            )

            offset += batch_size

    # ---------------------------------------------------------
    # CLOSE
    # ---------------------------------------------------------
    def close(self) -> None:
        # Chroma doesn't require explicit close
        pass
