from typing import Any, Dict, List, Optional
import uuid
import tempfile

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Range,
)

from ragstore.interfaces.backend import VectorBackend
from ragstore.interfaces.types import Vector, QueryResult
from ragstore.backends.registry import _resolve_path


class QdrantBackend(VectorBackend):

    # ============================================================
    # Ephemeral backend (default)
    # ============================================================
    def __init__(self):
        """
        Ephemeral backend:
        - Creates a temporary directory
        - Auto-deletes on exit
        - Warns the user
        """
        print(
            "[Warning] QdrantBackend initialized without registry. "
            "Using a temporary backend that will be deleted when closed "
            "or when the object is destroyed."
        )

        self._tempdir = tempfile.TemporaryDirectory(dir="./")
        self.path = self._tempdir.name
        self._is_temp = True
        self._closed = False

        self.collection_name = "temp"
        self.vector_size = 768

        self.client = QdrantClient(path=self.path)

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )

    # ============================================================
    # Persistent backend via registry
    # ============================================================
    @classmethod
    def from_registry(cls, project: str, collection: str, vector_size: int):
        """
        Persistent backend:
        - Uses registry path
        - Never temporary
        """
        path = _resolve_path(project, collection)

        obj = cls.__new__(cls)  # bypass __init__
        obj._is_temp = False
        obj._tempdir = None
        obj._closed = False

        obj.collection_name = collection
        obj.vector_size = vector_size
        obj.path = str(path)

        obj.client = QdrantClient(path=obj.path)

        if not obj.client.collection_exists(collection):
            obj.client.recreate_collection(
                collection_name=collection,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

        return obj

    # ============================================================
    # Context manager support
    # ============================================================
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ============================================================
    # Internal guard
    # ============================================================
    def _ensure_open(self):
        if self._closed:
            raise RuntimeError(
                "This QdrantBackend has been closed and can no longer be used."
            )

    # ============================================================
    # Cleanup
    # ============================================================
    def close(self):
        """Explicit cleanup for temporary backends."""
        if not self._closed:
            if self._is_temp and self._tempdir:
                self._tempdir.cleanup()
                self._tempdir = None

            self.client = None
            self._closed = True

    def __del__(self):
        try:
            self.close()
        except:
            pass

    # ============================================================
    # Representation
    # ============================================================
    def __repr__(self):
        mode = "temp" if self._is_temp else "persistent"
        status = "closed" if self._closed else "open"
        return f"<QdrantBackend {mode}, {status}, path={self.path}>"

    # ============================================================
    # Count
    # ============================================================
    def count(self, filter: Optional[Dict[str, Any]] = None) -> int:
        self._ensure_open()
        result = self.client.count(
            collection_name=self.collection_name,
            exact=True,
            count_filter=self._convert_filter(filter) if filter else None,
        )
        return result.count

    # ============================================================
    # Upsert
    # ============================================================
    def upsert(self, ids, vectors, metadatas, documents):
        self._ensure_open()

        points = []
        for idx, vec, meta, doc in zip(ids, vectors, metadatas, documents):
            payload = dict(meta)
            payload["text"] = doc
            payload["original_id"] = idx

            points.append(
                PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_DNS, idx)),
                    vector=vec,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    # ============================================================
    # Query
    # ============================================================
    def query(
        self,
        vector: Vector,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:

        self._ensure_open()

        qdrant_filter = self._convert_filter(filter)

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=k,
            query_filter=qdrant_filter,
            with_vectors=False,
        )

        ids = [str(p.id) for p in results.points]
        vectors = [p.vector for p in results.points]
        metadatas = [p.payload for p in results.points]
        documents = [p.payload.get("text") for p in results.points]
        distances = [p.score for p in results.points]

        return QueryResult(
            ids=ids,
            vectors=vectors,
            metadatas=metadatas,
            documents=documents,
            distances=distances,
        )

    # ============================================================
    # Filter conversion
    # ============================================================
    def _convert_filter(self, f: Optional[Dict[str, Any]]) -> Optional[Filter]:
        if f is None:
            return None

        if "$and" in f:
            return Filter(must=[self._convert_filter(x) for x in f["$and"]])

        if "$or" in f:
            return Filter(should=[self._convert_filter(x) for x in f["$or"]])

        conditions: List[FieldCondition] = []

        for key, val in f.items():

            if isinstance(val, dict) and any(op in val for op in ["$gte", "$gt", "$lte", "$lt"]):
                r = Range(
                    gte=val.get("$gte"),
                    gt=val.get("$gt"),
                    lte=val.get("$lte"),
                    lt=val.get("$lt"),
                )
                conditions.append(FieldCondition(key=key, range=r))
                continue

            if isinstance(val, dict) and "$in" in val:
                conditions.append(FieldCondition(key=key, match=MatchAny(any=val["$in"])))
                continue

            if isinstance(val, dict) and "$nin" in val:
                conditions.append(FieldCondition(key=key, match=MatchAny(must_not=val["$nin"])))
                continue

            conditions.append(FieldCondition(key=key, match=MatchValue(value=val)))

        return Filter(must=conditions)

    # ============================================================
    # Delete
    # ============================================================
    def delete(self, ids: List[str]) -> None:
        self._ensure_open()
        if not ids:
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=ids,
        )

    # ============================================================
    # Reset
    # ============================================================
    def reset(self):
        self._ensure_open()

        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )

    # ============================================================
    # Fetch by ID
    # ============================================================
    def get_by_ids(self, ids: list[str]):
        self._ensure_open()

        results = self.client.retrieve(
            collection_name=self.collection_name,
            ids=ids,
            with_vectors=False,
            with_payload=True,
        )

        out = []
        for r in results:
            out.append({
                "id": r.id,
                "text": r.payload.get("text"),
                "metadata": r.payload,
            })
        return out

    def update_metadata(self, point_id: str, metadata: dict):
        self._ensure_open()
        self.client.set_payload(
            collection_name=self.collection_name,
            payload=metadata,
            points=[point_id],
        )

    def update_metadata_batch(self, ids: list[str], metadata: dict):
        self._ensure_open()
        self.client.set_payload(
            collection_name=self.collection_name,
            payload=metadata,
            points=ids,
        )

    def existing_doc_ids(self, doc_ids: List[str]) -> set[str]:
        scroll_filter = Filter(
            must=[
                FieldCondition(
                    key="doc_id",
                    match=MatchAny(any=doc_ids)
                )
            ]
        )

        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=scroll_filter,
            limit=len(doc_ids),
            with_payload=True,
            with_vectors=False,
        )

        return {p.payload["doc_id"] for p in points}
