from typing import Any, Dict, List, Optional
import tempfile
import os

import lancedb

from ragstore.interfaces.backend import VectorBackend
from ragstore.interfaces.types import Vector, QueryResult
from ragstore.backends.registry import _resolve_path


class LanceBackend(VectorBackend):

    # ============================================================
    # Ephemeral backend (default)
    # ============================================================
    def __init__(self):
        print(
            "[Warning] LanceBackend initialized without registry. "
            "Using a temporary backend that will be deleted when closed "
            "or when the object is destroyed."
        )

        self._tempdir = tempfile.TemporaryDirectory(dir="./")
        self.path = self._tempdir.name
        self._is_temp = True
        self._closed = False

        self.collection_name = "temp"
        self.vector_size = 768

        self.db = lancedb.connect(self.path)
        self.table = (
            self.db.open_table(self.collection_name)
            if self.collection_name in self.db.table_names()
            else None
        )

    # ============================================================
    # Persistent backend via registry
    # ============================================================
    @classmethod
    def from_registry(cls, project: str, collection: str, vector_size: int):
        path = _resolve_path(project, collection)

        obj = cls.__new__(cls)
        obj._is_temp = False
        obj._tempdir = None
        obj._closed = False

        obj.collection_name = collection
        obj.vector_size = vector_size
        obj.path = str(path)

        obj.db = lancedb.connect(obj.path)
        obj.table = (
            obj.db.open_table(collection)
            if collection in obj.db.table_names()
            else None
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
            raise RuntimeError("This LanceBackend has been closed.")

    # ============================================================
    # Cleanup
    # ============================================================
    def close(self):
        if not self._closed:
            if self._is_temp and self._tempdir:
                self._tempdir.cleanup()
                self._tempdir = None

            self.db = None
            self.table = None
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
        return f"<LanceBackend {mode}, {status}, path={self.path}>"

    # ============================================================
    # Count
    # ============================================================
    def count(self, filter: Optional[Dict[str, Any]] = None) -> int:
        self._ensure_open()
        if self.table is None:
            return 0
        if filter is None:
            return self.table.count_rows()
        where = self._convert_filter_to_where(filter)
        return self.table.count_rows(where=where)

    # ============================================================
    # Upsert
    # ============================================================
    def upsert(self, ids, vectors, metadatas, documents):
        self._ensure_open()

        rows = []
        for idx, vec, meta, doc in zip(ids, vectors, metadatas, documents):
            rows.append(
                {
                    "id": str(idx),
                    "vector": list(vec),
                    "text": doc,
                    "original_id": idx,
                    "metadata": dict(meta),
                }
            )

        if not rows:
            return

        if self.table is None:
            self.table = self.db.create_table(
                self.collection_name,
                data=rows,
                mode="overwrite",
            )
        else:
            self.table.add(rows)

    # ============================================================
    # Query (vector search)
    # ============================================================
    def query(self, vector: Vector, k: int = 10, filter: Optional[Dict[str, Any]] = None) -> QueryResult:
        self._ensure_open()
        if self.table is None:
            return QueryResult([], [], [], [], [])

        search = self.table.search(vector).limit(k)

        if filter is not None:
            where = self._convert_filter_to_where(filter)
            search = search.where(where)

        results = search.to_list()

        ids = [str(r["id"]) for r in results]
        vectors = [r["vector"] for r in results]
        metadatas = [r.get("metadata", {}) for r in results]
        documents = [r.get("text") for r in results]
        distances = [r.get("score") for r in results] if results else []

        return QueryResult(ids, vectors, metadatas, documents, distances)

    # ============================================================
    # Filter conversion (dict → SQL WHERE)
    # ============================================================
    def _convert_filter_to_where(self, f: Optional[Dict[str, Any]]) -> Optional[str]:
        if f is None:
            return None

        def build_clause(d: Dict[str, Any]) -> str:
            parts = []
            for key, val in d.items():
                col = f"metadata['{key}']"
                if isinstance(val, dict):
                    if "$gte" in val:
                        parts.append(f"{col} >= {repr(val['$gte'])}")
                    if "$gt" in val:
                        parts.append(f"{col} > {repr(val['$gt'])}")
                    if "$lte" in val:
                        parts.append(f"{col} <= {repr(val['$lte'])}")
                    if "$lt" in val:
                        parts.append(f"{col} < {repr(val['$lt'])}")
                    if "$in" in val:
                        vals = ", ".join(repr(v) for v in val["$in"])
                        parts.append(f"{col} IN ({vals})")
                    if "$nin" in val:
                        vals = ", ".join(repr(v) for v in val["$nin"])
                        parts.append(f"{col} NOT IN ({vals})")
                    continue
                parts.append(f"{col} = {repr(val)}")
            return " AND ".join(parts) if parts else "TRUE"

        if "$and" in f:
            return " AND ".join(f"({self._convert_filter_to_where(x)})" for x in f["$and"])

        if "$or" in f:
            return " OR ".join(f"({self._convert_filter_to_where(x)})" for x in f["$or"])

        return build_clause(f)

    # ============================================================
    # Delete
    # ============================================================
    def delete(self, ids: List[str]) -> None:
        self._ensure_open()
        if not ids or self.table is None:
            return
        where = f"id IN ({', '.join(repr(str(i)) for i in ids)})"
        self.table.delete(where=where)

    # ============================================================
    # Reset
    # ============================================================
    def reset(self):
        self._ensure_open()
        if self.collection_name in self.db.table_names():
            self.db.drop_table(self.collection_name)
        self.table = None

    # ============================================================
    # Fetch by ID (scanner API)
    # ============================================================
    def get_by_ids(self, ids: list[str]):
        self._ensure_open()
        if not ids or self.table is None:
            return []

        where = f"id IN ({', '.join(repr(str(i)) for i in ids)})"

        df = self.table.scanner(filter=where).to_pandas()

        out = []
        for _, r in df.iterrows():
            out.append({
                "id": r["id"],
                "text": r.get("text"),
                "metadata": r.get("metadata", {}),
            })
        return out

    # ============================================================
    # Update metadata
    # ============================================================
    def update_metadata(self, point_id: str, metadata: dict):
        self._ensure_open()
        if self.table is None:
            return

        where = f"id = {repr(str(point_id))}"
        df = self.table.scanner(filter=where).to_pandas()
        if df.empty:
            return

        row = df.iloc[0].to_dict()
        row["metadata"] = metadata

        self.table.delete(where=where)
        self.table.add([row])

    # ============================================================
    # Update metadata batch
    # ============================================================
    def update_metadata_batch(self, ids: list[str], metadata: dict):
        self._ensure_open()
        if not ids or self.table is None:
            return

        where = f"id IN ({', '.join(repr(str(i)) for i in ids)})"
        df = self.table.scanner(filter=where).to_pandas()
        if df.empty:
            return

        new_rows = []
        for _, r in df.iterrows():
            row = r.to_dict()
            row["metadata"] = metadata
            new_rows.append(row)

        self.table.delete(where=where)
        self.table.add(new_rows)

    # ============================================================
    # Existing doc IDs
    # ============================================================
    def existing_doc_ids(self, doc_ids: List[str]) -> set[str]:
        self._ensure_open()
        if not doc_ids or self.table is None:
            return set()

        where = f"metadata['doc_id'] IN ({', '.join(repr(v) for v in doc_ids)})"
        df = self.table.scanner(filter=where).to_pandas()

        out = set()
        for _, r in df.iterrows():
            meta = r.get("metadata", {})
            if "doc_id" in meta:
                out.add(meta["doc_id"])
        return out

    # ============================================================
    # Get vectors (scanner API)
    # ============================================================
    def get_vectors(self, ids: list[str]) -> list[Optional[list[float]]]:
        self._ensure_open()
        if not ids or self.table is None:
            return []

        where = f"id IN ({', '.join(repr(str(i)) for i in ids)})"
        df = self.table.scanner(filter=where).to_pandas()

        vec_map = {str(r["id"]): r["vector"] for _, r in df.iterrows()}
        return [vec_map.get(str(id_), None) for id_ in ids]
