from typing import Any, Dict, List, Optional
from pathlib import Path

from ragstore.interfaces.backend import VectorBackend
from ragstore.interfaces.types import Vector, Metadata, QueryResult
from ragstore.backends.registry import _resolve_path
import os

os.environ["CHROMA_TELEMETRY"] = "False"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["POSTHOG_DISABLED"] = "True"

import chromadb



class ChromaBackend(VectorBackend):
    """
    Chroma backend aligned with QdrantBackend + LanceBackend semantics.
    Fully compatible with RAGStore, hybrid search, metadata filtering, and MMR.
    """

    # ============================================================
    # Direct init
    # ============================================================
    def __init__(
        self,
        db_path: str,
        collection_name: str,
        vector_size: int = 768,   # stored for RAGStore compatibility
    ):
        self._closed = False
        self.path = db_path               # REQUIRED by RAGStore
        self.collection_name = collection_name
        self.vector_size = vector_size    # RAGStore expects this

        self.client = chromadb.PersistentClient(path=db_path)

        # IMPORTANT: do NOT pass embedding_function here
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
        )

    # ============================================================
    # from_registry (Qdrant/Lance-parallel)
    # ============================================================
    @classmethod
    def from_registry(
        cls,
        project: str,
        collection: str,
        vector_size: int,
    ):
        """
        Mirror LanceBackend.from_registry / QdrantBackend.from_registry:
        - Resolve path via registry
        - Instantiate ChromaBackend at that path
        """
        
        path = _resolve_path(project, collection)
        return cls(
            db_path=str(path),
            collection_name=collection,
            vector_size=vector_size,
        )
        
    def _ensure_collection(self):
        """
        Recreate the Chroma collection after deletion.
        Mirrors LanceBackend/QdrantBackend semantics.
        """
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
        )


    # ============================================================
    # Context manager support
    # ============================================================
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("This ChromaBackend has been closed.")

    # ============================================================
    # UPSERT
    # ============================================================
    def upsert(
        self,
        ids: List[str],
        vectors: List[Vector],
        metadatas: List[Metadata],
        documents: Optional[List[str]] = None,
    ) -> None:
        self._ensure_open()

        # Filter out docs whose doc_id already exists
        ids, vectors, metadatas, documents = self._filter_new_docs(
            ids, vectors, metadatas, documents
        )

        if not ids:
            return  # nothing new to insert

        fixed_meta = []
        for mid, meta in zip(ids, metadatas):
            clean = {}
            for k, v in meta.items():
                if isinstance(v, list):
                    clean[k] = v[0] if v else None
                else:
                    clean[k] = v
            clean["id"] = mid
            fixed_meta.append(clean)

        self.collection.upsert(
            ids=ids,
            embeddings=vectors if vectors else None,
            metadatas=fixed_meta,
            documents=documents,
        )

        
    def _extract_doc_ids(self, metadatas: List[Metadata]) -> List[str]:
        out = []
        for m in metadatas:
            if "doc_id" in m:
                out.append(m["doc_id"])
        return out

    def _filter_new_docs(
        self,
        ids: List[str],
        vectors: List[Vector],
        metadatas: List[Metadata],
        documents: Optional[List[str]],
    ):
        # Extract doc_ids from metadata
        incoming_doc_ids = self._extract_doc_ids(metadatas)

        # Query Chroma for existing doc_ids
        existing = self.existing_doc_ids(incoming_doc_ids)

        # Filter out any items whose doc_id already exists
        new_ids = []
        new_vectors = []
        new_metas = []
        new_docs = [] if documents is not None else None

        for i, mid in enumerate(ids):
            doc_id = incoming_doc_ids[i]
            if doc_id in existing:
                continue  # skip duplicates

            new_ids.append(mid)
            new_vectors.append(vectors[i])
            new_metas.append(metadatas[i])
            if documents is not None:
                new_docs.append(documents[i])

        return new_ids, new_vectors, new_metas, new_docs




    # ============================================================
    # QUERY
    # ============================================================
    def query(self, vector: Vector, k: int = 10, filter: Optional[Dict[str, Any]] = None) -> QueryResult:
        self._ensure_open()

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

    # ============================================================
    # GET BY IDS
    # ============================================================
    def get_by_ids(self, ids: List[str]):
        self._ensure_open()
        if not ids:
            return []

        res = self.collection.get(
            where={"id": {"$in": ids}},
            include=["metadatas", "documents"],
        )

        out = []
        for mid, meta, doc in zip(
            res.get("ids", []),
            res.get("metadatas", []),
            res.get("documents", []),
        ):
            out.append({
                "id": mid,
                "text": doc,
                "metadata": meta,
            })

        return out

    # ============================================================
    # GET VECTORS
    # ============================================================
    def get_vectors(self, ids: List[str]) -> List[Optional[List[float]]]:
        
        self._ensure_open()
        if not ids:
            return []

        res = self.collection.get(
            where={"id": {"$in": ids}},
            include=["embeddings"],
        )

        emb_map = {
            mid: emb
            for mid, emb in zip(
                res.get("ids", []),
                res.get("embeddings", []),
            )
        }

        return [emb_map.get(i, None) for i in ids]

    # ============================================================
    # UPDATE METADATA
    # ============================================================
    def update_metadata(self, point_id: str, metadata: dict):
        self._ensure_open()

        res = self.collection.get(
            where={"id": point_id},
            include=["embeddings", "documents", "metadatas"],
        )

        if not res.get("ids"):
            return

        emb = res["embeddings"][0]
        doc = res["documents"][0]
        meta = dict(res["metadatas"][0])

        # Normalize incoming metadata values
        clean = {}
        for k, v in metadata.items():
            if isinstance(v, list):
                clean[k] = v[0] if v else None
            else:
                clean[k] = v

        meta.update(clean)
        meta["id"] = point_id

        self.collection.delete(ids=[point_id])
        self.collection.add(
            ids=[point_id],
            embeddings=[emb],
            metadatas=[meta],
            documents=[doc],
        )

    # ============================================================
    # UPDATE METADATA BATCH
    # ============================================================
    def update_metadata_batch(self, ids: List[str], metadata: dict, chunk_size: int = 500):
        self._ensure_open()
        if not ids:
            return

        # Normalize incoming metadata values once
        clean_update = {}
        for k, v in metadata.items():
            if isinstance(v, list):
                clean_update[k] = v[0] if v else None
            else:
                clean_update[k] = v

        # Process in chunks to avoid SQLite "too many SQL variables"
        for i in range(0, len(ids), chunk_size):
            chunk = ids[i:i + chunk_size]

            # Fetch existing rows for this chunk
            res = self.collection.get(
                where={"id": {"$in": chunk}},
                include=["embeddings", "documents", "metadatas"],
            )

            new_ids = []
            new_embs = []
            new_docs = []
            new_metas = []

            for mid, emb, doc, meta in zip(
                res.get("ids", []),
                res.get("embeddings", []),
                res.get("documents", []),
                res.get("metadatas", []),
            ):
                m = dict(meta)
                m.update(clean_update)
                m["id"] = mid

                new_ids.append(mid)
                new_embs.append(emb)
                new_docs.append(doc)
                new_metas.append(m)

            # IMPORTANT: delete by explicit IDs (reliable)
            self.collection.delete(ids=new_ids)

            # Re-add updated rows
            self.collection.add(
                ids=new_ids,
                embeddings=new_embs,
                metadatas=new_metas,
                documents=new_docs,
            )


    # ============================================================
    # EXISTING DOC IDS
    # ============================================================
    def existing_doc_ids(self, doc_ids: List[str]) -> set[str]:
        self._ensure_open()
        if not doc_ids:
            return set()

        out = set()
        BATCH = 500  # safely below SQLite's variable limit

        for i in range(0, len(doc_ids), BATCH):
            chunk = doc_ids[i:i+BATCH]

            res = self.collection.get(
                where={"doc_id": {"$in": chunk}},
                include=["metadatas"],
            )

            for meta in res.get("metadatas", []):
                if "doc_id" in meta:
                    out.add(meta["doc_id"])

        return out


    # ============================================================
    # DELETE
    # ============================================================
    def delete(self, ids: Optional[List[str]] = None, filter: Optional[Dict[str, Any]] = None) -> None:
        self._ensure_open()
        if ids:
            self.collection.delete(ids=ids)
        elif filter:
            self.collection.delete(where=filter)
        else:
            raise ValueError("delete() requires ids or filter")

    # ============================================================
    # RESET
    # ============================================================
    def reset(self):
        self.client.delete_collection(self.collection.name)
        self._ensure_collection()

        # Delete hybrid index if it exists
        hybrid_path = Path(self.path) / "hybrid_index.json"
        if hybrid_path.exists():
            hybrid_path.unlink()

    # ============================================================
    # COUNT
    # ============================================================
    def count(self, filter: Optional[Dict[str, Any]] = None) -> int:
        self._ensure_open()
        if filter is None:
            return self.collection.count()

        res = self.collection.get(where=filter)
        return len(res.get("ids", []))

    # ============================================================
    # STREAM EXPORT
    # ============================================================
    def stream_export(self, batch_size: int = 1000):
        self._ensure_open()
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

    # ============================================================
    # CLOSE
    # ============================================================
    def close(self) -> None:
        self._closed = True
