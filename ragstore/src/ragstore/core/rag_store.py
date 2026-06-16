from typing import Any, Dict, Iterable, List, Optional

from ragstore.interfaces.backend import VectorBackend
from ragstore.interfaces.embeddings import EmbeddingService
from ragstore.interfaces.types import Vector, QueryResult
from .chunking import Chunk, normalize_chunk, ChunkNormalizationError
from .hybrid_search import HybridSearchEngine

# ragstore/rag_store.py (add these imports at the top)

from .embedding_utils import (
    filter_and_fetch_chunks,
    pool_document_embeddings,
    format_chunk_embedding_output,
    format_document_embedding_output,
)


from datetime import datetime


class RAGStore:
    def __init__(
        self,
        backend: VectorBackend,
        embeddings: EmbeddingService,
        time_field: Optional[str] = None,
    ):
        """
        RAGStore is a thin orchestration layer over a vector backend + embeddings.

        :param backend: VectorBackend implementation.
        :param embeddings: EmbeddingService implementation.
        :param time_field: Optional name of the metadata field that stores
                           document time information (e.g. 'date', 'published_at').
                           If None, this store is considered time-agnostic.
        """
        self.backend = backend
        self.embeddings = embeddings
        self._time_field = time_field

        self.hybrid = HybridSearchEngine(
            persist_path=f"{self.backend.path}/hybrid_index.json"
        )

    # ---------------------------------------------------------
    # Time capabilities
    # ---------------------------------------------------------
    @property
    def time_field(self) -> Optional[str]:
        """
        Name of the metadata field that stores document time information,
        or None if this store does not support time-based metadata.
        """
        return self._time_field

    @property
    def supports_time(self) -> bool:
        """
        Whether this store supports time-based metadata (recency).
        """
        return self._time_field is not None
    

    def _normalize_time_value(self, value):
        # Already numeric → return as-is
        if isinstance(value, (int, float)):
            return int(value)

        # ISO date string → parse
        if isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value)
                return int(dt.timestamp())
            except Exception:
                raise ValueError(f"Invalid time format for '{self.time_field}': {value}")

        # datetime object → convert
        if isinstance(value, datetime):
            return int(value.timestamp())

        raise ValueError(f"Unsupported type for time field '{self.time_field}': {type(value)}")

    def reset(self):

        self.backend.reset()

    # ---------------------------------------------------------
    # Ingestion
    # ---------------------------------------------------------
    def add_chunks(self, chunks, batch_size=500):

        ids_batch = []
        texts_batch = []
        metas_batch = []

        DEBUG = True  # turn off when stable

        total_chunks = 0
        total_batches = 0

        def flush():
            nonlocal total_batches
            if not ids_batch:
                return

            if DEBUG:
                print(f"[FLUSH] batch {total_batches+1} → {len(ids_batch)} chunks")

            vectors = self.embeddings.embed_texts(texts_batch)

            self.backend.upsert(
                ids=list(ids_batch),
                vectors=vectors,
                metadatas=list(metas_batch),
                documents=list(texts_batch),
            )

            total_batches += 1
            ids_batch.clear()
            texts_batch.clear()
            metas_batch.clear()

        # -----------------------------------------
        # Main ingestion loop
        # -----------------------------------------
        for chunk in chunks:

            total_chunks += 1

            # enforce time metadata if needed
            if self.supports_time:
                if self.time_field not in chunk.metadata:
                    raise ValueError(
                        f"Chunk {chunk.id} missing required time field '{self.time_field}'"
                    )
                raw_time = chunk.metadata[self.time_field]
                chunk.metadata[self.time_field] = self._normalize_time_value(raw_time)

            # existing logic
            self.hybrid.add_document(chunk.id, chunk.text)

            ids_batch.append(chunk.id)
            texts_batch.append(chunk.text)
            metas_batch.append(chunk.metadata)

            if len(ids_batch) >= batch_size:
                flush()

        # final flush
        flush()

        if DEBUG:
            print(f"[DONE] {total_chunks} chunks ingested in {total_batches} batches")



    # ---------------------------------------------------------
    # Core query
    # ---------------------------------------------------------

    def query(self, query: str, k: int = 10, filter=None):
        """
        Default retrieval: hybrid (vector + keyword + fusion).
        """
        ids = self.hybrid.hybrid_search(
            query=query,
            ragstore=self,
            k=k,
            filter=filter,
        )
        return self.backend.get_by_ids(ids)

    def query_vector(
        self,
        query: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        """
        Canonical query method: everything flows through this.
        """
        norm_filter = self._normalize_filter(filter)
        q_vec = self.embeddings.embed_query(query)
        return self.backend.query(q_vec, k=k, filter=norm_filter)

    # ---------------------------------------------------------
    # Convenience queries
    # ---------------------------------------------------------
    def query_by_id(
        self,
        input_id: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        """
        Fetch text for an ID, then query.
        Implementation of ID→text is app-specific; stub for now.
        """
        text = self._get_text_from_id(input_id)
        return self.query(text, k=k, filter=filter)

    def query_by_url(
        self,
        input_url: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> QueryResult:
        """
        Fetch text for a URL, then query.
        Implementation of URL→text is app-specific; stub for now.
        """
        text = self._get_text_from_url(input_url)
        return self.query(text, k=k, filter=filter)

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------
    def _normalize_filter(self, f):
        if f is None:
            return None

        # If this is an operator ($and, $or), recurse but do NOT rewrite
        if "$and" in f:
            return {"$and": [self._normalize_filter(x) for x in f["$and"]]}
        if "$or" in f:
            return {"$or": [self._normalize_filter(x) for x in f["$or"]]}

        # Otherwise this is a field filter
        out = {}
        for key, val in f.items():
            if isinstance(val, list):
                out[key] = {"$in": val}
            else:
                out[key] = val
        return out


    def _get_text_from_id(self, input_id: str) -> str:
        """
        Placeholder: wire this to your existing 'get_text_from_id' logic.
        For now, we leave it abstract so RAGStore stays backend-agnostic.
        """
        raise NotImplementedError("get_text_from_id() not implemented yet")

    def _get_text_from_url(self, input_url: str) -> str:
        """
        Placeholder: wire this to your existing 'get_text_from_url' logic.
        """
        raise NotImplementedError("get_text_from_url() not implemented yet")
    
    def add_raw_chunks(
        self,
        raw_chunks: Iterable[object],
        batch_size: int = 500,
    ) -> None:
        """
        Accepts arbitrary objects and normalizes them into Chunk.
        This is the public-friendly ingestion API.
        """

        def gen():
            for obj in raw_chunks:
                try:
                    yield normalize_chunk(obj)
                except ChunkNormalizationError as e:
                    # You can log or raise depending on your preference
                    raise

        self.add_chunks(gen(), batch_size=batch_size)

    def hybrid_query(self, query: str, k: int = 10, filter=None):
        return self.query(query, k=k, filter=filter)
        
    def edit_metadata(self, chunk_id: str, edits: dict):
        result = self.backend.get_by_ids([chunk_id])
        if not result:
            raise ValueError(f"Chunk {chunk_id} not found")

        existing = result[0].metadata
        updated = {**existing, **edits}

        self.backend.update_metadata(chunk_id, updated)

    def edit_metadata_batch(self, ids: list[str], edits: dict):
        # Fetch existing metadata
        results = self.backend.get_by_ids(ids)

        if not results:
            raise KeyError(f"No chunks found for IDs: {ids}")

        # Merge metadata per ID
        merged = []
        for r in results:
            updated = {**r["metadata"], **edits}
            merged.append((r["id"], updated))

        # Apply updates
        for point_id, metadata in merged:
            self.backend.update_metadata(point_id, metadata)

    # ---------------------------------------------------------
    # Embedding access API
    # ---------------------------------------------------------
    def get_chunk_embeddings(
        self,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 10_000,
    ) -> Dict[str, Any]:
        """
        Return chunk-level embeddings with filtering + deterministic weighted sampling.
        """
        # 1. Normalize filter
        norm_filter = self._normalize_filter(filter)

        # 2. Fetch all matching chunks (sampling handled internally)
        chunks, sampled, seed = filter_and_fetch_chunks(self.backend, norm_filter, limit)

        # 3. Format output
        return format_chunk_embedding_output(chunks, sampled, seed)

    def get_document_embeddings(
        self,
        filter: Optional[Dict[str, Any]] = None,
        limit: int = 10_000,
        pooling: str = "mean",
    ) -> Dict[str, Any]:
        """
        Return document-level embeddings by:
        - filtering chunks
        - deterministic weighted sampling
        - pooling chunk embeddings into document embeddings
        """
        # 1. Normalize filter
        norm_filter = self._normalize_filter(filter)

        # 2. Fetch all matching chunks (sampling handled internally)
        chunks, sampled, seed = filter_and_fetch_chunks(self.backend, norm_filter, limit)

        # 3. Pool into document embeddings
        pooled = pool_document_embeddings(chunks, pooling)

        # 4. Format output (IMPORTANT: pass pooled["embeddings"], not the whole dict)
        return format_document_embedding_output(
            pooled["embeddings"], 
            sampled,
            seed,
            pooling,
        )



