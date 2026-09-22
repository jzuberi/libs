# ragstore/embedding_utils.py

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import hashlib
import random

from ragstore.interfaces.backend import VectorBackend


# ---------------------------------------------------------
# Helper: extract canonical document ID
# ---------------------------------------------------------
def _extract_original_id(payload: Dict[str, Any]) -> Optional[str]:
    """
    Determine the true document identifier.

    IMPORTANT:
    We IGNORE payload["original_id"] because it may contain the CHUNK ID
    (e.g., url-3), which breaks document grouping.

    Correct priority:
    1. doc_id   → your true document identifier (URL)
    2. source_id → same as doc_id in your chunker
    """
    return (
        payload.get("doc_id")
        or payload.get("source_id")
    )


# ---------------------------------------------------------
# QDRANT SCROLL RESULT NORMALIZATION
# ---------------------------------------------------------
def _normalize_scroll_result(page):
    if isinstance(page, tuple):  # old API
        points, next_offset = page
        return points, next_offset
    return page.points, page.next_page_offset  # new API


# ---------------------------------------------------------
# FILTER + SAFE SAMPLING + FETCH VECTORS
# ---------------------------------------------------------
def filter_and_fetch_chunks(
    backend: VectorBackend,
    norm_filter: Optional[Dict[str, Any]],
    limit: int,
) -> Tuple[List[Dict[str, Any]], bool, Optional[int]]:

    total = backend.count(norm_filter)

    if total <= limit:
        chunks = _scroll_with_vectors(backend, norm_filter)
        return chunks, False, None

    meta_chunks = _scroll_metadata_only(backend, norm_filter)
    sampled_ids, seed = _deterministic_weighted_sample_ids(meta_chunks, limit)

    print(f"[Warning] Sampling {limit} of {total} chunks (seed={seed})")

    sampled_chunks = _fetch_vectors_for_ids(backend, sampled_ids)
    return sampled_chunks, True, seed


# ---------------------------------------------------------
# INTERNAL HELPERS
# ---------------------------------------------------------
def _scroll_with_vectors(backend, norm_filter):
    results = []
    next_offset = None

    # Convert dict → Qdrant Filter object
    qdrant_filter = backend._convert_filter(norm_filter) if norm_filter else None

    while True:
        page = backend.client.scroll(
            collection_name=backend.collection_name,
            limit=1000,
            offset=next_offset,
            with_vectors=True,
            with_payload=True,
            scroll_filter=qdrant_filter,
        )

        points, next_offset = _normalize_scroll_result(page)

        for p in points:
            payload = p.payload or {}
            results.append({
                "id": p.id,
                "vector": np.array(p.vector),
                "metadata": payload,
                "original_id": _extract_original_id(payload),
            })

        if next_offset is None:
            break

    return results

def _scroll_metadata_only(backend, norm_filter):
    results = []
    next_offset = None

    qdrant_filter = backend._convert_filter(norm_filter) if norm_filter else None

    while True:
        page = backend.client.scroll(
            collection_name=backend.collection_name,
            limit=2000,
            offset=next_offset,
            with_vectors=False,
            with_payload=True,
            scroll_filter=qdrant_filter,
        )

        points, next_offset = _normalize_scroll_result(page)

        for p in points:
            payload = p.payload or {}
            results.append({
                "id": p.id,
                "metadata": payload,
                "original_id": _extract_original_id(payload),
            })

        if next_offset is None:
            break

    return results

def _deterministic_weighted_sample_ids(
    meta_chunks: List[Dict[str, Any]],
    limit: int,
) -> Tuple[List[str], int]:

    sorted_ids = sorted(c["id"] for c in meta_chunks)
    seed = int(hashlib.sha256(",".join(sorted_ids).encode()).hexdigest(), 16) % (2**32)
    random.seed(seed)

    doc_counts = {}
    for c in meta_chunks:
        doc_id = c["original_id"]
        doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

    weights = [doc_counts[c["original_id"]] for c in meta_chunks]

    sampled = random.choices(meta_chunks, weights=weights, k=limit)
    sampled_ids = [c["id"] for c in sampled]

    return sampled_ids, seed


def _fetch_vectors_for_ids(
    backend: VectorBackend,
    ids: List[str],
) -> List[Dict[str, Any]]:

    points = backend.client.retrieve(
        collection_name=backend.collection_name,
        ids=ids,
        with_vectors=True,
        with_payload=True,
    )

    chunks = []
    for p in points:
        payload = p.payload or {}
        chunks.append({
            "id": p.id,
            "vector": np.array(p.vector),
            "metadata": payload,
            "original_id": _extract_original_id(payload),
        })

    return chunks


# ---------------------------------------------------------
# POOL DOCUMENT EMBEDDINGS
# ---------------------------------------------------------
def pool_document_embeddings(
    chunks: List[Dict[str, Any]],
    pooling: str = "mean",
) -> Dict[str, Any]:

    grouped: Dict[str, List[np.ndarray]] = {}

    for c in chunks:
        doc_id = c["original_id"]
        if doc_id is None:
            continue
        grouped.setdefault(doc_id, []).append(c["vector"])

    doc_embeddings: Dict[str, np.ndarray] = {}

    for doc_id, vectors in grouped.items():
        arr = np.stack(vectors, axis=0)

        if pooling == "mean":
            pooled = arr.mean(axis=0)
        elif pooling == "max":
            pooled = arr.max(axis=0)
        elif pooling == "first":
            pooled = arr[0]
        elif pooling == "last":
            pooled = arr[-1]
        elif pooling == "weighted_mean":
            weights = [
                c["metadata"].get("chunk_length", 1)
                for c in chunks
                if c["original_id"] == doc_id
            ]
            weights = np.array(weights, dtype=float)
            weights /= weights.sum()
            pooled = np.average(arr, axis=0, weights=weights)
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")

        doc_embeddings[doc_id] = pooled

    doc_ids = sorted(doc_embeddings.keys())

    return {
        "embeddings": doc_embeddings,
        "doc_ids": doc_ids,
    }


# ---------------------------------------------------------
# FORMATTERS
# ---------------------------------------------------------
def format_chunk_embedding_output(
    chunks: List[Dict[str, Any]],
    sampled: bool,
    seed: Optional[int],
) -> Dict[str, Any]:

    embeddings = [c["vector"] for c in chunks]
    chunk_ids = [c["id"] for c in chunks]

    return {
        "embeddings": embeddings,
        "chunk_ids": chunk_ids,
        "sampled": sampled,
        "seed": seed,
    }


def format_document_embedding_output(
    doc_embeddings: Dict[str, np.ndarray],
    sampled: bool,
    seed: Optional[int],
    pooling: str,
) -> Dict[str, Any]:

    doc_ids = sorted(doc_embeddings.keys())

    return {
        "embeddings": doc_embeddings,
        "doc_ids": doc_ids,
        "sampled": sampled,
        "seed": seed,
        "pooling": pooling,
    }
