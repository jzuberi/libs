from collections import defaultdict, Counter
from typing import Dict, List, Tuple

import json, os
from pathlib import Path
import numpy as np


def mmr(
    query_vec,
    doc_ids,
    doc_vecs,
    k=10,
    lambda_mult=0.5
):
    cleaned = [(i, v) for i, v in zip(doc_ids, doc_vecs) if isinstance(v, (list, np.ndarray))]
    if not cleaned:
        return doc_ids[:k]

    doc_ids, doc_vecs = zip(*cleaned)
    doc_vecs = [np.array(v, dtype=float) for v in doc_vecs]

    if len(doc_vecs) == 0:
        return doc_ids[:k]

    doc_sims = np.array([np.dot(query_vec, dv) for dv in doc_vecs])

    try:
        doc_vecs_np = np.vstack(doc_vecs)
    except ValueError:
        return doc_ids[:k]

    selected = []
    remaining = list(range(len(doc_ids)))

    for _ in range(min(k, len(doc_ids))):
        if not remaining:
            break

        if not selected:
            idx = remaining[np.argmax(doc_sims[remaining])]
            selected.append(idx)
            remaining.remove(idx)
            continue

        selected_vecs = doc_vecs_np[selected]
        sim_to_selected = np.max(selected_vecs @ doc_vecs_np[remaining].T, axis=0)

        mmr_scores = (
            lambda_mult * doc_sims[remaining]
            - (1 - lambda_mult) * sim_to_selected
        )

        idx = remaining[np.argmax(mmr_scores)]
        selected.append(idx)
        remaining.remove(idx)

    return [doc_ids[i] for i in selected]


class HybridSearchEngine:
    def __init__(self, persist_path: str | None = None):
        self.persist_path = Path(persist_path) if persist_path else None

        self.documents = {}
        self.inverted_index = {}
        self.doc_lengths = {}

        self._dirty = False  # NEW: track unsaved changes

        if self.persist_path and self.persist_path.exists():
            self._load()

    # -----------------------------
    # Atomic save (unchanged)
    # -----------------------------
    def _save(self):
        tmp_path = Path(str(self.persist_path) + ".tmp")

        data = {
            "documents": self.documents,
            "inverted_index": self.inverted_index,
            "doc_lengths": self.doc_lengths,
        }

        with open(tmp_path, "w") as f:
            json.dump(data, f)

        os.replace(tmp_path, self.persist_path)

    # -----------------------------
    # NEW: flush method
    # -----------------------------
    def flush(self):
        if self._dirty:
            self._save()
            self._dirty = False

    # -----------------------------
    # Load
    # -----------------------------
        
    def _load(self):
        try:
            with open(self.persist_path, "r") as f:
                data = json.load(f)
        except Exception:
            print("Hybrid index corrupted or unreadable — rebuilding empty index...")
            self.documents = {}
            self.inverted_index = {}
            self.doc_lengths = {}
            self._save()   # write a clean empty index
            return

        self.documents = data["documents"]
        self.inverted_index = data["inverted_index"]
        self.doc_lengths = data["doc_lengths"]


    # -----------------------------
    # Tokenization
    # -----------------------------
    def _tokenize(self, text: str) -> List[str]:
        return [
            t.lower()
            for t in text.replace(".", " ").replace(",", " ").split()
            if t.strip()
        ]

    # -----------------------------
    # Indexing
    # -----------------------------
    def add_document(self, doc_id: str, text: str) -> None:
        tokens = self._tokenize(text)
        self.documents[doc_id] = text
        self.doc_lengths[doc_id] = len(tokens)

        counts = Counter(tokens)
        for tok, freq in counts.items():
            if tok not in self.inverted_index:
                self.inverted_index[tok] = {}
            self.inverted_index[tok][doc_id] = freq

        # OLD (removed): self._save()
        self._dirty = True  # NEW: mark as needing save

    # -----------------------------
    # Keyword search
    # -----------------------------
    def keyword_search(
        self,
        query: str,
        k: int = 10,
        allowed_ids: List[str] | None = None,
    ) -> List[Tuple[str, float]]:
        tokens = self._tokenize(query)
        scores: Dict[str, float] = defaultdict(float)

        for tok in tokens:
            if tok not in self.inverted_index:
                continue

            postings = self.inverted_index[tok]
            for doc_id, tf in postings.items():
                if allowed_ids is not None and doc_id not in allowed_ids:
                    continue
                scores[doc_id] += float(tf)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]

    # -----------------------------
    # Hybrid search
    # -----------------------------
    def hybrid_search(
        self,
        query: str,
        ragstore,
        k: int = 10,
        filter: dict | None = None,
        expansion_factor: int = 3,
        diversify: bool = False,
    ) -> list[str]:

        expanded_k = k * expansion_factor

        vec_results = ragstore.query_vector(query, k=expanded_k, filter=filter)
        vec_ids = vec_results["ids"]
        vec_dists = vec_results["distances"]

        kw_hits = self.keyword_search(query, k=expanded_k, allowed_ids=None)

        fused_ids = self._fuse_rrf(
            list(zip(vec_ids, vec_dists)),
            kw_hits,
            k=expanded_k
        )

        if diversify:
            query_vec = ragstore.embeddings.embed_query(query)
            doc_vecs = ragstore.backend.get_vectors(fused_ids)

            diversified = mmr(
                query_vec=query_vec,
                doc_ids=fused_ids,
                doc_vecs=doc_vecs,
                k=k,
                lambda_mult=0.75
            )
            return diversified

        return fused_ids[:k]

    # -----------------------------
    # RRF fusion
    # -----------------------------
    def _fuse_rrf(
        self,
        vec_hits: list[tuple[str, float]],
        kw_hits: list[tuple[str, float]],
        k: int = 10,
        c: int = 60,
    ) -> list[str]:
        scores = defaultdict(float)

        for rank, (doc_id, _) in enumerate(vec_hits):
            scores[doc_id] += 1 / (c + rank)

        for rank, (doc_id, _) in enumerate(kw_hits):
            scores[doc_id] += 1 / (c + rank)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in ranked[:k]]
