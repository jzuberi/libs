from collections import defaultdict, Counter
from typing import Dict, List, Tuple


import json
from pathlib import Path

class HybridSearchEngine:
    def __init__(self, persist_path: str | None = None):
        self.persist_path = Path(persist_path) if persist_path else None

        self.documents = {}
        self.inverted_index = {}
        self.doc_lengths = {}

        # If persistence exists, load it
        if self.persist_path and self.persist_path.exists():
            self._load()

    def _save(self):
        if not self.persist_path:
            return

        data = {
            "documents": self.documents,
            "inverted_index": self.inverted_index,
            "doc_lengths": self.doc_lengths,
        }

        with open(self.persist_path, "w") as f:
            json.dump(data, f)

    def _load(self):
        with open(self.persist_path, "r") as f:
            data = json.load(f)

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

        # NEW: persist after update
        self._save()


    # -----------------------------
    # Simple keyword search
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
                # very simple score: sum of term frequencies
                scores[doc_id] += float(tf)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:k]
    
    # -----------------------------
    # Hybrid search (vector + keyword)
    # -----------------------------
    def hybrid_search(
        self,
        query: str,
        ragstore,
        k: int = 10,
        filter: dict | None = None,
    ) -> list[str]:

        # 1. Vector search via RAGStore (not backend)
        vec_results = ragstore.query_vector(query, k=k, filter=filter)
        vec_hits = list(zip(vec_results["ids"], vec_results["distances"]))


        # 2. Keyword search (filter-aware)
        allowed_ids = vec_results["ids"] if filter else None

        kw_hits = self.keyword_search(query, k=k, allowed_ids=allowed_ids)

        # 3. Fuse
        fused = self._fuse_rrf(vec_hits, kw_hits, k=k)
        return fused


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

        # vector hits
        for rank, (doc_id, _) in enumerate(vec_hits):
            scores[doc_id] += 1 / (c + rank)

        # keyword hits
        for rank, (doc_id, _) in enumerate(kw_hits):
            scores[doc_id] += 1 / (c + rank)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_id for doc_id, _ in ranked[:k]]

