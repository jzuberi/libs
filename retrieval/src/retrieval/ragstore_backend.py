import json

class RAGStoreBackend:
    """
    Adapter that makes rag_store compatible with adaptive_batches().
    Caches full search results so pagination works correctly.
    """

    def __init__(self, rag_store):
        self.rag = rag_store
        self._cache = {}

    def query(
        self,
        text_query: str,
        offset: int = 0,
        limit: int = 10,
        filter: dict | None = None,   # <-- NEW
    ):
        """
        Query the RAG store with optional metadata filtering.
        Results are cached per (query, filter) pair so pagination works.
        """

        # Cache key must include filter to avoid mixing filtered/unfiltered results
        key = (text_query, json.dumps(filter, sort_keys=True) if filter else None)

        if key not in self._cache:
            # Pass filter directly into rag_store.query()
            full_results = self.rag.query(
                text_query,
                k=200,
                filter=filter,        
            )
            self._cache[key] = full_results

        docs = self._cache[key]
        return docs[offset:offset + limit]
