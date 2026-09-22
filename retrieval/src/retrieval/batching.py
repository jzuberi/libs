# libs/retrieval/src/retrieval/batching.py

def adaptive_batches(
    backend,
    query: str,
    filter_dict=None,
    batch_size: int = 4,
    max_batches: int = 2,
    diversify: bool = False,
):
    """
    Adaptive batching that respects backend's expanded retrieval pool.
    Continues yielding until backend runs out of results.
    """

    offset = 0

    for _ in range(max_batches):
        batch = backend.query(
            query,
            offset=offset,
            limit=batch_size,
            filter=filter_dict,
            diversify=diversify,
        )

        if not batch:
            break

        yield batch

        # If backend returned fewer than batch_size, no more results exist
        if len(batch) < batch_size:
            break

        offset += batch_size
