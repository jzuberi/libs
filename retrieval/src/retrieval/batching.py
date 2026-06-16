# libs/retrieval/src/retrieval/batching.py

def adaptive_batches(
    backend,
    query: str,
    filter_dict=None,
    batch_size: int = 4,
    max_batches: int = 2,
):
    offset = 0

    for _ in range(max_batches):
        batch = backend.query(
            query,
            offset=offset,
            limit=batch_size,
            filter=filter_dict,   
        )

        if not batch:
            break

        yield batch
        offset += batch_size
