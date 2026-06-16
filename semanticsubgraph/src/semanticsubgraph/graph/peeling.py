from typing import Dict, Any


def remove_clustered_embeddings(
    embeddings_dict: Dict[str, Any],
    subgraph_response: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Remove embeddings that belong to any of the discovered clusters.

    Args:
        embeddings_dict: {"embeddings": {id: vector} or [vector, ...]}
        subgraph_response: output of get_subgraph_from_embeddings / get_subgraph_strs

    Returns:
        New embeddings_dict with clustered IDs removed (dict form).
    """
    clustered_ids = {
        x for cluster in subgraph_response.get("clusters", []) for x in cluster
    }

    raw = embeddings_dict.get("embeddings", {})

    if isinstance(raw, dict):
        filtered = {
            doc_id: vec
            for doc_id, vec in raw.items()
            if doc_id not in clustered_ids
        }
    else:
        # if it's a list, we can't map IDs cleanly; keep behavior simple:
        # drop by index if IDs are ints
        filtered = [
            vec
            for idx, vec in enumerate(raw)
            if idx not in clustered_ids
        ]

    return {"embeddings": filtered}
