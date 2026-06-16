from typing import Dict, Any, List

import numpy as np
from sklearn.metrics.pairwise import cosine_distances


def get_cos_distance_matrix_from_embeddings(
    embeddings_dict: Dict[str, Any],
    batch_size: int = 1000,
) -> Dict[str, Any] | list:
    """
    Compute a cosine distance matrix from an embeddings dict.

    Supports both:
        {"embeddings": {id: vector}}
        {"embeddings": [vector, vector, ...]}

    Returns:
        - [] if no embeddings
        - {"ids": [...], "distance_matrix": np.ndarray} otherwise
    """
    if not embeddings_dict or "embeddings" not in embeddings_dict:
        return []

    raw = embeddings_dict["embeddings"]

    if isinstance(raw, dict):
        ids = list(raw.keys())
        vectors = list(raw.values())
    else:
        ids = list(range(len(raw)))
        vectors = raw

    if len(vectors) == 0:
        return []

    vectors = np.stack(vectors, axis=0)
    distances = cosine_distances(vectors)

    return {
        "ids": ids,
        "distance_matrix": distances,
    }


def get_cos_distance_matrix(
    str_list: List[str],
    lm_client,
    batch_size: int = 1000,
):
    """
    Compute cosine distance matrix using an embedding client.

    Args:
        str_list: list of strings to embed.
        lm_client: object with an `embed_texts(list[str]) -> list[vector]` method.
        batch_size: max batch size per embedding call.

    Returns:
        - [] if no strings
        - np.ndarray distance matrix otherwise
    """
    if not isinstance(str_list, list) or len(str_list) == 0:
        return []

    embeddings = []

    for i in range(0, len(str_list), batch_size):
        batch = str_list[i : i + batch_size]
        batch_embeds = lm_client.embed_texts(batch)
        embeddings.extend(batch_embeds)

    distances = cosine_distances(embeddings)
    return distances
