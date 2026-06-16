from typing import Dict, Any, Tuple, List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_distances

from .distance import get_cos_distance_matrix
from .utils import get_connected_components


def sweep_thresholds(
    dmatrix,
    dfm: pd.DataFrame,
    indices: List,
    max_thresholds: int = 100,
    relative_threshold: float = 0.05,
    min_steps: int = 5,
    verbose: bool = False,
) -> Tuple[list[list], float | None, list[int]]:
    """
    Sweep cosine distance thresholds and stop when the relative
    change in component count becomes small.

    Returns:
        - clusters (list of components)
        - convergence_threshold (float or None)
        - component_history (list of component counts)
    """
    flat = sorted([d for row in dmatrix for d in row if d != 0])

    if len(flat) > max_thresholds:
        thresholds = np.linspace(min(flat), max(flat), max_thresholds)
    else:
        thresholds = flat

    component_history: list[int] = []
    best_subgraphs: list[list] = []
    convergence_threshold: float | None = None

    for thr in thresholds:
        edges: list[list] = []

        for i in indices:
            for j in indices:
                if j > i and dfm.loc[i, j] < thr:
                    edges.append([i, j])

        subgraphs = get_connected_components(edges)
        num_components = len(subgraphs)

        if verbose:
            print(f"thr={thr:.4f}, components={num_components}")

        best_subgraphs = subgraphs
        component_history.append(num_components)

        if len(component_history) < 2:
            continue

        C_t = component_history[-1]
        C_t1 = component_history[-2]
        rel_change = abs(C_t - C_t1) / (C_t1 + 1e-9)

        if verbose:
            print(f"rel_change={rel_change:.4f}")

        if rel_change < relative_threshold and len(component_history) >= min_steps:
            convergence_threshold = thr
            if verbose:
                print("Converged — relative change small.")
            break

    return best_subgraphs, convergence_threshold, component_history


def build_subgraph_response(
    clusters: list[list],
    convergence_threshold: float | None,
    component_history: list[int],
) -> Dict[str, Any]:
    """
    Build the structured return object.

    ID-safe: preserves arbitrary IDs (URLs, UUIDs, doc_ids, etc.).
    """
    cleaned_clusters = [[x for x in sub] for sub in clusters]

    largest_cluster = (
        sorted(max(cleaned_clusters, key=len)) if cleaned_clusters else []
    )

    return {
        "clusters": cleaned_clusters,
        "largest_cluster": largest_cluster,
        "convergence_threshold": convergence_threshold,
        "component_history": component_history,
    }


def compute_distance_df(content_list: list[str], lm_client) -> tuple:
    """
    Compute cosine distance matrix (via lm_client) and return:
        - dmatrix (np.ndarray)
        - dfm (pandas DataFrame)
        - indices (list of row indices)
    """
    dmatrix = get_cos_distance_matrix(content_list, lm_client=lm_client)

    if len(dmatrix) == 0:
        return None, None, None

    n = len(content_list)
    indices = list(range(n))
    dfm = pd.DataFrame(dmatrix, index=indices, columns=indices)

    return dmatrix, dfm, [int(i) for i in indices]


def get_subgraph_strs(
    content_list: list[str],
    lm_client,
    max_thresholds: int = 100,
    relative_threshold: float = 0.05,
    min_steps: int = 5,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    High-level orchestrator for subgraph discovery from raw strings.
    """
    dmatrix, dfm, indices = compute_distance_df(content_list, lm_client=lm_client)

    if dmatrix is None:
        return {
            "clusters": [],
            "largest_cluster": [],
            "convergence_threshold": None,
            "component_history": [],
        }

    clusters, thr, history = sweep_thresholds(
        dmatrix,
        dfm,
        indices,
        max_thresholds=max_thresholds,
        relative_threshold=relative_threshold,
        min_steps=min_steps,
        verbose=verbose,
    )

    return build_subgraph_response(clusters, thr, history)


def get_subgraph_from_embeddings(
    embeddings_dict: Dict[str, Any],
    max_thresholds: int = 100,
    relative_threshold: float = 0.05,
    min_steps: int = 5,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    High-level orchestrator for subgraph discovery using arbitrary IDs
    (URLs, doc_ids, chunk_ids, etc.) and precomputed embeddings.
    """
    if "embeddings" not in embeddings_dict:
        return {
            "clusters": [],
            "largest_cluster": [],
            "convergence_threshold": None,
            "component_history": [],
        }

    raw = embeddings_dict["embeddings"]

    if isinstance(raw, dict):
        ids = list(raw.keys())
        vectors = list(raw.values())
    else:
        ids = list(range(len(raw)))
        vectors = raw

    if len(vectors) == 0:
        return {
            "clusters": [],
            "largest_cluster": [],
            "convergence_threshold": None,
            "component_history": [],
        }

    vectors = np.stack(vectors, axis=0)
    dmatrix = cosine_distances(vectors)

    dfm = pd.DataFrame(dmatrix, index=ids, columns=ids)
    indices = ids

    clusters, thr, history = sweep_thresholds(
        dmatrix,
        dfm,
        indices,
        max_thresholds=max_thresholds,
        relative_threshold=relative_threshold,
        min_steps=min_steps,
        verbose=verbose,
    )

    return build_subgraph_response(clusters, thr, history)
