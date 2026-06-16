

from datetime import datetime
import time
from itertools import chain
from typing import Iterable, List, Sequence, Dict
import re
import unicodedata
from urllib.parse import urlparse, unquote
from collections import defaultdict

from typing import List, Dict, Any, Callable, Set, Tuple, Union
from rapidfuzz import fuzz


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .models import RawRecord

def rawrecordlist_to_id_text_map(raw_records: List[RawRecord]) -> Dict[int, str]:
    """
    Convert a list of RawRecord into a dict mapping:
        int(raw_id) -> text
    """
    id_map: Dict[int, str] = {}

    for rec in raw_records:
        try:
            rid = int(rec.raw_id)
        except ValueError:
            raise ValueError(f"raw_id '{rec.raw_id}' is not convertible to int")

        if rid in id_map:
            raise ValueError(f"Duplicate raw_id detected: {rid}")

        id_map[rid] = rec.text

    return id_map


def flatten_list_of_lists(list_of_lists: Iterable[Sequence]) -> list:
    return list(chain(*list_of_lists))


def extract_and_normalize_slug(url: str) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if not path:
        return ""

    slug = path.split("/")[-1]
    slug = unquote(slug)

    slug = unicodedata.normalize("NFKD", slug)
    slug = slug.encode("ascii", "ignore").decode("ascii")
    slug = slug.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug)
    slug = slug.strip("-")

    return slug


def extract_domain_heuristic(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.hostname or ""
    host = host.lower()

    if host.startswith("www."):
        host = host[4:]

    parts = host.split(".")

    if len(parts) <= 2:
        return host

    last = parts[-1]
    second_last = parts[-2]

    if len(last) <= 3 and len(second_last) <= 3:
        return ".".join(parts[-3:])

    return ".".join(parts[-2:])


def rawrecords_grouped(raw_records: List[RawRecord], metadata_name='doc_id') -> Dict[str, Dict[int, str]]:
    """
    Group list[RawRecord] into:
        group_value -> { int(raw_id): text }
    """
    grouped: Dict[str, Dict[int, str]] = defaultdict(dict)

    for rec in raw_records:
        group_name = rec.metadata.get(metadata_name, "")

        try:
            rid = int(rec.raw_id)
        except ValueError:
            raise ValueError(f"raw_id '{rec.raw_id}' is not convertible to int")

        if rid in grouped[group_name]:
            raise ValueError(f"Duplicate raw_id {rid} found in group '{group_name}'")

        grouped[group_name][rid] = rec.text

    return grouped


def remove_by_raw_id(raw_records: List[RawRecord], ids_to_remove: List[str]) -> List[RawRecord]:
    ids_to_remove = set(ids_to_remove)
    return [r for r in raw_records if r.raw_id not in ids_to_remove]




# ---------------------------------------------------------
# Similarity function
# ---------------------------------------------------------
def rapidfuzz_similarity(a: str, b: str) -> float:
    return fuzz.ratio(a, b) / 100.0


# ---------------------------------------------------------
# Bucketing
# ---------------------------------------------------------
def bucketize(items: List[str]) -> Dict[Tuple[int, str], List[int]]:
    buckets = defaultdict(list)
    for idx, s in enumerate(items):
        prefix = s[:2].lower() if len(s) >= 2 else s.lower()
        length_bucket = len(s) // 5
        key = (length_bucket, prefix)
        buckets[key].append(idx)
    return buckets


# ---------------------------------------------------------
# Similarity graph
# ---------------------------------------------------------
def build_similarity_graph(
    items: List[str],
    similarity_fn: Callable[[str, str], float],
    threshold: float,
) -> Dict[int, Set[int]]:
    n = len(items)
    graph = {i: set() for i in range(n)}

    for i in range(n):
        for j in range(i + 1, n):
            sim = similarity_fn(items[i], items[j])
            if sim >= threshold:
                graph[i].add(j)
                graph[j].add(i)

    return graph


# ---------------------------------------------------------
# Connected components
# ---------------------------------------------------------
def connected_components(graph: Dict[int, Set[int]]) -> List[Set[int]]:
    visited = set()
    components = []

    for node in graph:
        if node in visited:
            continue

        stack = [node]
        comp = set()

        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            comp.add(cur)
            for nb in graph[cur]:
                if nb not in visited:
                    stack.append(nb)

        components.append(comp)

    return components


# ---------------------------------------------------------
# Unbatched clustering
# ---------------------------------------------------------
def cluster_strings_unbatched(
    items: List[str],
    threshold: float,
    similarity_fn: Callable[[str, str], float],
) -> List[List[int]]:
    graph = build_similarity_graph(items, similarity_fn, threshold)
    comps = connected_components(graph)
    return [sorted(list(comp)) for comp in comps]


# ---------------------------------------------------------
# Bucketed clustering
# ---------------------------------------------------------
def cluster_strings_bucketed(
    items: List[str],
    threshold: float,
    similarity_fn: Callable[[str, str], float],
) -> List[List[int]]:
    buckets = bucketize(items)
    all_clusters = []

    for _, idxs in buckets.items():
        if len(idxs) == 1:
            all_clusters.append([idxs[0]])
            continue

        bucket_strings = [items[i] for i in idxs]
        local_clusters = cluster_strings_unbatched(bucket_strings, threshold, similarity_fn)

        # Map local indices back to global indices
        for comp in local_clusters:
            all_clusters.append([idxs[i] for i in comp])

    return all_clusters


# ---------------------------------------------------------
# ⭐ Adaptive clustering (list or dict input)
# ---------------------------------------------------------
def cluster_strings_adaptive(
    data: Union[List[str], Dict[Any, str]],
    threshold: float = 0.8,
    similarity_fn: Callable[[str, str], float] = rapidfuzz_similarity,
    switch_at: int = 300,
) -> List[List[Any]]:
    """
    Accepts:
        - list of strings
        - dict of {id: string}

    Returns:
        - list of clusters
          * if input was a list → clusters of strings
          * if input was a dict → clusters of IDs (sorted ascending)
    """

    # -----------------------------------------------------
    # Normalize input
    # -----------------------------------------------------
    if isinstance(data, dict):
        ids = list(data.keys())
        items = list(data.values())

        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate IDs found in input dict")

        id_mode = True
    else:
        items = list(data)
        ids = list(range(len(items)))
        id_mode = False

    n = len(items)

    # -----------------------------------------------------
    # Choose strategy
    # -----------------------------------------------------
    if n <= switch_at:
        comps = cluster_strings_unbatched(items, threshold, similarity_fn)
    else:
        comps = cluster_strings_bucketed(items, threshold, similarity_fn)

    # -----------------------------------------------------
    # ⭐ Map back to IDs and sort ascending
    # -----------------------------------------------------
    if id_mode:
        return [sorted([ids[i] for i in comp]) for comp in comps]
    else:
        return [[items[i] for i in comp] for comp in comps]


def text_metadata_distances_below_threshold(
    records: List[RawRecord],
    embeddings,
    metadata_key: str,
    threshold: float = 0.15,
) -> Dict[str, float]:
    """
    Generalized version of slug-summary similarity:
    Computes cosine similarity between record.text and record.metadata[metadata_key].

    Returns a dict mapping raw_id -> similarity for all records
    whose similarity is below the threshold.
    """

    # -------------------------
    # Extract metadata text + summaries
    # -------------------------
    meta_texts = []
    summaries = []

    for r in records:
        meta_val = r.metadata.get(metadata_key, "")
        if not isinstance(meta_val, str):
            meta_val = str(meta_val)
        meta_texts.append(meta_val)
        summaries.append(r.text)

    # -------------------------
    # Batch embed metadata + summaries
    # -------------------------
    meta_vecs = embeddings.embed_texts(meta_texts)
    summary_vecs = embeddings.embed_texts(summaries)

    if len(meta_vecs) != len(summary_vecs):
        raise RuntimeError(
            "Embedding mismatch: meta_vecs and summary_vecs differ in length"
        )

    # -------------------------
    # Compute cosine similarities
    # -------------------------
    bad: Dict[str, float] = {}

    for i in range(len(meta_vecs)):
        v_meta = np.array(meta_vecs[i]).reshape(1, -1)
        v_sum = np.array(summary_vecs[i]).reshape(1, -1)
        sim = cosine_similarity(v_meta, v_sum)[0][0]

        if sim < threshold:
            bad[str(records[i].raw_id)] = float(sim)

    return bad
