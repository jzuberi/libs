from typing import List, Dict, Any, Callable, Set, Tuple, Union, Optional
from collections import defaultdict
from itertools import chain
import random

import numpy as np
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_distances
import unicodedata
import re

def normalized_levenshtein(a: str, b: str) -> float:
    
    # -----------------------------
    # Normalization pipeline
    # -----------------------------
    def normalize_for_diff(s: str) -> str:
        # Lowercase
        s = s.lower()
        # Unicode normalize (compatibility decomposition)
        s = unicodedata.normalize("NFKD", s)
        # Strip accents
        s = s.encode("ascii", "ignore").decode("ascii")
        # Remove symbols/punctuation
        s = re.sub(r"[^a-z0-9]+", "", s)
        return s
    
    # -----------------------------
    # Levenshtein distance
    # -----------------------------
    def levenshtein(a: str, b: str) -> int:
        # Ensure a is the longer string
        if len(a) < len(b):
            a, b = b, a
    
        previous = list(range(len(b) + 1))
        for i, ca in enumerate(a, start=1):
            current = [i]
            for j, cb in enumerate(b, start=1):
                insert = current[j-1] + 1
                delete = previous[j] + 1
                replace = previous[j-1] + (ca != cb)
                current.append(min(insert, delete, replace))
            previous = current
    
        return previous[-1]
    na, nb = normalize_for_diff(a), normalize_for_diff(b)
    d = levenshtein(na, nb)
    max_len = max(len(na), len(nb))
    return 1 - d / max_len if max_len else 1

def get_closest(
    query: str,
    candidates: List[str],
    query_vector: Optional[np.ndarray] = None,
    candidate_vectors: Optional[List[np.ndarray]] = None,
    embed_fn=None,
    min_words: int = 1,
    max_words: Optional[int] = None,
    min_chars: int = 1,
    max_chars: Optional[int] = None,
):
    """
    Return candidates reordered by semantic closeness to `query`.
    Optional: provide precomputed embeddings for query and candidates.
    Filtering: min/max words, min/max chars.
    """

    # -----------------------------------------------------
    # Helper: ensure vectors are 1-D numpy arrays
    # -----------------------------------------------------
    def to_vec(v):
        v = np.asarray(v)
        if v.ndim == 1:
            return v
        return v.reshape(-1)  # flatten any weird shape

    # -----------------------------------------------------
    # 1. FILTER CANDIDATES
    # -----------------------------------------------------
    filtered = []
    filtered_vectors = []

    candidates = list(set(candidates))

    for idx, s in enumerate(candidates):
        wcount = len(s.split())
        if wcount < min_words:
            continue
        if max_words is not None and wcount > max_words:
            continue

        if len(s) < min_chars:
            continue
        if max_chars is not None and len(s) > max_chars:
            continue

        filtered.append(s)

        if candidate_vectors is not None:
            filtered_vectors.append(to_vec(candidate_vectors[idx]))

    if not filtered:
        return []

    # -----------------------------------------------------
    # 2. EMBEDDINGS
    # -----------------------------------------------------
    # Query embedding
    if query_vector is None:
        if embed_fn is None:
            raise ValueError("embed_fn must be provided if query_vector is None")
        query_vector = to_vec(embed_fn([query])[0])
    else:
        query_vector = to_vec(query_vector)

    # Candidate embeddings
    if candidate_vectors is None:
        if embed_fn is None:
            raise ValueError("embed_fn must be provided if candidate_vectors is None")
        filtered_vectors = [to_vec(v) for v in embed_fn(filtered)]

    # -----------------------------------------------------
    # 3. DISTANCES
    # -----------------------------------------------------
    qvec = query_vector.reshape(1, -1)
    cvecs = np.stack(filtered_vectors, axis=0)
    dists = cosine_distances(qvec, cvecs)[0]

    # -----------------------------------------------------
    # 4. SORT BY DISTANCE
    # -----------------------------------------------------
    sorted_pairs = sorted(zip(filtered, dists), key=lambda x: x[1])
    sorted_strings = [s for s, _ in sorted_pairs]

    return sorted_strings


class SemanticSubgraph:
    
    def __init__(
        self,
        embed_fn,
        llm_fn=None,
        llm_call=None,
        sample_size=5000,
        fuzz_threshold=0.92,
        embed_threshold=0.80,
        switch_at=300,
        relative_threshold=0.05,
        min_steps=5,
        min_sweep_nodes=20,
        min_chars=20,
        max_chars=300,
        verbose=False,
        random_state=None,
    ):
        self.embed_fn = embed_fn
        self.llm_fn = llm_fn
        self.llm_call = llm_call
        self.sample_size = sample_size
        self.fuzz_threshold = fuzz_threshold
        self.embed_threshold = embed_threshold
        self.switch_at = switch_at
        self.relative_threshold = relative_threshold
        self.min_steps = min_steps
        self.min_sweep_nodes = min_sweep_nodes
        self.verbose = verbose
        self._rng = random.Random(random_state)
        self.min_chars = min_chars
        self.max_chars = max_chars



    # -----------------------------------------------------
    # 1. TEXTUAL EQUIVALENCE (RapidFuzz + clustering)
    # -----------------------------------------------------
    @staticmethod
    def _rapidfuzz_similarity(a: str, b: str) -> float:
        return fuzz.ratio(a, b) / 100.0

    @staticmethod
    def _bucketize(items: List[str]) -> Dict[Tuple[int, str], List[int]]:
        buckets: Dict[Tuple[int, str], List[int]] = defaultdict(list)
        for idx, s in enumerate(items):
            prefix = s[:2].lower() if len(s) >= 2 else s.lower()
            length_bucket = len(s) // 5
            key = (length_bucket, prefix)
            buckets[key].append(idx)
        return buckets

    @staticmethod
    def _build_similarity_graph(
        items: List[str],
        similarity_fn: Callable[[str, str], float],
        threshold: float,
    ) -> Dict[int, Set[int]]:
        n = len(items)
        graph: Dict[int, Set[int]] = {i: set() for i in range(n)}

        for i in range(n):
            for j in range(i + 1, n):
                sim = similarity_fn(items[i], items[j])
                if sim >= threshold:
                    graph[i].add(j)
                    graph[j].add(i)

        return graph

    @staticmethod
    def _connected_components(graph: Dict[int, Set[int]]) -> List[Set[int]]:
        visited: Set[int] = set()
        components: List[Set[int]] = []

        for node in graph:
            if node in visited:
                continue

            stack = [node]
            comp: Set[int] = set()

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

    def _cluster_strings_unbatched(
        self,
        items: List[str],
        threshold: float,
        similarity_fn: Callable[[str, str], float],
    ) -> List[List[int]]:
        graph = self._build_similarity_graph(items, similarity_fn, threshold)
        comps = self._connected_components(graph)
        return [sorted(list(comp)) for comp in comps]

    def _cluster_strings_bucketed(
        self,
        items: List[str],
        threshold: float,
        similarity_fn: Callable[[str, str], float],
    ) -> List[List[int]]:
        buckets = self._bucketize(items)
        all_clusters: List[List[int]] = []

        for _, idxs in buckets.items():
            if len(idxs) == 1:
                all_clusters.append([idxs[0]])
                continue

            bucket_strings = [items[i] for i in idxs]
            local_clusters = self._cluster_strings_unbatched(
                bucket_strings, threshold, similarity_fn
            )

            for comp in local_clusters:
                all_clusters.append([idxs[i] for i in comp])

        return all_clusters

    def _cluster_strings_adaptive(
        self,
        data: Union[List[str], Dict[Any, str]],
        threshold: float,
        similarity_fn: Callable[[str, str], float],
    ) -> List[List[Any]]:
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

        if n <= self.switch_at:
            comps = self._cluster_strings_unbatched(items, threshold, similarity_fn)
        else:
            comps = self._cluster_strings_bucketed(items, threshold, similarity_fn)

        if id_mode:
            return [sorted([ids[i] for i in comp]) for comp in comps]
        else:
            return [[items[i] for i in comp] for comp in comps]
        
    def _semantic_equivalence_collapse(
        self,
        strings: List[str],
        vectors: Optional[List[np.ndarray]] = None,   # NEW: optional
        min_chars: int = 20,
        min_words: int = 3,
        min_window: int = 5,
        equiv_threshold: float = 0.70,
    ):
        # -----------------------------------------------------
        # 0. FILTER STRINGS (+ FILTER VECTORS IF PROVIDED)
        # -----------------------------------------------------
        filtered = []
        filtered_vectors = []
        orig_map = {}  # filtered_idx -> original_idx
    
        for idx, s in enumerate(strings):
            if len(s) < min_chars:
                continue
            if len(s.split()) < min_words:
                continue
    
            fidx = len(filtered)
            orig_map[fidx] = idx
            filtered.append(s)
    
            if vectors is not None:
                filtered_vectors.append(vectors[idx])
    
        if not filtered:
            return {
                "anchors": [],
                "eq_map": {},
                "vectors": [],
                "dmatrix": [],
            }
    
        # -----------------------------------------------------
        # 1. EMBEDDINGS (USE PROVIDED OR COMPUTE)
        # -----------------------------------------------------
        if vectors is None:
            # original behavior
            vectors_all = self.embed_fn(filtered)
        else:
            # filtered vectors already aligned
            vectors_all = filtered_vectors
    
        # -----------------------------------------------------
        # 2. DISTANCE MATRIX
        # -----------------------------------------------------
        dmatrix_all = cosine_distances(np.stack(vectors_all, axis=0))
        n = len(filtered)
    
        # -----------------------------------------------------
        # 3. ZERO-CLOSENESS AND POSITIVE PAIRS
        # -----------------------------------------------------
        zero_pairs = []
        pos_pairs = []
    
        for i in range(n):
            for j in range(i + 1, n):
                closeness = 1.0 - dmatrix_all[i, j]
                if closeness <= 0.0:
                    zero_pairs.append((i, j))
                else:
                    pos_pairs.append((i, j, closeness))
    
        pos_pairs.sort(key=lambda x: x[2], reverse=True)
    
        # -----------------------------------------------------
        # 4. INITIAL CLUSTERS
        # -----------------------------------------------------
        clusters = []
        index_to_cluster = {}
    
        def create_singleton(i):
            if i not in index_to_cluster:
                cid = len(clusters)
                clusters.append({"members": {i}})
                index_to_cluster[i] = cid
    
        for (i, j) in zero_pairs:
            create_singleton(i)
            create_singleton(j)
    
        # -----------------------------------------------------
        # 5. SWEEP THROUGH POSITIVE PAIRS
        # -----------------------------------------------------
        equiv_history = []
    
        for (i, j, closeness) in pos_pairs:
            if len(equiv_history) >= min_window:
                window = equiv_history[-min_window:]
                avg_equiv = sum(window) / min_window
                if avg_equiv < equiv_threshold:
                    break
    
            is_equiv = self.llm_fn(filtered[i], filtered[j], self.llm_call)
            equiv_history.append(1 if is_equiv else 0)
    
            if not is_equiv:
                continue
    
            ci = index_to_cluster.get(i)
            cj = index_to_cluster.get(j)
    
            if ci is None and cj is None:
                cid = len(clusters)
                clusters.append({"members": {i, j}})
                index_to_cluster[i] = cid
                index_to_cluster[j] = cid
    
            elif ci is not None and cj is None:
                clusters[ci]["members"].add(j)
                index_to_cluster[j] = ci
    
            elif ci is None and cj is not None:
                clusters[cj]["members"].add(i)
                index_to_cluster[i] = cj
    
            elif ci != cj:
                keep = min(ci, cj)
                drop = max(ci, cj)
                for idx in clusters[drop]["members"]:
                    clusters[keep]["members"].add(idx)
                    index_to_cluster[idx] = keep
                clusters[drop]["members"].clear()
    
        final_clusters = [c for c in clusters if c["members"]]
    
        # -----------------------------------------------------
        # 6. ANCHORS + EQ_MAP (filtered → original)
        # -----------------------------------------------------
        anchors = []
        eq_map = {}
        used = set()
    
        for c in final_clusters:
            members = sorted(c["members"])
            anchor = max(members, key=lambda i: len(filtered[i]))
            anchors.append(anchor)
            used.update(members)
            orig_members = [orig_map[m] for m in members]
            eq_map[anchor] = orig_members
    
        for i in range(n):
            if i not in used:
                anchors.append(i)
                eq_map[i] = [orig_map[i]]
    
        anchors = sorted(set(anchors))
    
        return {
            "anchors": anchors,
            "eq_map": eq_map,
            "vectors": vectors_all,
            "dmatrix": dmatrix_all,
        }

    
    
    # -----------------------------------------------------
    # 4. EMBEDDING + DISTANCE MATRIX
    # -----------------------------------------------------
    def _embed_representatives(self, reps: List[str]) -> List[np.ndarray]:
        return self.embed_fn(reps)

    @staticmethod
    def _distance_matrix(vectors: List[np.ndarray]) -> np.ndarray:
        return cosine_distances(np.stack(vectors, axis=0))


    # -----------------------------------------------------
    # 5. THRESHOLD SWEEP / non_singleton_nodes stopping criteria
    # -----------------------------------------------------
    def _sweep_thresholds(
        self,
        dmatrix: np.ndarray,
        ids: List[int],
    ) -> Tuple[List[List[int]], float, List[Dict[str, Any]]]:


        n = dmatrix.shape[0]

        # -----------------------------------------------------
        # 1. Minimum node guardrail
        # -----------------------------------------------------
        if n < self.min_sweep_nodes:
            clusters = [[i] for i in ids]
            return clusters, 0.0, []

        # -----------------------------------------------------
        # 2. Extract meaningful thresholds (unique distances)
        # -----------------------------------------------------
        upper = dmatrix[np.triu_indices(n, k=1)]
        unique_dists = sorted(set(float(d) for d in upper))

        if len(unique_dists) == 0:
            clusters = [[i] for i in ids]
            return clusters, 0.0, []

        # -----------------------------------------------------
        # 3. Sweep over actual transition points
        # -----------------------------------------------------
        history: List[Dict[str, Any]] = []
        stable_steps = 0
        chosen_thr = unique_dists[-1]
        chosen_clusters: List[List[int]] = []

        prev_ns_nodes: Optional[int] = None

        for thr in unique_dists:

            # Build graph at this threshold
            graph: Dict[int, Set[int]] = {i: set() for i in range(n)}
            for i in range(n):
                for j in range(i + 1, n):
                    if dmatrix[i, j] <= thr:
                        graph[i].add(j)
                        graph[j].add(i)

            # Connected components
            comps = self._connected_components(graph)
            clusters = [sorted([ids[i] for i in comp]) for comp in comps]

            # Count non-singleton nodes
            ns_nodes = sum(len(comp) for comp in clusters if len(comp) > 1)

            # Record history
            history.append({
                "threshold": thr,
                "cluster_count": len(clusters),
                "non_singleton_nodes": ns_nodes,
            })

            # -----------------------------------------------------
            # Stability detection based on non-singleton node growth
            # -----------------------------------------------------
            if prev_ns_nodes is not None:
                delta_ns = ns_nodes - prev_ns_nodes

                # If non-singleton nodes stop increasing → stability
                if delta_ns <= 0:
                    stable_steps += 1
                else:
                    stable_steps = 0

                if stable_steps >= self.min_steps:
                    chosen_thr = thr
                    chosen_clusters = clusters
                    break

            prev_ns_nodes = ns_nodes
            chosen_clusters = clusters

        return chosen_clusters, chosen_thr, history


    # -----------------------------------------------------
    # 6. EXPAND SUBGRAPHS
    # -----------------------------------------------------
    @staticmethod
    def _expand_clusters(
        clusters: List[List[int]],
        eq_map: Dict[int, List[int]],
    ) -> List[List[int]]:
        expanded: List[List[int]] = []
        for comp in clusters:
            full: List[int] = []
            for rep_idx in comp:
                full.extend(eq_map.get(rep_idx, []))
            expanded.append(sorted(full))
        return expanded


    # -----------------------------------------------------
    # 7. SAMPLING
    # -----------------------------------------------------
    def _sample_subsets(self, strings: List[str]) -> List[Tuple[List[int], List[str]]]:
        n = len(strings)
        if n <= self.sample_size:
            idxs = list(range(n))
            return [(idxs, strings)]

        idxs = list(range(n))
        self._rng.shuffle(idxs)

        subsets: List[Tuple[List[int], List[str]]] = []
        for start in range(0, n, self.sample_size):
            chunk = idxs[start : start + self.sample_size]
            subset = [strings[i] for i in chunk]
            subsets.append((chunk, subset))

        return subsets


    def _merge_equivalence(
        self,
        global_reps: List[str],
        global_eq_map: Dict[int, List[int]],
        new_reps: List[str],
        new_eq_map: Dict[int, List[int]],
    ) -> None:
        offset = len(global_reps)
        global_reps.extend(new_reps)
        for k, v in new_eq_map.items():
            global_eq_map[offset + k] = v
    
    def _check_vector_shapes(self, vectors: List[np.ndarray]) -> None:
        """
        Warn if embeddings are not 1-D numeric arrays.
        Does not raise unless shapes are completely unusable.
        """
        if vectors is None:
            return
    
        bad_shapes = []
        for i, v in enumerate(vectors):
            if v is None:
                bad_shapes.append((i, "None"))
                continue
    
            arr = np.asarray(v)
    
            if arr.ndim != 1:
                bad_shapes.append((i, f"{arr.shape} (ndim={arr.ndim})"))
                continue
    
            if not np.issubdtype(arr.dtype, np.number):
                bad_shapes.append((i, f"non-numeric dtype {arr.dtype}"))
    
        if bad_shapes:
            print("\n[SemanticSubgraph WARNING] Some embeddings have unexpected shapes:")
            for idx, shape in bad_shapes[:10]:
                print(f"  - vector[{idx}] has shape/type: {shape}")
            if len(bad_shapes) > 10:
                print(f"  ... and {len(bad_shapes) - 10} more.")
            print("Expected: each embedding must be a 1-D numeric array.\n")

            
            
    def run(self, strings: List[str], vectors: Optional[List[np.ndarray]] = None) -> Dict[str, Any]:
        original_strings = strings
    
        # -----------------------------------------------------
        # 0. SHAPE CHECK (optional embeddings)
        # -----------------------------------------------------
        if vectors is not None:
            self._check_vector_shapes(vectors)

    
        # -----------------------------------------------------
        # Helper: filter strings and build filtered→original map
        # -----------------------------------------------------
        def filter_strings(strings, vectors):
            filtered = []
            filtered_to_orig = {}
            filtered_vectors = [] if vectors is not None else None
    
            for orig_idx, s in enumerate(strings):
                if self.min_chars <= len(s) <= self.max_chars:
                    fidx = len(filtered)
                    filtered.append(s)
                    filtered_to_orig[fidx] = orig_idx
    
                    if vectors is not None:
                        filtered_vectors.append(vectors[orig_idx])
    
            return filtered, filtered_to_orig, filtered_vectors
    
        # -----------------------------------------------------
        # Helper: remap eq_map (filtered indices → original indices)
        # -----------------------------------------------------
        def remap_eq_map(eq_map_filtered, filtered_to_orig):
            eq_map_original = {}
            for a_f, members_f in eq_map_filtered.items():
                eq_map_original[a_f] = [filtered_to_orig[m] for m in members_f]
            return eq_map_original
    
        # -----------------------------------------------------
        # Helper: expand sweep clusters using eq_map_original
        # -----------------------------------------------------
        def expand_clusters(clusters_semantic, anchors_filtered, eq_map_original):
            final_clusters = []
            for cluster in clusters_semantic:
                expanded = []
                for sweep_idx in cluster:
                    a_f = anchors_filtered[sweep_idx]   # filtered anchor idx
                    expanded.extend(eq_map_original[a_f])
                final_clusters.append(sorted(expanded))
            return final_clusters
    
        # -----------------------------------------------------
        # 0. FILTER
        # -----------------------------------------------------
        filtered, filtered_to_orig, filtered_vectors = filter_strings(original_strings, vectors)
    
        if not filtered:
            return {
                "clusters": [],
                "representatives": [],
                "equivalence_map": {},
                "threshold": 0.0,
                "history": [],
            }
    
        # -----------------------------------------------------
        # 1. SEMANTIC EQUIVALENCE COLLAPSE (FILTERED SPACE)
        # -----------------------------------------------------
        stage1 = self._semantic_equivalence_collapse(
            filtered,
            vectors=filtered_vectors  # may be None → collapse computes embeddings
        )
    
        anchors_filtered = stage1["anchors"]          # filtered indices
        eq_map_filtered = stage1["eq_map"]            # filtered_idx -> filtered_idx list
        vectors_all = stage1["vectors"]
        dmatrix_all = stage1["dmatrix"]
    
        # -----------------------------------------------------
        # 2. REMAP eq_map TO ORIGINAL INDEX SPACE
        # -----------------------------------------------------
        eq_map_original = remap_eq_map(eq_map_filtered, filtered_to_orig)
    
        # -----------------------------------------------------
        # 3. REPRESENTATIVES (ORIGINAL STRINGS)
        # -----------------------------------------------------
        representatives = [
            original_strings[filtered_to_orig[a_f]]
            for a_f in anchors_filtered
        ]
    
        # -----------------------------------------------------
        # 4. SWEEP (FILTERED SPACE)
        # -----------------------------------------------------
        semantic_ids = list(range(len(anchors_filtered)))
        dmatrix_sweep = dmatrix_all[np.ix_(anchors_filtered, anchors_filtered)]
    
        clusters_semantic, thr, history = self._sweep_thresholds(
            dmatrix_sweep,
            semantic_ids,
        )
    
        # -----------------------------------------------------
        # 5. EXPAND CLUSTERS BACK TO ORIGINAL INDICES
        # -----------------------------------------------------
        final_clusters = expand_clusters(
            clusters_semantic,
            anchors_filtered,
            eq_map_original
        )
    
        return {
            "clusters": final_clusters,              # ORIGINAL indices
            "representatives": representatives,      # ORIGINAL strings
            "equivalence_map": eq_map_original,      # filtered anchor -> ORIGINAL idx list
            "threshold": thr,
            "history": history,
        }


