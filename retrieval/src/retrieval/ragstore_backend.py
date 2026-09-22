import json
import random
from typing import Any, Optional


def convert_rules_to_chroma_where(rules) -> dict:
    """
    Convert subject/operator/object rules into a Chroma-compatible where filter.

    Accepts:
        - a single rule dict
        - a list of rule dicts
    """

    # Normalize input to a list
    if isinstance(rules, dict):
        rules = [rules]

    if not isinstance(rules, list):
        raise TypeError(
            f"Filter must be a rule dict or list of rule dicts, got {type(rules)}"
        )

    if len(rules) == 0:
        raise ValueError("Rule list cannot be empty")

    op_map = {
        "==": "$eq",
        "!=": "$ne",
        "<": "$lt",
        "<=": "$lte",
        ">": "$gt",
        ">=": "$gte",
        "in": "$in",
        "not in": "$nin",
    }

    # ------------------------------------------------------------
    # Helper: normalize datelike values → UNIX timestamp (seconds)
    # ------------------------------------------------------------
    from datetime import datetime, date

    def _normalize_datelike_value(v):
        """
        Normalize date-like values into UNIX timestamps (seconds).

        Supported:
            - UNIX timestamps (int) → unchanged
            - 'YYYY-MM-DD' → parsed → timestamp
            - 'YYYYMMDD' → parsed → timestamp
            - datetime.datetime → timestamp
            - datetime.date → timestamp
        """

        # Already a UNIX timestamp (rough heuristic)
        if isinstance(v, int) and v > 10**9:
            return v

        # YYYY-MM-DD
        if isinstance(v, str) and len(v) == 10 and v[4] == "-" and v[7] == "-":
            dt = datetime.strptime(v, "%Y-%m-%d")
            return int(dt.timestamp())

        # YYYYMMDD
        if isinstance(v, str) and len(v) == 8 and v.isdigit():
            dt = datetime.strptime(v, "%Y%m%d")
            return int(dt.timestamp())

        # datetime.datetime
        if isinstance(v, datetime):
            return int(v.timestamp())

        # datetime.date
        if isinstance(v, date):
            dt = datetime(v.year, v.month, v.day)
            return int(dt.timestamp())

        return v

    # ------------------------------------------------------------
    # Build Chroma clauses
    # ------------------------------------------------------------
    chroma_clauses = []

    for rule in rules:
        if not isinstance(rule, dict):
            raise TypeError(f"Each rule must be a dict, got {type(rule)}")

        missing = [key for key in ("subject", "operator", "object") if key not in rule]
        if missing:
            raise ValueError(
                f"Rule missing required fields: {missing}. Rule was: {rule}"
            )

        subject = rule["subject"]
        operator = rule["operator"]
        obj = _normalize_datelike_value(rule["object"])

        if operator not in op_map:
            raise ValueError(f"Unsupported operator: {operator}")

        chroma_operator = op_map[operator]
        chroma_clauses.append({subject: {chroma_operator: obj}})

    # ------------------------------------------------------------
    # Chroma rejects $and with only one clause → return single dict
    # ------------------------------------------------------------
    if len(chroma_clauses) == 1:
        return chroma_clauses[0]

    return {"$and": chroma_clauses}




class RAGStoreBackend:
    """
    Unified backend adapter for BOTH Qdrant and Chroma.
    Works with your RAGStore wrapper.
    """

    def __init__(self, rag_store):
        self.rag = rag_store
        self._cache = {}

        # actual vector backend (Chroma or Qdrant)
        backend = rag_store.backend
        name = backend.__class__.__name__.lower()
        module = backend.__class__.__module__.lower()

        # QDRANT DETECTION
        self.is_qdrant = (
            "qdrant" in name
            or "qdrant" in module
        )

        # CHROMA DETECTION
        self.is_chroma = (
            "chroma" in name
            or "chromadb" in module
        )

        # sanity: never both true
        if self.is_qdrant and self.is_chroma:
            # prefer explicit class name
            self.is_qdrant = "qdrant" in name
            self.is_chroma = "chroma" in name

        print("BACKEND CLASS:", backend.__class__)
        print("is_chroma:", self.is_chroma)
        print("is_qdrant:", self.is_qdrant)

    def _normalize_filter(self, filter_dict):
        if not filter_dict:
            return None

        # ============================================================
        # QDRANT FILTER NORMALIZATION
        # ============================================================
        if self.is_qdrant:
            must_clauses = []

            # Flatten $and if present
            if "$and" in filter_dict:
                clauses = filter_dict["$and"]
            else:
                clauses = [filter_dict]

            for clause in clauses:
                for key, val in clause.items():
                    if isinstance(val, list):
                        must_clauses.append({
                            "key": key,
                            "match": {"any": val}
                        })
                    else:
                        must_clauses.append({
                            "key": key,
                            "match": {"value": val}
                        })

            return {"must": must_clauses}

        # ============================================================
        # CHROMA FILTER NORMALIZATION
        # ============================================================


        # ============================================================
        # RULE FORMAT → CHROMA FILTER (only if Chroma + list)
        # ============================================================
        if self.is_chroma and isinstance(filter_dict, list):
            filter_dict = convert_rules_to_chroma_where(filter_dict)

        # If Chroma: return filter_dict exactly as-is.
        # (Chroma already supports $and, $in, $lte, etc.)
        if self.is_chroma:
            return filter_dict

        # ============================================================
        # DEFAULT: return unchanged
        # ============================================================
        return filter_dict


    def _normalize_results(self, res):
        # QDRANT → return raw dict
        if self.is_qdrant:
            return res

        # CHROMA → res is a LIST of dicts
        if isinstance(res, list):
            out = []
            for item in res:
                out.append({
                    "id": item.get("id"),
                    "text": item.get("document") or item.get("text"),
                    "metadata": item.get("metadata", {}),
                    "distance": item.get("distance"),
                })
            return out

        # fallback: unexpected format
        raise RuntimeError(f"Unexpected Chroma result format: {type(res)}")


    def query(
        self,
        text_query: str,
        offset: int = 0,
        limit: int = 10,
        filter: Optional[Any] = None,
        diversify: bool = False,
        expansion_factor: int = 3,
        minimum_pool: int = 20,
    ):
        filter_norm = self._normalize_filter(filter)

        print('ragstore backend filter:')
        print(filter_norm)

        key = (
            text_query,
            json.dumps(filter_norm, sort_keys=True) if filter_norm else None,
            diversify,
        )

        if key is key:

            expanded_k = max(limit * expansion_factor, minimum_pool)

            if self.is_qdrant:

                full_results = self.rag.query(
                    text_query,
                    k=expanded_k,
                    filter=filter_norm,
                    diversify=diversify,
                )

            elif self.is_chroma:

                query_vec = self.rag.embeddings.embed_query(text_query)
                
                res = self.rag.backend.query(
                    query_vec,
                    k=expanded_k,
                    filter=filter_norm
                    )

                ids = res['ids']

                res_reformat = self.rag.backend.get_by_ids(ids)

                full_results = self._normalize_results(res_reformat)

            else:
                raise RuntimeError("Unknown backend type: cannot query")

            if diversify:
                random.shuffle(full_results)

            self._cache[key] = full_results

        docs = self._cache[key]
        return docs[offset:offset + limit]


    def get_ids(self, rules=None, text_query=None, k=100):
        """
        Return IDs matching metadata rules, optionally filtered by a text query.
        Uses the same logic as `query()` for Chroma, but returns only IDs.
        """

        where_filter = self._normalize_filter(rules)

        # ---------------------------------------------------------
        # CASE 1: metadata-only filtering
        # ---------------------------------------------------------
        if text_query is None:

            if self.is_chroma:
                res = self.rag.backend.collection.get(
                    where=where_filter,
                    include=["metadatas"],
                )
                return res.get("ids", [])

            elif self.is_qdrant:
                scroll_res = self.rag.backend.client.scroll(
                    collection_name=self.rag.backend.collection_name,
                    scroll_filter=where_filter,
                    limit=k,
                    with_payload=False,
                )
                return [p.id for p in scroll_res[0]]

            else:
                raise RuntimeError("Unknown backend type")

        # ---------------------------------------------------------
        # CASE 2: hybrid filtering (text query + metadata filter)
        # ---------------------------------------------------------
        else:

            # ---------- CHROMA HYBRID PATH ----------
            if self.is_chroma:

                # 1) embed query
                query_vec = self.rag.embeddings.embed_query(text_query)

                # 2) call Chroma exactly like query()
                res = self.rag.backend.query(
                    query_vec,
                    k,
                    where_filter
                )

                # 3) extract IDs (Chroma returns nested lists)
                ids = res.get("ids", [])
                if isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
                    ids = ids[0]

                return ids

            # ---------- QDRANT HYBRID PATH ----------
            elif self.is_qdrant:

                full_results = self.rag.query(
                    text_query,
                    k=k,
                    filter=where_filter,
                    diversify=False,
                )
                return [r["id"] for r in full_results]

            else:
                raise RuntimeError("Unknown backend type")



    def get_sample_ids(
        self,
        rules,
        sample_size=50,
        seed=1234,
    ):
        """
        Deterministically sample IDs matching metadata rules.
        """
        ids = self.get_ids(rules)

        random.seed(seed)
        if len(ids) <= sample_size:
            return ids

        return random.sample(ids, sample_size)
