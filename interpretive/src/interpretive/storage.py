from pathlib import Path
from typing import List, Dict, Any, Optional
from file_ops import FileOps

from .models import LayerNode, NodeMetadata

from retrieval.grading import BooleanGrader
from retrieval.retrieval import RetrievalLayer
from datetime import datetime

from llm import BaseLLMEngine, get_backend


from ragstore import (
    generate_chunks_from_documents
)

from ragstore import filter_and_fetch_chunks

class RetrievalBuilder:
    @staticmethod
    def build(rag, llm_call):
        grader = BooleanGrader(llm_call)
        return RetrievalLayer(
            rag_store=rag,
            grader=grader,
            llm_call=llm_call,
        )

class LayerStorage:

    def __init__(
        self,
        root: str | Path,
        rag,
        llm_backend_name: str,
        llm_timeout: int = 30,
    ):
        self.root = Path(root).expanduser().resolve()
        self.fs = FileOps(root=self.root)
        self.rag = rag

        backend_path = Path(self.rag.backend.path)
        self.layer_name = backend_path.name

        # ------------------------------------------------------------
        # Build LLM + llm_call
        # ------------------------------------------------------------
        backend = get_backend(llm_backend_name, timeout=llm_timeout)
        self.llm = BaseLLMEngine(backend, timeout=llm_timeout)
        self.llm_call = lambda prompt: self.llm._call_backend(prompt)

        # ------------------------------------------------------------
        # Build retrieval layer bound to this RAG
        # ------------------------------------------------------------
        grader = BooleanGrader(self.llm_call)
        self.retrieval = RetrievalLayer(
            rag_store=self.rag,
            grader=grader,
            llm_call=self.llm_call,
        )



    # ------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------
    def _abs(self, relative_path: str | Path) -> Path:
        return (self.root / relative_path).resolve()
    


    # ------------------------------------------------------------
    # Metadata validation helper
    # ------------------------------------------------------------
    def _validate_nodes(self, nodes: List[LayerNode]) -> List[LayerNode]:
        """
        validated = []
        for n in nodes:
            # Validate metadata using NodeMetadata
            # IMPORTANT: NodeMetadata.created is now a STRING, not datetime
            meta = NodeMetadata(**n["metadata"]).model_dump()

            validated.append({
                "id": n["id"],
                "text": n["text"],
                "metadata": meta,
            })
        """
        return nodes

    # ------------------------------------------------------------
    # JSON: Save (upsert) nodes
    # ------------------------------------------------------------
    def save_layer(self, nodes: List[LayerNode], layer_name: Optional[str] = None) -> None:
        layer = layer_name or self.layer_name
        path = f"{layer}/nodes.json"

        # Validate before saving
        nodes = self._validate_nodes(nodes)

        self.fs.ensure_dir(layer)
        self.fs.upsert_json_records(path, new_records=nodes, key="id", default=[])

    # ------------------------------------------------------------
    # JSON: Load nodes
    # ------------------------------------------------------------
    def load_layer(self, layer_name: Optional[str] = None) -> List[LayerNode]:
        """
        Load all nodes from the backend (Chroma) instead of only from nodes.json.
        This becomes the canonical view of the layer.
        """
        # Ignore layer_name for now; your backend is already scoped by collection.
        backend = self.rag.backend

        nodes: List[LayerNode] = []

        # Use stream_export to iterate over all records
        for ids, embeddings, metadatas, documents in backend.stream_export(batch_size=1000):
            for mid, meta, doc in zip(ids, metadatas, documents):
                nodes.append({
                    "id": mid,
                    "text": doc,
                    "metadata": meta,
                })

        # Validate after loading
        return nodes


    # ------------------------------------------------------------
    # JSON: Upsert subset
    # ------------------------------------------------------------
    def upsert_layer(self, nodes: List[LayerNode], layer_name: Optional[str] = None) -> None:
        layer = layer_name or self.layer_name
        path = f"{layer}/nodes.json"

        # Validate before saving
        nodes = self._validate_nodes(nodes)

        self.fs.ensure_dir(layer)
        self.fs.upsert_json_records(path, new_records=nodes, key="id", default=[])

    # ------------------------------------------------------------
    # RAG: Upsert docs
    # ------------------------------------------------------------
    def upsert_rag(self, docs: List[List[Any]]) -> None:
        """
        docs: List of [text, metadata] pairs.
        Skips documents whose doc_id already exists in the RAG.
        """

        # Extract doc_ids from incoming docs
        incoming_ids = []
        for _, meta in docs:
            doc_id = meta.get("doc_id")
            if doc_id is None:
                raise ValueError("Missing doc_id in metadata")
            incoming_ids.append(doc_id)

        # Batch existence check (fast)
        existing = self.rag.backend.existing_doc_ids(incoming_ids)

        # Filter out docs that already exist
        filtered = []
        for (text, meta) in docs:
            doc_id = meta["doc_id"]
            if doc_id not in existing:
                filtered.append([text, meta])
            

        if not filtered:
            print("[DONE] No new documents to ingest")
            return

        # Only chunk + embed + upsert new docs
        chunks = generate_chunks_from_documents(filtered)
        self.rag.add_raw_chunks(chunks)
        self.rag.hybrid.flush()

    # ------------------------------------------------------------
    # RAG: Delete docs
    # ------------------------------------------------------------

    def delete_rag(self, ids: List[str]) -> None:
        """
        Delete chunk IDs directly from the backend.
        """
        if not ids:
            return

        # ChromaBackend.delete supports ids directly
        self.rag.backend.delete(ids=ids)

    # ------------------------------------------------------------
    # RAG: Slice
    # ------------------------------------------------------------
    def slice_rag(self, filter=None, limit=10000):
        norm = self.rag._normalize_filter(filter)
        raw, sampled, seed = filter_and_fetch_chunks(
            self.rag.backend,
            norm,
            limit,
        )

        chunk_like = []
        for rec in raw:
            chunk_like.append({
                "id": rec["id"],
                "text": rec["metadata"].get("text", ""),
                "metadata": rec.get("metadata", {}),
            })

        # Validate slice output too
        return self._validate_nodes(chunk_like)

    def get_nodes_by_filter(self, where: Dict[str, Any]) -> List[LayerNode]:
        nodes = self.load_layer()
        out = []

        for n in nodes:
            meta = n["metadata"]
            match = True
            for k, v in where.items():
                if meta.get(k) != v:
                    match = False
                    break
            if match:
                out.append(n)

        return out

    def delete_nodes_by_ids(self, ids: List[str]) -> None:
        # Delete from RAG backend
        self.delete_rag(ids)

        # Delete from JSON layer
        nodes = self.load_layer()
        remaining = [n for n in nodes if n["id"] not in ids]
        self.save_layer(remaining)

    def _apply_rule(self, nodes, rule: dict) -> List[str]:
        """
        Apply a single subject–operator–object rule to all nodes.

        A rule has the form:
            {
                "subject": <metadata key>,
                "operator": <comparison operator>,
                "object": <value to compare against>
            }

        Supported operators depend on the metadata type:

        STRING OPERATORS:
            "=="       — exact match
            "!="       — not equal
            "in"       — value is in a list of strings
            "not in"   — value is not in a list of strings

            Examples:
                {"subject": "ticker", "operator": "==", "object": "SMCI"}
                {"subject": "speaker", "operator": "!=", "object": "CEO"}
                {"subject": "ticker", "operator": "in", "object": ["SMCI", "NVDA"]}
                {"subject": "doc_id", "operator": "not in", "object": ["SMCI20264-17"]}

        NUMERIC OPERATORS:
            "=="       — equal
            "!="       — not equal
            "<"        — less than
            "<="       — less than or equal
            ">"        — greater than
            ">="       — greater than or equal
            "in"       — value is in a list of numbers
            "not in"   — value is not in a list of numbers

            Numeric rule objects may be:
                - integers (2026)
                - floats
                - numeric strings ("2026", "1786400000")
                - date strings ("2026-01-01") which are automatically
                converted to UNIX timestamps

            Examples:
                {"subject": "year", "operator": "==", "object": 2026}
                {"subject": "quarter", "operator": "<", "object": 3}
                {"subject": "date", "operator": ">=", "object": "2026-01-01"}
                {"subject": "chunk_index", "operator": "in", "object": [0, 1, 2]}

        BOOLEAN OPERATORS:
            "=="       — equal
            "!="       — not equal

            Examples:
                {"subject": "is_summary", "operator": "==", "object": True}
                {"subject": "is_question", "operator": "!=", "object": False}

        Returns:
            A list of chunk IDs whose metadata satisfies the rule.
        """

        subject = rule["subject"]
        operator = rule["operator"]
        obj = rule["object"]

        matched = []

        for n in nodes:
            meta = n["metadata"]

            if subject not in meta:
                continue

            value = meta[subject]

            # ---- STRING OPERATORS ----
            if isinstance(value, str):
                if operator == "==":
                    ok = value == obj
                elif operator == "!=":
                    ok = value != obj
                elif operator == "in":
                    ok = value in obj
                elif operator == "not in":
                    ok = value not in obj
                else:
                    ok = False

            # ---- NUMERIC OPERATORS ----
            elif isinstance(value, (int, float)):
                if operator in ("in", "not in"):
                    normalized_list = [self._normalize_numeric_object(x) for x in obj]
                    ok = (value in normalized_list) if operator == "in" else (value not in normalized_list)
                else:
                    try:
                        target = self._normalize_numeric_object(obj)
                    except Exception:
                        ok = False
                    else:
                        if operator == "==":
                            ok = value == target
                        elif operator == "!=":
                            ok = value != target
                        elif operator == "<":
                            ok = value < target
                        elif operator == "<=":
                            ok = value <= target
                        elif operator == ">":
                            ok = value > target
                        elif operator == ">=":
                            ok = value >= target
                        else:
                            ok = False

            # ---- BOOLEAN OPERATORS ----
            elif isinstance(value, bool):
                if operator == "==":
                    ok = value is obj
                elif operator == "!=":
                    ok = value is not obj
                else:
                    ok = False

            else:
                ok = False

            if ok:
                matched.append(n["id"])

        return matched


    def _normalize_numeric_object(self, obj):
        """
        Convert numeric rule objects into ints.
        Supports:
        - int
        - float
        - numeric strings ("2026", "1786400000")
        - date strings ("2026-01-01")
        """
        # Already numeric
        if isinstance(obj, (int, float)):
            return int(obj)

        # Numeric string
        if isinstance(obj, str) and obj.isdigit():
            return int(obj)

        # Date string: YYYY-MM-DD
        if isinstance(obj, str):
            try:
                dt = datetime.strptime(obj, "%Y-%m-%d")
                return int(dt.timestamp())
            except ValueError:
                pass  # Not a date string

        # Fallback: let caller handle failure
        return obj

    def evict(self, rule: dict) -> List[str]:
        """
        Apply a single eviction rule and delete all matching nodes.

        The rule must follow the subject–operator–object format used by _apply_rule.

        Example:
            storage.evict({
                "subject": "ticker",
                "operator": "==",
                "object": "SMCI"
            })

            storage.evict({
                "subject": "date",
                "operator": "<",
                "object": "2026-01-01"
            })

        Returns:
            A list of chunk IDs that were evicted.
        """

        # 1. Load all nodes from backend
        nodes = self.load_layer()

        # 2. Apply rule to determine which IDs to evict
        ids_to_evict = self._apply_rule(nodes, rule)

        if not ids_to_evict:
            return []

        # 3. Delete from backend + JSON layer
        self.delete_nodes_by_ids(ids_to_evict)

        return ids_to_evict
