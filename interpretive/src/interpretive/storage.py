from pathlib import Path
from typing import List, Dict, Any, Optional
from file_ops import FileOps

from .models import LayerNode, NodeMetadata

from retrieval.grading import BooleanGrader
from retrieval.retrieval import RetrievalLayer


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

from retrieval.grading import BooleanGrader
from retrieval.retrieval import RetrievalLayer
from llm import BaseLLMEngine, get_backend

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
        return validated

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
        layer = layer_name or self.layer_name
        relative = f"{layer}/nodes.json"
        absolute = self._abs(relative)

        if not absolute.is_file():
            return []

        nodes = self.fs.read_json(relative)

        # Validate after loading
        return self._validate_nodes(nodes)

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


    # ------------------------------------------------------------
    # RAG: Delete docs
    # ------------------------------------------------------------
    def delete_rag(self, doc_ids: List[str]) -> None:
        for doc_id in doc_ids:
            self.rag.delete_by_doc_id(doc_id)

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
