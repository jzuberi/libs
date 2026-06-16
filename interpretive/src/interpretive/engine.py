
from ragstore import (
    RAGStore,
    QdrantBackend,
    LMStudioEmbeddingService,
    generate_chunks_from_documents
)

from llm import BaseLLMEngine, get_backend
from retrieval.grading import BooleanGrader
from retrieval.retrieval import RetrievalLayer

from pydantic import BaseModel
from llm import BaseLLMEngine, get_backend

from typing import Dict, Any

from .storage import LayerStorage

class CustomLLM(BaseLLMEngine):

    def __init__(self, backend, timeout, debug=True):
        # Call BaseLLMEngine initializer
        super().__init__(backend, timeout=timeout, debug=debug)

class InterpretiveEngine:
    """
    Triple-layer InterpretiveEngine.

    Responsibilities:
    - Build LLM + llm_call
    - Build retrieval layer (bound to ephemeral RAG)
    - Allow dynamic slicing to populate ephemeral RAG
    """

    def __init__(
        self,
        source_storage: LayerStorage,
        target_storage: LayerStorage,
        rag_embedding_model: str = "nomic-embed-text-v1.5",
        rag_embedding_url: str = "http://localhost:1234/v1/embeddings",
        llm_backend_name: str = 'qwen3.5-9b-claude-4.6-highiq-instruct-heretic-uncensored',
        llm_timeout: int = 30,
    ):
        self.source_storage = source_storage
        self.target_storage = target_storage

        self.rag_embedding_model = rag_embedding_model
        self.rag_embedding_url = rag_embedding_url

        # LLM
        self.custom_llm = self._build_llm(llm_backend_name, llm_timeout)
        self.llm_call = lambda prompt: self.custom_llm._call_backend(prompt)

        # Ephemeral RAG starts empty
        self.rag = None

    # ============================================================
    # HELPERS
    # ============================================================

    def _build_llm(self, backend_name: str, timeout: int):
        llm_backend = get_backend(backend_name, timeout=timeout)
        return CustomLLM(llm_backend, timeout=timeout)

    # ============================================================
    # DYNAMIC SLICE LOADING
    # ============================================================

    def load_slice(self, slice_filter: Dict[str, Any], slice_limit: int = 500):
        """
        Build an ephemeral RAG from a slice of the persistent RAG.
        Can be called multiple times to refresh the working set.
        """

        # 1. Slice the persistent RAG from the source layer
        slice_chunks = self.source_storage.slice_rag(
            filter=slice_filter,
            limit=slice_limit,
        )

        # 2. Build ephemeral backend + embeddings
        backend = QdrantBackend()
        embeddings = LMStudioEmbeddingService(
            model=self.rag_embedding_model,
            url=self.rag_embedding_url,
        )

        # 3. Populate ephemeral RAG
        temp_rag = RAGStore(backend, embeddings)
        temp_rag.add_raw_chunks(slice_chunks)

        # 4. Attach to engine
        self.rag = temp_rag
        

        grader = BooleanGrader(self.llm_call)
        self.retrieval = RetrievalLayer(
            rag_store=self.rag,
            grader=grader,
            llm_call=self.llm_call,
        )
