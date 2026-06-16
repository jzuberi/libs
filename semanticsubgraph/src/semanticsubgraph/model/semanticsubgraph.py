from typing import Any, Dict, Optional, List

from semanticsubgraph.graph.extractor import get_subgraph_from_embeddings
from semanticsubgraph.graph.peeling import remove_clustered_embeddings


class SemanticSubgraph:
    """
    SemanticSubgraph (v0.3)

    This class manages:
      • structural subgraph extraction (clusters of story IDs)
      • semantic context for the dataset (data_context)
      • semantic labels for each subgraph (subgraph_labels)

    ───────────────────────────────────────────────────────────────
    DATA CONTEXT SCHEMA (REQUIRED)
    -------------------------------
    data_context: Dict[str, Any]
        {
            "scope": str,          # semantic floor for labeling
            "description": str,    # description of the canonical story universe
            "metadata": {          # optional domain hints
                "domain": "news",
                "granularity": "story",
                ...
            }
        }

    ───────────────────────────────────────────────────────────────
    LABEL OUTPUT SCHEMA (OPTIONAL, LOADED OR GENERATED)
    ---------------------------------------------------
    subgraph_labels: Dict[str, Dict[str, Any]]

        {
            "label": str,                 # short human-readable name
            "description": str,           # 1–3 sentence explanation
            "keywords": List[str],        # optional LLM-generated keywords
            "supporting_ids": List[str],  # story IDs defining the cluster
            "metadata": {                 # optional structured signals
                "is_loc": bool,
                "is_event": bool,
                "geo": [...],
                "time_range": [...],
                ...
            },
            "context": {                  # hierarchical refinement context
                "parent_scope": str,      # inherited from data_context["scope"]
                "refined_scope": str,     # new scope for this subgraph
            }
        }

    ───────────────────────────────────────────────────────────────
    STRUCTURAL VS SEMANTIC LAYERS
    ------------------------------
    • graph            → structural (NetworkX or similar)
    • subgraphs        → structural clusters (list of dicts)
    • subgraph_labels  → semantic annotations (parallel to subgraphs)
    • data_context     → semantic floor for labeling
    """

    def __init__(
        self,
        rag: Any,
        llm_call: Any,
        data_context: Dict[str, Any],     # REQUIRED semantic universe description
        graph: Optional[Any] = None,      # structural graph (optional)
        subgraph_labels: Optional[Dict[str, Dict[str, Any]]] = None,  # semantic labels (optional)
        config: Optional[Dict[str, Any]] = None,
    ):
        # Required components
        self.rag = rag
        self.llm_call = llm_call
        self.data_context = data_context  # structured semantic context

        # Optional components
        self.config = config or {}
        self.graph = graph                # structural graph (if loading an old one)

        # Structural layer: list of extracted subgraphs (clusters)
        self.subgraphs: List[Dict[str, Any]] = []

        # Semantic layer: labels for each subgraph (parallel to subgraphs)
        self.subgraph_labels = subgraph_labels or {}

        # Embedding state for iterative peeling
        self.remaining_embeddings: Optional[Dict[str, Any]] = None

        # Validate inputs
        self._validate_inputs()

    # ───────────────────────────────────────────────────────────────
    # VALIDATION LAYER
    # ───────────────────────────────────────────────────────────────

    def _validate_inputs(self):
        """Validate data_context, graph, labels, and their alignment."""
        self._validate_data_context()

        if self.graph is not None:
            self._validate_graph()

        if self.subgraph_labels:
            self._validate_label_schema()

        if self.graph is not None and self.subgraph_labels:
            self._validate_graph_label_alignment()

    def _validate_data_context(self):
        """Ensure data_context contains required semantic fields."""
        required = ["scope", "description"]
        for key in required:
            if key not in self.data_context:
                raise ValueError(f"data_context missing required field: {key}")

    def _validate_graph(self):
        """Minimal structural validation for graph objects."""
        if not hasattr(self.graph, "nodes"):
            raise TypeError("graph must be a NetworkX-like object with .nodes()")

    def _validate_label_schema(self):
        """Ensure each label follows the agreed-upon schema."""
        required_fields = ["label", "description", "supporting_ids", "context"]

        for subgraph_id, label in self.subgraph_labels.items():
            for field in required_fields:
                if field not in label:
                    raise ValueError(
                        f"Label for subgraph {subgraph_id} missing required field: {field}"
                    )

            ctx = label["context"]
            if "parent_scope" not in ctx or "refined_scope" not in ctx:
                raise ValueError(
                    f"Label for subgraph {subgraph_id} has invalid context structure"
                )

    def _validate_graph_label_alignment(self):
        """
        Ensure:
          • every supporting_id exists in the graph
          • parent_scope matches data_context["scope"]
        """
        for subgraph_id, label in self.subgraph_labels.items():
            if label["context"]["parent_scope"] != self.data_context["scope"]:
                raise ValueError(
                    f"Label for subgraph {subgraph_id} has mismatched parent_scope"
                )

            for story_id in label["supporting_ids"]:
                if story_id not in self.graph.nodes:
                    raise ValueError(
                        f"Label for subgraph {subgraph_id} references unknown story_id: {story_id}"
                    )

    # ───────────────────────────────────────────────────────────────
    # SUBGRAPH EXTRACTION METHODS
    # ───────────────────────────────────────────────────────────────

    def build_from_embeddings(
        self,
        embeddings_dict: Dict[str, Any],
        verbose: bool = False,
        **kwargs,
    ):
        """
        First pass: build the initial subgraph from embeddings and
        compute the remaining embeddings for further peeling.

        Returns:
            (subgraph, remaining_embeddings)
        """
        sub = get_subgraph_from_embeddings(
            embeddings_dict,
            verbose=verbose,
            **kwargs,
        )
        rem = remove_clustered_embeddings(embeddings_dict, sub)

        self.subgraphs.append(sub)
        self.remaining_embeddings = rem

        return sub, rem

    def peel_next(self, verbose: bool = False, **kwargs):
        """
        Peel the next subgraph from remaining embeddings, if any.

        Returns:
            (subgraph, remaining_embeddings) or (None, None)
        """
        if not self.remaining_embeddings or not self.remaining_embeddings.get("embeddings"):
            return None, None

        sub = get_subgraph_from_embeddings(
            self.remaining_embeddings,
            verbose=verbose,
            **kwargs,
        )
        rem = remove_clustered_embeddings(self.remaining_embeddings, sub)

        self.subgraphs.append(sub)
        self.remaining_embeddings = rem

        return sub, rem

    # ───────────────────────────────────────────────────────────────
    # LABELING STUB (to be implemented later)
    # ───────────────────────────────────────────────────────────────

    def label_subgraph(self, subgraph_id: str):
        """
        Placeholder for future labeling logic.

        Expected behavior (not implemented yet):
          • take a structural subgraph
          • generate a semantic label object matching the schema
          • store it in self.subgraph_labels[subgraph_id]

        This method intentionally does nothing for now.
        """
        pass
