from typing import Dict, Any, List
from pydantic import BaseModel


class RawRecord(BaseModel):
    """
    A single raw input record before preprocessing.
    """
    raw_id: str
    text: str
    metadata: Dict[str, Any] = {}



class SemanticNode(BaseModel):
    """
    A canonical semantic unit produced by preprocessing.
    This becomes a single RAGStore document.
    """
    semantic_node_id: str
    text: str
    metadata: Dict[str, Any]


class SemanticNodeRAG(BaseModel):
    """
    A container mapping semantic_node_id → SemanticNode.
    Passed into SemanticSubgraph alongside the RAGStore.
    """
    nodes: Dict[str, SemanticNode]

    # -------------------------
    # Basic list-like behavior
    # -------------------------
    def __iter__(self):
        return iter(self.nodes.values())

    def __len__(self):
        return len(self.nodes)

    def __getitem__(self, key):
        """
        Supports:
            rag["id"]   → dict-style lookup
            rag[0]      → index lookup
            rag[0:5]    → slice
        """
        if isinstance(key, int):
            return list(self.nodes.values())[key]
        if isinstance(key, slice):
            return list(self.nodes.values())[key]
        return self.nodes[key]

    # -------------------------
    # Inspection helpers
    # -------------------------
    def head(self, n: int = 5):
        """Return the first n SemanticNodes."""
        return list(self.nodes.values())[:n]

    def tail(self, n: int = 5):
        """Return the last n SemanticNodes."""
        return list(self.nodes.values())[-n:]

    def sample(self, n: int = 5):
        """Return n random SemanticNodes."""
        import random
        vals = list(self.nodes.values())
        return random.sample(vals, min(n, len(vals)))

    def ids(self):
        """Return a list of semantic_node_ids."""
        return list(self.nodes.keys())

    def items(self):
        """Return (id, node) pairs like a dict."""
        return self.nodes.items()

    def filter(self, fn):
        """
        Return a new SemanticNodeRAG containing only nodes
        where fn(node) is True.
        """
        filtered = {k: v for k, v in self.nodes.items() if fn(v)}
        return SemanticNodeRAG(nodes=filtered)

    def find(self, substring: str):
        """
        Return nodes whose text contains the substring (case-insensitive).
        """
        substring = substring.lower()
        matches = {
            k: v for k, v in self.nodes.items()
            if substring in v.text.lower()
        }
        return SemanticNodeRAG(nodes=matches)

    def summary(self, n: int = 5):
        """
        Pretty-print a quick summary of the first few nodes.
        """
        preview = self.head(n)
        lines = [f"SemanticNodeRAG: {len(self)} nodes"]
        for node in preview:
            lines.append(f"- {node.semantic_node_id}: {node.text[:60]!r}...")
        return "\n".join(lines)


class PreprocessActions(BaseModel):
    """
    A standardized container describing side-effects that should be applied
    to the source-of-truth database AFTER a preprocessing step completes.

    Preprocessing steps remain pure and return only decisions.
    The application layer is responsible for executing these actions.
    """

    # Raw records that should be marked inactive / soft-deleted in the DB
    deactivate_raw_ids: List[str] = []

    # Arbitrary metadata updates to apply to specific raw records
    # Example:
    #   update_metadata = {
    #       "raw123": {"quality_score": 0.82},
    #       "raw456": {"source": "dedupe_removed"}
    #   }
    update_metadata: Dict[str, Dict[str, Any]] = {}

    # Optional: records that should be merged or linked
    # (Useful for equivalence clustering provenance)
    link_raw_ids: Dict[str, List[str]] = {}

    # Optional: any warnings or notes for logging/auditing
    notes: List[str] = []
