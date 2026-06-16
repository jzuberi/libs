from typing import Any, Dict, List, TypedDict

# A vector is just a list of floats
Vector = List[float]

# Metadata is a flexible dictionary
Metadata = Dict[str, Any]


class QueryResult(TypedDict, total=False):
    ids: List[str]
    vectors: List[Vector]
    metadatas: List[Metadata]
    documents: List[str]
    distances: List[float]
