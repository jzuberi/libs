from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class Chunk:
    id: str
    text: str
    metadata: Dict[str, Any]


class ChunkNormalizationError(Exception):
    pass


def normalize_chunk(obj: Any) -> Chunk:
    """
    Convert arbitrary objects into a canonical Chunk.
    Supported shapes:
      - Already a Chunk
      - LangChain-like: obj.page_content + obj.metadata
      - Dicts: {"id": ..., "text": ..., "metadata": ...}
      - Minimal dicts: {"id": ..., "text": ...}
    """

    # 1. Already normalized
    if isinstance(obj, Chunk):
        return obj

    # 2. Dict-like
    if isinstance(obj, dict):
        if "id" in obj and "text" in obj:
            return Chunk(
                id=str(obj["id"]),
                text=str(obj["text"]),
                metadata=obj.get("metadata", {}),
            )
        raise ChunkNormalizationError(
            f"Dict missing required keys: {obj}"
        )

    # 3. LangChain Document-like
    if hasattr(obj, "page_content") and hasattr(obj, "metadata"):
        meta = getattr(obj, "metadata")
        if "id" not in meta:
            raise ChunkNormalizationError(
                f"metadata must contain 'id': {meta}"
            )
        return Chunk(
            id=str(meta["id"]),
            text=str(getattr(obj, "page_content")),
            metadata=meta,
        )

    # 4. Unsupported type
    raise ChunkNormalizationError(
        f"Cannot normalize object of type {type(obj)}"
    )
