from pathlib import Path

# INTERNAL, not user-editable
_QDRANT_REGISTRY = {
    "test": {
        "chunks": "/Users/pense/projects/scotus/notebooks/ragstore_test",
    },
    "scotus":{
        "cases":"/Users/pense/projects/content/scotus/data/dbs/rag/cases/",
    },
    "news":{
        "layer_0":"/Users/pense/projects/data/news/interpretive_layers/0/",
        "triples":"/Users/pense/projects/data/news/interpretive_layers/triples/",
    }
}

def _resolve_path(project: str, collection: str) -> Path:
    try:
        path = _QDRANT_REGISTRY[project][collection]
    except KeyError:
        raise KeyError(f"Unknown project/collection: {project}/{collection}")

    p = Path(path).expanduser().resolve()
    p.mkdir(parents=True, exist_ok=True)
    return p
