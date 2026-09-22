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
        "layer_0_lance":"/Users/pense/projects/data/news/interpretive_layers/0_lance/",
        "layer_0_chroma":"/Users/pense/projects/data/news/interpretive_layers/0_chroma/",
        "triples":"/Users/pense/projects/data/news/interpretive_layers/triples/",
    },
    "earnings":{
        "transcripts":"/Users/pense/projects/content/earnings_calls/data/rag/transcripts/",
        "hooks":"/Users/pense/projects/content/earnings_calls/data/rag/hooks/"
    },
    "congress":{
        "house_transcripts":"/Users/pense/projects/content/congress/data/rag/house_transcripts/",
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
