# ragstore/chunking/pipeline.py

from ragstore.chunking.preprocess import preprocess
from ragstore.chunking.strategies import UniversalChunker

def chunk_text(text: str, source_id: str, chunker=None):
    chunker = chunker or UniversalChunker()
    cleaned = preprocess(text)
    raw_chunks = chunker.chunk(cleaned)

    for i, chunk in enumerate(raw_chunks):
        yield {
            "id": f"{source_id}-{i}",
            "text": chunk,
            "metadata": {
                "source_id": source_id,
                "chunk_index": i,
                "strategy": chunker.__class__.__name__,
            }
        }


def generate_chunks_from_documents(docs):
    for text, metadata in docs:
        source_id = metadata["doc_id"]
        for chunk in chunk_text(text, source_id=source_id):
            # merge metadata into chunk metadata
            chunk["metadata"].update(metadata)
            yield chunk