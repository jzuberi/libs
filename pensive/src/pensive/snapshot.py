# src/pensive/snapshot.py

from dataclasses import dataclass
from typing import Any, List


@dataclass(frozen=True)
class Snapshot:
    """Immutable view of corpus + decisions at a point in time."""
    corpus: List[Any]
    decisions: Any
    metadata: dict

    def __iter__(self):
        return iter(self.corpus)



class SnapshotIngestor:
    """Loads corpus + decisions into a deterministic Snapshot."""

    def __init__(self, corpus_loader, corpus_model, decision_store):
        self.corpus_loader = corpus_loader
        self.corpus_model = corpus_model
        self.decision_store = decision_store

    def ingest(self) -> Snapshot:
        # 1. Load raw corpus
        raw_corpus = self.corpus_loader()

        # 2. Optionally validate/structure with corpus_model
        if self.corpus_model:
            corpus = [self.corpus_model(**item) for item in raw_corpus]
        else:
            corpus = raw_corpus

        # 3. Load decisions
        decisions = self.decision_store.load()

        # 4. Metadata (expand later)
        metadata = {
            "corpus_count": len(corpus),
            "decision_count": len(decisions),
        }

        """
        print("INGESTOR USING:", self.corpus_model)
        print("RAW CORPUS:", raw_corpus)
        print("FIRST ITEM TYPE:", type(raw_corpus[0]))
        print("MODEL INIT TEST:", self.corpus_model(**raw_corpus[0]))
        """


        return Snapshot(corpus=corpus, decisions=decisions, metadata=metadata)
