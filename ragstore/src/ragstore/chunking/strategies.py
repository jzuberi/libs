# ragstore/chunking/strategies.py
import re

class ChunkingStrategy:
    def chunk(self, text: str) -> list[str]:
        raise NotImplementedError


class UniversalChunker(ChunkingStrategy):
    def __init__(self, max_chars=800):
        self.max_chars = max_chars

    def split_sentences(self, text: str):
        return re.split(r'(?<=[.!?]) +', text)

    def chunk(self, text: str):
        sentences = self.split_sentences(text)
        chunks, current = [], ""

        for s in sentences:
            if len(current) + len(s) > self.max_chars:
                chunks.append(current.strip())
                current = s
            else:
                current += " " + s

        if current:
            chunks.append(current.strip())

        return chunks
