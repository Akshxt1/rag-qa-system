"""
BM25 Index
Keyword-based search using BM25Okapi algorithm.
Complements vector search by catching exact keyword matches
that semantic search sometimes misses.
"""
from typing import List, Dict, Tuple, Optional
from loguru import logger


class BM25Index:
    """
    BM25 keyword search index.
    Must be rebuilt after adding/removing documents.
    """

    def __init__(self):
        self._chunks: List[Dict] = []
        self._bm25 = None
        self._is_built = False

    def build(self, chunks: List[Dict]) -> None:
        """Build BM25 index from a list of chunks."""
        from rank_bm25 import BM25Okapi

        if not chunks:
            self._is_built = False
            return

        self._chunks = chunks
        tokenized = [
            self._tokenize(c["text"]) for c in chunks
        ]
        self._bm25 = BM25Okapi(tokenized)
        self._is_built = True
        logger.info(f"BM25 index built with {len(chunks)} chunks")

    def search(
        self,
        query: str,
        top_k: int = 20,
        filter_sources: Optional[set] = None,
    ) -> List[Tuple[Dict, float]]:
        """
        Search using BM25 keyword matching.
        Returns list of (chunk, score) tuples.
        """
        if not self._is_built or not self._bm25:
            return []

        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Pair chunks with scores and sort
        scored = [
            (self._chunks[i], float(scores[i]))
            for i in range(len(self._chunks))
            if scores[i] > 0
        ]

        # Apply source filter if provided
        if filter_sources:
            scored = [
                (c, s) for c, s in scored
                if c.get("filename") in filter_sources
            ]

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def rebuild_from_store(self, vector_store) -> None:
        """Rebuild BM25 index from existing vector store chunks."""
        self.build(vector_store._chunks)

    @property
    def is_ready(self) -> bool:
        return self._is_built

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + lowercase tokenizer."""
        import re
        # Lowercase, remove punctuation, split
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return text.split()