import os
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
import numpy as np
from loguru import logger


class FAISSVectorStore:
    INDEX_FILE = "index.faiss"
    CHUNKS_FILE = "chunks.pkl"

    def __init__(self, dimension: int, store_path: str = "./vector_store"):
        import faiss
        self.dimension = dimension
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self._index = faiss.IndexFlatIP(dimension)
        self._chunks: List[Dict] = []

    def add(self, chunks: List[Dict], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Mismatch: chunks vs embeddings count")
        self._index.add(embeddings.astype(np.float32))
        self._chunks.extend(chunks)
        logger.info(f"Added {len(chunks)} chunks. Total: {len(self._chunks)}")

    def clear(self) -> None:
        import faiss
        self._index = faiss.IndexFlatIP(self.dimension)
        self._chunks = []
        logger.info("Vector store cleared")

    def delete_by_source(self, filename: str) -> int:
        """Remove all chunks from a specific document. Rebuilds index."""
        import faiss
        keep_indices = [
            i for i, c in enumerate(self._chunks)
            if c.get("filename") != filename and c.get("source") != filename
        ]
        removed = len(self._chunks) - len(keep_indices)
        if removed == 0:
            return 0
        if keep_indices:
            vectors = np.vstack([
                self._index.reconstruct(i).reshape(1, -1)
                for i in keep_indices
            ]).astype(np.float32)
            new_chunks = [self._chunks[i] for i in keep_indices]
        else:
            vectors = None
            new_chunks = []
        self._index = faiss.IndexFlatIP(self.dimension)
        if vectors is not None:
            self._index.add(vectors)
        self._chunks = new_chunks
        logger.info(f"Deleted {removed} chunks from '{filename}'")
        return removed

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        score_threshold: float = 0.0,
        filter_sources: Optional[Set[str]] = None,
    ) -> List[Tuple[Dict, float]]:
        if self._index.ntotal == 0:
            return []
        k = self._index.ntotal if filter_sources else min(top_k, self._index.ntotal)
        query = query_vector.reshape(1, -1).astype(np.float32)
        scores, indices = self._index.search(query, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < score_threshold:
                continue
            chunk = self._chunks[idx]
            if filter_sources:
                if chunk.get("filename") not in filter_sources:
                    continue
            results.append((chunk, float(score)))
            if len(results) >= top_k:
                break
        return results

    def save(self) -> None:
        import faiss
        faiss.write_index(self._index, str(self.store_path / self.INDEX_FILE))
        with open(self.store_path / self.CHUNKS_FILE, "wb") as f:
            pickle.dump(self._chunks, f)
        logger.info(f"Saved {len(self._chunks)} chunks")

    def load(self) -> bool:
        import faiss
        index_path = self.store_path / self.INDEX_FILE
        chunks_path = self.store_path / self.CHUNKS_FILE
        if not index_path.exists() or not chunks_path.exists():
            return False
        try:
            self._index = faiss.read_index(str(index_path))
            with open(chunks_path, "rb") as f:
                self._chunks = pickle.load(f)
            logger.info(f"Loaded {len(self._chunks)} chunks")
            return True
        except Exception as e:
            logger.error(f"Failed to load: {e}")
            return False

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)

    @property
    def unique_sources(self) -> List[str]:
        seen = set()
        sources = []
        for c in self._chunks:
            fn = c.get("filename", c.get("source", "unknown"))
            if fn not in seen:
                seen.add(fn)
                sources.append(fn)
        return sources

    def stats(self) -> Dict:
        return {
            "total_chunks": self.total_chunks,
            "total_vectors": self._index.ntotal,
            "unique_documents": len(self.unique_sources),
            "sources": self.unique_sources,
            "dimension": self.dimension,
        }