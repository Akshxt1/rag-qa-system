"""
CrossEncoder Reranker
Re-scores retrieved chunks by directly comparing (query, chunk) pairs.
Much more accurate than vector similarity alone — catches subtle relevance
that embedding-based search misses.

Model options (speed vs quality):
  cross-encoder/ms-marco-MiniLM-L-6-v2   ← fast, good quality (default)
  cross-encoder/ms-marco-MiniLM-L-12-v2  ← slower, better quality
  cross-encoder/ms-marco-electra-base    ← best quality, slowest
"""
from typing import List, Tuple, Dict
from loguru import logger


class CrossEncoderReranker:
    """
    Reranks retrieved chunks using a CrossEncoder model.

    Usage:
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank(query, retrieved_chunks, top_k=5)
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        from sentence_transformers import CrossEncoder

        logger.info(f"Loading reranker model: {model_name}")
        self.model_name = model_name
        self._model = CrossEncoder(model_name)
        logger.info("Reranker ready")

    def rerank(
        self,
        query: str,
        chunks_with_scores: List[Tuple[Dict, float]],
        top_k: int = 5,
    ) -> List[Tuple[Dict, float]]:
        """
        Rerank chunks by direct (query, chunk_text) relevance scoring.

        Args:
            query:              The user's question
            chunks_with_scores: List of (chunk_dict, score) from retrieval
            top_k:              How many to return after reranking

        Returns:
            Top-k reranked list of (chunk_dict, reranker_score)
        """
        if not chunks_with_scores:
            return []

        # Build (query, passage) pairs for CrossEncoder
        pairs = [
            (query, chunk["text"])
            for chunk, _ in chunks_with_scores
        ]

        # Score all pairs at once (batched internally)
        scores = self._model.predict(pairs, show_progress_bar=False)

        # Zip back with original chunks and sort by new score
        reranked = sorted(
            zip(chunks_with_scores, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        # Return top_k with reranker scores (replacing original scores)
        results = []
        for (chunk, _original_score), reranker_score in reranked[:top_k]:
            results.append((chunk, float(reranker_score)))

        logger.debug(
            f"Reranked {len(chunks_with_scores)} → {len(results)} chunks"
        )
        return results

    def __repr__(self) -> str:
        return f"CrossEncoderReranker(model={self.model_name})"