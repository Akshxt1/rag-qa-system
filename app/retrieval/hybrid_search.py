"""
Hybrid Search
Combines FAISS vector search + BM25 keyword search using
Reciprocal Rank Fusion (RRF) for better retrieval than either alone.

RRF Formula: score(d) = Σ 1 / (k + rank(d))
  - k=60 is the standard smoothing constant
  - Higher score = more consistently ranked across both methods
"""
from typing import List, Tuple, Dict, Optional, Set
from loguru import logger


def reciprocal_rank_fusion(
    results_list: List[List[Tuple[Dict, float]]],
    k: int = 60,
) -> List[Tuple[Dict, float]]:
    """
    Merge multiple ranked result lists using RRF.

    Args:
        results_list: List of retrieval results, each being [(chunk, score)]
        k:            RRF smoothing constant (default 60)

    Returns:
        Merged and re-ranked list of (chunk, rrf_score)
    """
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict] = {}

    for results in results_list:
        for rank, (chunk, _score) in enumerate(results):
            # Unique key per chunk
            key = f"{chunk.get('filename', '')}__chunk_{chunk.get('chunk_idx', 0)}"

            if key not in rrf_scores:
                rrf_scores[key] = 0.0
                chunk_map[key] = chunk

            # RRF score accumulation
            rrf_scores[key] += 1.0 / (k + rank + 1)

    # Sort by RRF score descending
    sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    fused = [(chunk_map[k], rrf_scores[k]) for k in sorted_keys]

    logger.debug(
        f"RRF fusion: {[len(r) for r in results_list]} results → {len(fused)} unique chunks"
    )
    return fused


class HybridSearcher:
    """
    Orchestrates hybrid FAISS + BM25 search with RRF fusion.

    Retrieval flow:
      1. FAISS vector search  → top candidate_k chunks
      2. BM25 keyword search  → top candidate_k chunks
      3. RRF fusion           → merged ranking
      4. Return top final_k   → passed to reranker or generator
    """

    def __init__(self, vector_store, bm25_index):
        self.vector_store = vector_store
        self.bm25_index = bm25_index

    def search(
        self,
        query_vector,
        query_text: str,
        top_k: int = 5,
        candidate_k: int = 20,
        score_threshold: float = 0.0,
        filter_sources: Optional[Set[str]] = None,
        use_bm25: bool = True,
    ) -> List[Tuple[Dict, float]]:
        """
        Run hybrid search and return fused top-k results.

        Args:
            query_vector:    Embedded query from HuggingFace embedder
            query_text:      Raw query string for BM25
            top_k:           Final results to return
            candidate_k:     Candidates per method before fusion
            score_threshold: Min FAISS score filter
            filter_sources:  Restrict to specific filenames
            use_bm25:        Toggle BM25 (False = pure vector search)

        Returns:
            List of (chunk, score) tuples
        """
        # ── FAISS vector search ───────────────────────────────────────────────
        vector_results = self.vector_store.search(
            query_vector,
            top_k=candidate_k,
            score_threshold=score_threshold,
            filter_sources=filter_sources,
        )
        logger.debug(f"FAISS retrieved {len(vector_results)} candidates")

        # ── BM25 keyword search ───────────────────────────────────────────────
        if use_bm25 and self.bm25_index.is_ready:
            bm25_results = self.bm25_index.search(
                query_text,
                top_k=candidate_k,
                filter_sources=filter_sources,
            )
            logger.debug(f"BM25 retrieved {len(bm25_results)} candidates")
            results_to_fuse = [vector_results, bm25_results]
        else:
            results_to_fuse = [vector_results]

        # ── RRF Fusion ────────────────────────────────────────────────────────
        if len(results_to_fuse) > 1:
            fused = reciprocal_rank_fusion(results_to_fuse)
        else:
            fused = vector_results

        return fused[:top_k]