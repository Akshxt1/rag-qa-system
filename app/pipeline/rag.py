import time
from typing import List, Dict, Optional, Set
from loguru import logger

import config
from app.ingestion.loader import DocumentLoader
from app.ingestion.chunker import TextChunker
from app.embeddings.embedder import Embedder
from app.vectorstore.faiss_store import FAISSVectorStore
from app.generation.gemini import GeminiGenerator
from app.retrieval.bm25_index import BM25Index
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.hybrid_search import HybridSearcher


class IngestResult:
    def __init__(self, success, documents_loaded=0, chunks_created=0,
                 elapsed_seconds=0.0, errors=None):
        self.success = success
        self.documents_loaded = documents_loaded
        self.chunks_created = chunks_created
        self.elapsed_seconds = round(elapsed_seconds, 2)
        self.errors = errors or []

    def to_dict(self):
        return self.__dict__


class QueryResult:
    def __init__(self, question, answer, sources, elapsed_seconds,
                 chunks_retrieved, retrieval_method="vector"):
        self.question = question
        self.answer = answer
        self.sources = sources
        self.elapsed_seconds = round(elapsed_seconds, 2)
        self.chunks_retrieved = chunks_retrieved
        self.retrieval_method = retrieval_method

    def to_dict(self):
        return self.__dict__


class RAGPipeline:
    def __init__(self):
        logger.info("Initializing RAG Pipeline...")
        config.validate()

        self.loader = DocumentLoader()
        self.chunker = TextChunker(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )
        self.embedder = Embedder(model_name=config.HF_MODEL_NAME)
        self.vector_store = FAISSVectorStore(
            dimension=self.embedder.dimension,
            store_path=config.VECTOR_STORE_PATH,
        )
        self.generator = GeminiGenerator(
            api_key=config.GEMINI_API_KEY,
            model_name=config.GEMINI_MODEL,
        )

        # ── Retrieval improvements ────────────────────────────────────────────
        self.bm25_index = BM25Index()
        self.reranker = CrossEncoderReranker(
            model_name=config.RERANKER_MODEL
        ) if config.ENABLE_RERANKING else None
        self.hybrid_searcher = HybridSearcher(self.vector_store, self.bm25_index)

        # Load persisted vector store + rebuild BM25
        self._loaded_from_disk = self.vector_store.load()
        if self._loaded_from_disk:
            self.bm25_index.rebuild_from_store(self.vector_store)
            logger.info("BM25 index rebuilt from disk")

        logger.info(f"Pipeline ready | chunks={self.vector_store.total_chunks} "
                    f"| reranking={config.ENABLE_RERANKING} "
                    f"| hybrid={config.ENABLE_HYBRID_SEARCH}")

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest_files(self, file_paths: List[str]) -> IngestResult:
        t0 = time.time()
        errors, docs = [], []
        for path in file_paths:
            try:
                docs.append(self.loader.load_file(path))
            except Exception as e:
                errors.append(f"{path}: {e}")
        if not docs:
            return IngestResult(success=False,
                                errors=errors or ["No valid documents loaded"])
        chunks = self.chunker.chunk_documents(docs)
        if not chunks:
            return IngestResult(success=False, documents_loaded=len(docs),
                                errors=["No text extracted"])
        embeddings = self.embedder.embed_batch([c["text"] for c in chunks])
        self.vector_store.add(chunks, embeddings)
        self.vector_store.save()
        self.bm25_index.rebuild_from_store(self.vector_store)
        return IngestResult(
            success=True,
            documents_loaded=len(docs),
            chunks_created=len(chunks),
            elapsed_seconds=time.time() - t0,
            errors=errors,
        )

    def ingest_from_bytes(self, file_bytes: bytes, filename: str) -> IngestResult:
        t0 = time.time()
        try:
            doc = self.loader.load_uploaded_file(file_bytes, filename)
        except Exception as e:
            return IngestResult(success=False, errors=[str(e)])
        chunks = self.chunker.chunk_document(doc)
        if not chunks:
            return IngestResult(success=False, documents_loaded=1,
                                errors=["No text extracted"])
        embeddings = self.embedder.embed_batch(
            [c["text"] for c in chunks], show_progress=False
        )
        self.vector_store.add(chunks, embeddings)
        self.vector_store.save()
        self.bm25_index.rebuild_from_store(self.vector_store)
        return IngestResult(
            success=True,
            documents_loaded=1,
            chunks_created=len(chunks),
            elapsed_seconds=time.time() - t0,
        )

    # ── Deletion ──────────────────────────────────────────────────────────────

    def delete_document(self, filename: str) -> int:
        removed = self.vector_store.delete_by_source(filename)
        if removed > 0:
            self.vector_store.save()
            self.bm25_index.rebuild_from_store(self.vector_store)
            logger.info(f"Deleted '{filename}' ({removed} chunks)")
        return removed

    # ── Querying ──────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        filter_sources: Optional[Set[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        score_threshold: Optional[float] = None,
        use_hybrid: Optional[bool] = None,
        use_reranking: Optional[bool] = None,
    ) -> QueryResult:
        if not self.is_ready:
            return QueryResult(
                question=question,
                answer="⚠️ No documents ingested yet. Please upload documents first.",
                sources=[], elapsed_seconds=0.0, chunks_retrieved=0,
            )

        t0 = time.time()
        k = top_k or config.TOP_K
        candidate_k = config.CANDIDATE_K

        # Determine which methods to use
        hybrid = use_hybrid if use_hybrid is not None else config.ENABLE_HYBRID_SEARCH
        reranking = use_reranking if use_reranking is not None else config.ENABLE_RERANKING

        # ── Step 1: Embed query ───────────────────────────────────────────────
        query_vec = self.embedder.embed_query(question)

        # ── Step 2: Hybrid retrieval (FAISS + BM25 + RRF) ────────────────────
        retrieved = self.hybrid_searcher.search(
            query_vector=query_vec,
            query_text=question,
            top_k=candidate_k if reranking else k,
            candidate_k=candidate_k,
            score_threshold=score_threshold or 0.0,
            filter_sources=filter_sources if filter_sources else None,
            use_bm25=hybrid,
        )

        retrieval_method = "hybrid" if hybrid else "vector"

        # ── Step 3: CrossEncoder reranking ────────────────────────────────────
        if reranking and self.reranker and retrieved:
            retrieved = self.reranker.rerank(question, retrieved, top_k=k)
            retrieval_method += "+reranked"

        # ── Step 4: Generate answer ───────────────────────────────────────────
        answer = self.generator.generate(
            question, retrieved,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        sources = []
        for chunk, score in retrieved:
            sources.append({
                "filename": chunk.get("filename", "unknown"),
                "source": chunk.get("source", "unknown"),
                "relevance_score": round(score, 4),
                "chunk_idx": chunk.get("chunk_idx", 0),
                "excerpt": chunk["text"][:300] + (
                    "…" if len(chunk["text"]) > 300 else ""
                ),
                "doc_type": chunk.get("doc_type", "unknown"),
            })

        return QueryResult(
            question=question,
            answer=answer,
            sources=sources,
            elapsed_seconds=time.time() - t0,
            chunks_retrieved=len(retrieved),
            retrieval_method=retrieval_method,
        )

    # ── Model switching ───────────────────────────────────────────────────────

    def set_model(self, model_name: str):
        self.generator.set_model(model_name)

    # ── Control ───────────────────────────────────────────────────────────────

    def reset(self):
        self.vector_store.clear()
        self.bm25_index.build([])

    @property
    def is_ready(self) -> bool:
        return self.vector_store.total_chunks > 0

    def status(self) -> Dict:
        return {
            "ready": self.is_ready,
            "embedding_model": config.HF_MODEL_NAME,
            "embedding_dim": self.embedder.dimension,
            "generative_model": self.generator.model_name,
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "top_k": config.TOP_K,
            "candidate_k": config.CANDIDATE_K,
            "hybrid_search": config.ENABLE_HYBRID_SEARCH,
            "reranking": config.ENABLE_RERANKING,
            "reranker_model": config.RERANKER_MODEL,
            "loaded_from_disk": self._loaded_from_disk,
            **self.vector_store.stats(),
        }