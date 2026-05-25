import time
from typing import List, Dict, Optional, Set
from loguru import logger

import config
from app.ingestion.loader import DocumentLoader
from app.ingestion.chunker import TextChunker
from app.embeddings.embedder import Embedder
from app.vectorstore.faiss_store import FAISSVectorStore
from app.generation.gemini import GeminiGenerator


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
    def __init__(self, question, answer, sources, elapsed_seconds, chunks_retrieved):
        self.question = question
        self.answer = answer
        self.sources = sources
        self.elapsed_seconds = round(elapsed_seconds, 2)
        self.chunks_retrieved = chunks_retrieved

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
        self._loaded_from_disk = self.vector_store.load()
        logger.info(f"Pipeline ready. Chunks: {self.vector_store.total_chunks}")

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest_files(self, file_paths: List[str]) -> IngestResult:
        t0 = time.time()
        errors = []
        docs = []
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
        return IngestResult(
            success=True,
            documents_loaded=1,
            chunks_created=len(chunks),
            elapsed_seconds=time.time() - t0,
        )

    # ── Deletion ──────────────────────────────────────────────────────────────

    def delete_document(self, filename: str) -> int:
        """Remove a document from the vector store. Returns chunks removed."""
        removed = self.vector_store.delete_by_source(filename)
        if removed > 0:
            self.vector_store.save()
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
    ) -> QueryResult:
        if not self.is_ready:
            return QueryResult(
                question=question,
                answer="⚠️ No documents ingested yet. Please upload documents first.",
                sources=[],
                elapsed_seconds=0.0,
                chunks_retrieved=0,
            )

        t0 = time.time()
        k = top_k or config.TOP_K

        query_vec = self.embedder.embed_query(question)
        retrieved = self.vector_store.search(
            query_vec,
            top_k=k,
            score_threshold=score_threshold or 0.0,
            filter_sources=filter_sources if filter_sources else None,
        )
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
        )

    # ── Model switching ───────────────────────────────────────────────────────

    def set_model(self, model_name: str):
        self.generator.set_model(model_name)

    # ── Control ───────────────────────────────────────────────────────────────

    def reset(self):
        self.vector_store.clear()

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
            "loaded_from_disk": self._loaded_from_disk,
            **self.vector_store.stats(),
        }