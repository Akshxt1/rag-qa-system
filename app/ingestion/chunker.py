import re
from typing import List, Dict
from loguru import logger


class TextChunker:
    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 60):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, doc: Dict) -> List[Dict]:
        text = doc.get("text", "").strip()
        if not text:
            return []

        sentences = self._split_into_sentences(text)
        raw_chunks = self._build_chunks(sentences)

        chunks = []
        for idx, chunk_text in enumerate(raw_chunks):
            chunks.append({
                "text": chunk_text,
                "source": doc["source"],
                "filename": doc.get("filename", doc["source"]),
                "doc_type": doc.get("doc_type", "unknown"),
                "chunk_idx": idx,
                "total_chunks": len(raw_chunks),
                "word_count": len(chunk_text.split()),
            })
        return chunks

    def chunk_documents(self, docs: List[Dict]) -> List[Dict]:
        all_chunks = []
        for doc in docs:
            doc_chunks = self.chunk_document(doc)
            all_chunks.extend(doc_chunks)
            logger.debug(f"{doc.get('filename', '?')} → {len(doc_chunks)} chunks")
        logger.info(f"Chunking complete: {len(docs)} docs → {len(all_chunks)} chunks")
        return all_chunks

    def _split_into_sentences(self, text: str) -> List[str]:
        text = re.sub(r"\n{3,}", "\n\n", text)
        protected = re.sub(
            r"\b(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|Fig|vol|no|pp|est)\.",
            r"\1<DOT>", text, flags=re.IGNORECASE,
        )
        protected = re.sub(r"(\d+)\.(\d+)", r"\1<DOT>\2", protected)
        parts = re.split(r"(?<=[.!?])\s+", protected)
        sentences = [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]
        expanded = []
        for s in sentences:
            paras = s.split("\n\n")
            expanded.extend([p.strip() for p in paras if p.strip()])
        return expanded

    def _build_chunks(self, sentences: List[str]) -> List[str]:
        chunks = []
        current_words = []
        for sentence in sentences:
            sentence_words = sentence.split()
            if len(current_words) + len(sentence_words) > self.chunk_size and current_words:
                chunks.append(" ".join(current_words))
                current_words = current_words[-self.chunk_overlap:]
            current_words.extend(sentence_words)
        if current_words:
            chunks.append(" ".join(current_words))
        return chunks