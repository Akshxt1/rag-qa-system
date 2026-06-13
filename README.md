# 🧠 IntelliRAG — Intelligent RAG System

<div align="center">

![IntelliRAG Banner](ui/assets/logo.png)

**An end-to-end Retrieval-Augmented Generation system for intelligent document Q&A**

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38+-red?style=flat-square&logo=streamlit)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Embeddings-yellow?style=flat-square&logo=huggingface)](https://huggingface.co)
[![Gemini](https://img.shields.io/badge/Google-Gemini-blue?style=flat-square&logo=google)](https://ai.google.dev)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Store-orange?style=flat-square)](https://github.com/facebookresearch/faiss)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

[Live Demo](https://intellirag.streamlit.app) · [Report Bug](https://github.com/Akshxt1/rag-qa-system/issues) · [Request Feature](https://github.com/Akshxt1/rag-qa-system/issues)

</div>

---

## 📌 Overview

**IntelliRAG** is a production-grade Retrieval-Augmented Generation (RAG) pipeline that lets you upload any document — PDF, DOCX, or TXT — and ask questions in plain English. Answers are grounded in your documents with page-level citations, eliminating hallucinations.

Built as a portfolio project demonstrating end-to-end ML engineering: from document parsing and vector indexing to hybrid retrieval, neural reranking, and LLM-based generation.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 **Multi-format ingestion** | PDF (with page tracking), DOCX (tables included), TXT, MD |
| 🔍 **Hybrid Search** | FAISS vector search + BM25 keyword search fused via Reciprocal Rank Fusion |
| 🎯 **CrossEncoder Reranking** | ms-marco MiniLM reranker for precision retrieval |
| 📍 **Page Citations** | Answers cite exact page numbers from source documents |
| 💬 **Chat History** | Multiple named chat sessions with inline renaming |
| 📁 **Document Filtering** | Query across all docs or filter to specific ones |
| ⚙️ **Configurable** | Tune model, temperature, top-k, hybrid/reranking toggles |
| 🚀 **REST API** | FastAPI endpoints for `/ingest`, `/query`, `/health`, `/stats` |
| 🌐 **Streamlit UI** | Dark-themed responsive chat interface |

---

## 🏗️ Architecture

```
                        ┌─────────────────────────────────────────┐
                        │            IntelliRAG Pipeline           │
                        │                                          │
  Documents             │  ┌──────────┐    ┌───────────────────┐  │
  PDF/DOCX/TXT  ──────► │  │  Loader  │──► │  Text Chunker     │  │
                        │  │ +page no.│    │  (sentence-aware   │  │
                        │  └──────────┘    │   + overlap)       │  │
                        │                 └────────┬──────────┘  │
                        │                          │              │
                        │                          ▼              │
                        │               ┌─────────────────┐       │
                        │               │  HuggingFace    │       │
                        │               │  Embedder       │       │
                        │               │ all-MiniLM-L6   │       │
                        │               └────────┬────────┘       │
                        │                        │                │
                        │              ┌─────────▼────────┐       │
                        │              │  FAISS + BM25    │       │
                        │              │  Vector Store    │       │
                        │              │  (persistent)    │       │
                        │              └─────────┬────────┘       │
                        │                        │                │
  User Query   ──────► │         ┌──────────────▼──────────────┐ │
                        │         │      Hybrid Retrieval       │ │
                        │         │  FAISS + BM25 + RRF Fusion  │ │
                        │         └──────────────┬──────────────┘ │
                        │                        │                │
                        │         ┌──────────────▼──────────────┐ │
                        │         │   CrossEncoder Reranker     │ │
                        │         │   ms-marco-MiniLM-L-6-v2   │ │
                        │         └──────────────┬──────────────┘ │
                        │                        │                │
                        │         ┌──────────────▼──────────────┐ │
                        │         │      Google Gemini          │ │
                        │         │  (context-grounded RAG      │ │
                        │         │   prompt + page citations)  │ │
                        │         └──────────────┬──────────────┘ │
                        └────────────────────────┼────────────────┘
                                                 │
                                                 ▼
                                    Answer + Page Citations
```

---

## 🚀 Retrieval Pipeline

IntelliRAG uses a **4-stage retrieval pipeline** for maximum accuracy:

```
Query
  │
  ├──► FAISS Vector Search  (semantic similarity, top-20)
  │
  ├──► BM25 Keyword Search  (exact keyword matching, top-20)
  │
  ├──► RRF Fusion           (Reciprocal Rank Fusion merges both)
  │
  ├──► CrossEncoder Rerank  (neural reranker, top-5)
  │
  └──► Gemini Generation    (grounded answer with page citations)
```

This multi-stage approach significantly reduces hallucination vs. direct prompting.

---

## 📁 Project Structure

```
intellirag/
├── app/
│   ├── ingestion/
│   │   ├── loader.py          # PDF/DOCX/TXT loader with page tracking
│   │   └── chunker.py         # Sentence-aware chunker with page metadata
│   ├── embeddings/
│   │   └── embedder.py        # HuggingFace sentence-transformers wrapper
│   ├── vectorstore/
│   │   └── faiss_store.py     # FAISS IndexFlatIP + disk persistence
│   ├── retrieval/
│   │   ├── bm25_index.py      # BM25Okapi keyword search index
│   │   ├── reranker.py        # CrossEncoder neural reranker
│   │   └── hybrid_search.py   # RRF fusion of FAISS + BM25
│   ├── generation/
│   │   └── gemini.py          # Google Gemini generator with page citations
│   └── pipeline/
│       └── rag.py             # End-to-end RAG orchestrator
├── api/
│   └── routes.py              # FastAPI REST endpoints
├── ui/
│   ├── streamlit_app.py       # Streamlit chat UI
│   └── assets/
│       └── logo.png           # IntelliRAG logo
├── config.py                  # Environment config
├── requirements.txt
├── runtime.txt                # Python 3.11 pin
├── pyproject.toml
└── .env.example
```

---

## ⚙️ Setup

### Prerequisites
- Python 3.11
- Google Gemini API key → [Get one free](https://aistudio.google.com/app/apikey)

### 1. Clone & install

```bash
git clone https://github.com/Akshxt1/rag-qa-system.git
cd rag-qa-system

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
# or with uv (faster):
uv pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
HF_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
GEMINI_MODEL=gemini-2.5-flash
CHUNK_SIZE=400
CHUNK_OVERLAP=60
TOP_K=5
CANDIDATE_K=30
ENABLE_HYBRID_SEARCH=true
ENABLE_RERANKING=true
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
VECTOR_STORE_PATH=./vector_store
UPLOAD_DIR=./uploaded_docs
```

### 3. Run

**Streamlit UI:**
```bash
streamlit run ui/streamlit_app.py
```
→ Open http://localhost:8501

**REST API:**
```bash
python -m api.routes
# or
uvicorn api.routes:app --reload --port 8000
```
→ API docs at http://localhost:8000/docs

---

## 🔌 REST API

```bash
# Health check
curl http://localhost:8000/health

# Ingest documents
curl -X POST http://localhost:8000/ingest \
  -F "files=@report.pdf" \
  -F "files=@notes.docx"

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key findings?", "top_k": 5}'

# Stats
curl http://localhost:8000/stats
```

---

## 🧠 Tech Stack

| Layer | Technology |
|---|---|
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, L2-normalized) |
| **Vector Store** | FAISS `IndexFlatIP` — inner product on normalized vectors = cosine sim |
| **Keyword Search** | BM25Okapi via `rank-bm25` |
| **Fusion** | Reciprocal Rank Fusion (RRF, k=60) |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Generator** | Google Gemini (via `google-genai` SDK) |
| **PDF Parsing** | `pdfplumber` with per-page text extraction |
| **DOCX Parsing** | `python-docx` including table extraction |
| **API** | FastAPI + Uvicorn |
| **UI** | Streamlit |

---

## 📊 Performance

| Metric | Value |
|---|---|
| Response time | < 5 seconds (typical 2–4s) |
| Embedding | ~200ms for 400-word chunks |
| FAISS search | < 10ms for 10K vectors |
| Reranking | ~300ms for 20 candidates |
| Hallucination rate | Significantly lower vs. direct prompting |

---

## 🎛️ Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `HF_MODEL_NAME` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model string |
| `CHUNK_SIZE` | `400` | Words per chunk |
| `CHUNK_OVERLAP` | `60` | Overlap words between chunks |
| `TOP_K` | `5` | Final chunks passed to Gemini |
| `CANDIDATE_K` | `30` | Candidates before reranking |
| `ENABLE_HYBRID_SEARCH` | `true` | Enable BM25 + FAISS fusion |
| `ENABLE_RERANKING` | `true` | Enable CrossEncoder reranking |

---

## 🔮 Roadmap

- [ ] HyDE (Hypothetical Document Embeddings) for better query matching
- [ ] Query rewriting — generate 3 phrasings, merge results
- [ ] Parent-child chunking for richer context
- [ ] RAPTOR — hierarchical document summarization
- [ ] Multi-user support with persistent chat history
- [ ] Docker deployment
- [ ] Batch document ingestion via API

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first.

1. Fork the repo
2. Create your feature branch (`git checkout -b feat/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feat/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with ❤️ using HuggingFace · FAISS · BM25 · CrossEncoder · Google Gemini · Streamlit

**[⭐ Star this repo](https://github.com/Akshxt1/rag-qa-system)** if you found it useful!
