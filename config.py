import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ──────────────────────────────────────────────────────────────────
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
HUGGINGFACE_TOKEN: str = os.getenv("HUGGINGFACE_TOKEN", "")

# ── Models ────────────────────────────────────────────────────────────────────
HF_MODEL_NAME: str = os.getenv(
    "HF_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
)
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 400))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 60))

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K: int = int(os.getenv("TOP_K", 5))

# ── Storage ───────────────────────────────────────────────────────────────────
VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", "./vector_store")
UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploaded_docs")

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", 8000))

# ── Ensure dirs exist ─────────────────────────────────────────────────────────
Path(VECTOR_STORE_PATH).mkdir(parents=True, exist_ok=True)
Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)

# ── Validation ────────────────────────────────────────────────────────────────
def validate():
    errors = []
    if not GEMINI_API_KEY:
        errors.append("GEMINI_API_KEY is not set in .env")
    if errors:
        raise EnvironmentError("\n".join(errors))
    

# ── Retrieval Improvements ────────────────────────────────────────────────────
ENABLE_HYBRID_SEARCH: bool = os.getenv("ENABLE_HYBRID_SEARCH", "true").lower() == "true"
ENABLE_RERANKING: bool = os.getenv("ENABLE_RERANKING", "true").lower() == "true"
RERANKER_MODEL: str = os.getenv(
    "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
CANDIDATE_K: int = int(os.getenv("CANDIDATE_K", 30))