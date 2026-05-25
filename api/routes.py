import os
import shutil
import tempfile
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger

app = FastAPI(
    title="RAG Document Q&A API",
    description="End-to-end RAG over custom document corpora.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from app.pipeline.rag import RAGPipeline
        _pipeline = RAGPipeline()
    return _pipeline


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: Optional[int] = Field(None, ge=1, le=20)


@app.get("/health")
def health():
    pipeline = get_pipeline()
    return {
        "status": "ok",
        "pipeline_ready": pipeline.is_ready,
        "stats": pipeline.status(),
    }


@app.get("/stats")
def stats():
    return get_pipeline().status()


@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    pipeline = get_pipeline()
    ALLOWED = {".pdf", ".docx", ".txt", ".md"}
    temp_paths = []
    errors = []

    try:
        for file in files:
            ext = os.path.splitext(file.filename)[1].lower()
            if ext not in ALLOWED:
                errors.append(f"{file.filename}: unsupported type '{ext}'")
                continue
            tmp_dir = tempfile.mkdtemp()
            tmp_path = os.path.join(tmp_dir, file.filename)
            with open(tmp_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            temp_paths.append((tmp_path, tmp_dir))

        if not temp_paths and errors:
            raise HTTPException(status_code=400, detail={"errors": errors})

        result = pipeline.ingest_files([p for p, _ in temp_paths])
        return JSONResponse(
            status_code=200 if result.success else 422,
            content={**result.to_dict(), "file_errors": errors},
        )
    finally:
        for _, tmp_dir in temp_paths:
            shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/query")
def query(request: QueryRequest):
    pipeline = get_pipeline()
    if not pipeline.is_ready:
        raise HTTPException(status_code=400,
                            detail="No documents ingested yet.")
    result = pipeline.query(request.question, top_k=request.top_k)
    return result.to_dict()


@app.post("/reset")
def reset():
    get_pipeline().reset()
    return {"message": "Vector store cleared."}


if __name__ == "__main__":
    import uvicorn
    import config
    uvicorn.run("api.routes:app", host=config.API_HOST,
                port=config.API_PORT, reload=True)