"""EduMIND API — RAG (Retrieval-Augmented Generation) Endpoints.

Routes:
    GET    /api/v1/rag/status       → Collection stats
    POST   /api/v1/rag/ingest       → Upload PDF file(s) → index into Qdrant
    POST   /api/v1/rag/ingest-text  → Ingest plain text into Qdrant
    POST   /api/v1/rag/query        → Semantic QA with optional LLM answer synthesis
    DELETE /api/v1/rag/index        → Wipe entire Qdrant collection

Usage (Postman / curl):
    # Upload and index a PDF
    curl -X POST http://localhost:8000/api/v1/rag/ingest \\
         -F "files=@lecture_slides.pdf"

    # Query the knowledge base
    curl -X POST http://localhost:8000/api/v1/rag/query \\
         -H "Content-Type: application/json" \\
         -d '{"query": "What is the attention mechanism?", "top_k": 5}'

    # Check status
    curl http://localhost:8000/api/v1/rag/status
"""

from __future__ import annotations

import gc

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from edumind.api.dependencies import get_rag
from edumind.models.api import (
    RAGIngestResponse,
    RAGIngestTextRequest,
    RAGQueryRequest,
    RAGQueryResponse,
    RAGStatusResponse,
    RetrievedChunkDTO,
)
from edumind.modules.rag_engine import MultimodalRAG
from edumind.utils.data_manager import get_raw_dir

router = APIRouter(prefix="/rag", tags=["RAG — Knowledge Base"])


@router.get(
    "/status",
    response_model=RAGStatusResponse,
    summary="Vector store status",
    description="Returns Qdrant collection name and number of indexed chunks.",
)
def rag_status(rag: MultimodalRAG = Depends(get_rag)) -> RAGStatusResponse:
    """Return current vector store collection info."""
    info = rag.get_collection_info()
    return RAGStatusResponse(
        collection_name=str(info.get("collection_name", "unknown")),
        points_count=int(info.get("points_count") or 0),
        status=str(info.get("status", "unknown")),
    )


@router.post(
    "/ingest",
    response_model=RAGIngestResponse,
    summary="Ingest PDF files",
    description=(
        "Upload one or more PDF files. Each file is parsed with Docling (layout-aware), "
        "split into chunks, embedded, and stored in Qdrant. "
        "Returns the total number of indexed chunks."
    ),
)
async def ingest_pdfs(
    files: list[UploadFile] = File(..., description="One or more PDF files to index"),
    rag: MultimodalRAG = Depends(get_rag),
) -> RAGIngestResponse:
    """Parse, embed, and store PDF documents in the vector index."""
    if not files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one file is required.",
        )

    total_chunks = 0
    source_files: list[str] = []

    for uploaded_file in files:
        if not uploaded_file.filename or not uploaded_file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"File '{uploaded_file.filename}' is not a PDF.",
            )

        # Persist to data/raw
        dest_path = get_raw_dir() / uploaded_file.filename
        content = await uploaded_file.read()
        with open(dest_path, "wb") as f:
            f.write(content)

        # Parse → embed → store
        chunks = rag.ingest_pdf(dest_path)
        for chunk in chunks:
            chunk.metadata["source_file"] = uploaded_file.filename

        stored = rag.embed_and_store(chunks)
        total_chunks += stored
        source_files.append(uploaded_file.filename)

        # Release memory between large PDFs
        gc.collect()

    return RAGIngestResponse(chunks_indexed=total_chunks, source_files=source_files)


@router.post(
    "/ingest-text",
    response_model=RAGIngestResponse,
    summary="Ingest plain text",
    description=(
        "Ingest a plain-text string (e.g. a lecture transcript) directly into Qdrant. "
        "The text is split into chunks and embedded automatically."
    ),
)
def ingest_text(
    body: RAGIngestTextRequest,
    rag: MultimodalRAG = Depends(get_rag),
) -> RAGIngestResponse:
    """Split, embed, and store a plain text string in the vector index."""
    chunks = rag.ingest_text(
        body.text,
        source_name=body.source_name,
        metadata_extra=body.metadata_extra,
    )
    stored = rag.embed_and_store(chunks)
    return RAGIngestResponse(chunks_indexed=stored, source_files=[body.source_name])


@router.post(
    "/query",
    response_model=RAGQueryResponse,
    summary="Semantic Q&A",
    description=(
        "Perform semantic vector search over the indexed knowledge base. "
        "Optionally generate a synthesized answer via the configured LLM provider. "
        "Returns matched chunks with scores and source metadata."
    ),
)
def query_rag(
    body: RAGQueryRequest,
    rag: MultimodalRAG = Depends(get_rag),
) -> RAGQueryResponse:
    """Run semantic search and (optionally) generate an LLM answer."""
    if not rag.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG engine is not ready. Check vector store connection.",
        )

    results = rag.query(body.query, top_k=body.top_k)

    answer = ""
    if body.generate_answer and results:
        answer = rag.generate_answer(body.query, results)

    result_dtos = [
        RetrievedChunkDTO(text=r.text, score=r.score, metadata=r.metadata)
        for r in results
    ]

    return RAGQueryResponse(
        query=body.query,
        answer=answer,
        results=result_dtos,
    )


@router.delete(
    "/index",
    summary="Clear vector index",
    description="⚠️ Permanently deletes all indexed chunks from the Qdrant collection.",
    status_code=status.HTTP_200_OK,
)
def clear_index(rag: MultimodalRAG = Depends(get_rag)) -> dict:
    """Wipe the entire Qdrant vector collection."""
    success = rag.clear_index()
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to clear the Qdrant index.",
        )
    return {"message": "Vector index cleared successfully."}
