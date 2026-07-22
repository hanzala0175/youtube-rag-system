from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from backend.config import settings
from backend.core import vector_store
from backend.core.rag_pipeline import answer_question
from backend.core.text_processor import chunk_transcript
from backend.core.youtube_loader import (
    VideoProcessingError,
    extract_video_id,
    fetch_transcript_documents,
    fetch_video_metadata,
)
from backend.schemas import (
    ChatRequest,
    ChatResponse,
    ProcessVideoRequest,
    ProcessVideoResponse,
    SourceChunk,
    VideoStatusResponse,
)

logger = logging.getLogger("youtube_rag.api")

router = APIRouter()


@router.get("/health", tags=["system"])
def health_check() -> dict:
    return {"status": "ok"}


@router.post("/videos/process", response_model=ProcessVideoResponse, tags=["videos"])
def process_video(payload: ProcessVideoRequest) -> ProcessVideoResponse:
    """
    Fetch a YouTube video's transcript, chunk it, embed it, and build a FAISS
    index. If an index already exists for this video, it is reused unless
    `force_reprocess=True`.
    """
    try:
        video_id = extract_video_id(payload.youtube_url)
    except VideoProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    metadata = fetch_video_metadata(video_id)

    if vector_store.index_exists(video_id) and not payload.force_reprocess:
        try:
            index = vector_store.load_index(video_id)
            num_chunks = index.index.ntotal
        except Exception:  # noqa: BLE001 — fall through to reprocessing on any load failure
            num_chunks = 0
        else:
            return ProcessVideoResponse(
                video_id=video_id,
                title=metadata["title"],
                channel=metadata["channel"],
                thumbnail_url=metadata["thumbnail_url"],
                num_chunks=num_chunks,
                already_indexed=True,
            )

    try:
        raw_docs = fetch_transcript_documents(video_id, video_title=metadata["title"])
    except VideoProcessingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    chunks = chunk_transcript(
        raw_docs,
        chunk_size=payload.chunk_size or settings.chunk_size,
        chunk_overlap=payload.chunk_overlap or settings.chunk_overlap,
    )

    try:
        vector_store.build_and_save_index(video_id, chunks)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to build index for video_id=%s", video_id)
        raise HTTPException(status_code=500, detail=f"Failed to build vector index: {exc}") from exc

    return ProcessVideoResponse(
        video_id=video_id,
        title=metadata["title"],
        channel=metadata["channel"],
        thumbnail_url=metadata["thumbnail_url"],
        num_chunks=len(chunks),
        already_indexed=False,
    )


@router.get("/videos/{video_id}/status", response_model=VideoStatusResponse, tags=["videos"])
def video_status(video_id: str) -> VideoStatusResponse:
    return VideoStatusResponse(video_id=video_id, indexed=vector_store.index_exists(video_id))


@router.delete("/videos/{video_id}", tags=["videos"])
def delete_video_index(video_id: str) -> dict:
    deleted = vector_store.delete_index(video_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="No index found for this video_id.")
    return {"deleted": True, "video_id": video_id}


@router.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(payload: ChatRequest) -> ChatResponse:
    if not vector_store.index_exists(payload.video_id):
        raise HTTPException(
            status_code=404,
            detail="This video hasn't been processed yet. Call /videos/process first.",
        )

    try:
        answer, source_docs = answer_question(
            video_id=payload.video_id,
            question=payload.question,
            top_k=payload.top_k,
        )
    except RuntimeError as exc:
        # e.g. missing GROQ_API_KEY
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("RAG pipeline failed for video_id=%s", payload.video_id)
        raise HTTPException(status_code=500, detail=f"Failed to generate an answer: {exc}") from exc

    sources = [
        SourceChunk(
            timestamp=doc.metadata.get("timestamp", "0:00"),
            start_seconds=doc.metadata.get("start", 0.0),
            text=doc.page_content,
            youtube_url=f"https://www.youtube.com/watch?v={payload.video_id}&t={int(doc.metadata.get('start', 0))}s",
        )
        for doc in source_docs
    ]

    return ChatResponse(answer=answer, sources=sources)
