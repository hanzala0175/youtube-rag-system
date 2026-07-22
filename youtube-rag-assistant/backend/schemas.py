"""Request/response models shared by the API layer."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProcessVideoRequest(BaseModel):
    youtube_url: str = Field(..., description="Full YouTube URL or bare 11-char video ID")
    chunk_size: int | None = Field(default=None, ge=200, le=4000)
    chunk_overlap: int | None = Field(default=None, ge=0, le=1000)
    force_reprocess: bool = Field(
        default=False, description="Rebuild the index even if one already exists for this video"
    )


class ProcessVideoResponse(BaseModel):
    video_id: str
    title: str
    channel: str | None = None
    thumbnail_url: str | None = None
    num_chunks: int
    already_indexed: bool


class VideoStatusResponse(BaseModel):
    video_id: str
    indexed: bool


class ChatRequest(BaseModel):
    video_id: str = Field(..., description="ID of a previously processed video")
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=10)


class SourceChunk(BaseModel):
    timestamp: str
    start_seconds: float
    text: str
    youtube_url: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


class ErrorResponse(BaseModel):
    detail: str
