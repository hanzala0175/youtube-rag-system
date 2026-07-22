"""
Fetches transcripts and lightweight metadata for a YouTube video.

Transcript fetching uses `youtube-transcript-api` v1.x, which switched from
static `YouTubeTranscriptApi.get_transcript(...)` calls to an instance-based
API: `YouTubeTranscriptApi().fetch(video_id)`. The older static methods were
removed in v1.2, so this module intentionally uses only the current API.

Video title/channel/thumbnail come from YouTube's public oEmbed endpoint,
which requires no API key.
"""

from __future__ import annotations

import re

import requests
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

_VIDEO_ID_PATTERNS = [
    re.compile(r"(?:youtube\.com/watch\?v=|youtube\.com/live/)([A-Za-z0-9_-]{11})"),
    re.compile(r"youtube\.com/embed/([A-Za-z0-9_-]{11})"),
    re.compile(r"youtube\.com/shorts/([A-Za-z0-9_-]{11})"),
    re.compile(r"youtu\.be/([A-Za-z0-9_-]{11})"),
]
_BARE_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{11}$")

OEMBED_URL = "https://www.youtube.com/oembed"


class VideoProcessingError(Exception):
    """Raised for any user-facing failure while loading a video's transcript."""


def extract_video_id(url_or_id: str) -> str:
    """Extract the 11-character YouTube video ID from a URL or return it as-is."""
    candidate = url_or_id.strip()

    if _BARE_ID_PATTERN.match(candidate):
        return candidate

    for pattern in _VIDEO_ID_PATTERNS:
        match = pattern.search(candidate)
        if match:
            return match.group(1)

    raise VideoProcessingError(
        "Could not extract a valid YouTube video ID from the given input. "
        "Provide a full YouTube URL (watch, youtu.be, shorts, or embed) or an 11-character video ID."
    )


def fetch_video_metadata(video_id: str) -> dict:
    """Fetch title/author/thumbnail via YouTube's public oEmbed endpoint (no API key needed)."""
    watch_url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        response = requests.get(
            OEMBED_URL,
            params={"url": watch_url, "format": "json"},
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()
        return {
            "title": data.get("title", "Untitled video"),
            "channel": data.get("author_name"),
            "thumbnail_url": data.get("thumbnail_url"),
        }
    except requests.RequestException:
        # Metadata is a nice-to-have; never let it block indexing.
        return {"title": "Untitled video", "channel": None, "thumbnail_url": None}


def fetch_transcript_documents(video_id: str, video_title: str = "") -> list[Document]:
    """
    Fetch the raw transcript and convert each caption snippet into a
    LangChain Document, one per snippet, preserving start/duration metadata
    so later chunking can attach accurate timestamps.
    """
    ytt_api = YouTubeTranscriptApi()

    try:
        fetched = ytt_api.fetch(video_id, languages=["en", "en-US", "en-GB"])
    except NoTranscriptFound:
        # No English transcript — fall back to whatever is available
        # (manually created transcripts are preferred automatically by the library).
        try:
            transcript_list = ytt_api.list(video_id)
            transcript = next(iter(transcript_list))
            fetched = transcript.fetch()
        except (NoTranscriptFound, StopIteration) as exc:
            raise VideoProcessingError(
                "This video has no available transcript in any language."
            ) from exc
    except TranscriptsDisabled as exc:
        raise VideoProcessingError(
            "Transcripts are disabled for this video by the uploader."
        ) from exc
    except VideoUnavailable as exc:
        raise VideoProcessingError("This video is unavailable (private, deleted, or region-locked).") from exc
    except Exception as exc:  # noqa: BLE001 — surface any other library error as a clean message
        raise VideoProcessingError(f"Failed to fetch transcript: {exc}") from exc

    documents: list[Document] = []
    for snippet in fetched:
        text = snippet.text.strip()
        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "video_id": video_id,
                    "video_title": video_title,
                    "start": float(snippet.start),
                    "duration": float(snippet.duration),
                },
            )
        )

    if not documents:
        raise VideoProcessingError("The transcript for this video is empty.")

    return documents
