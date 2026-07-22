"""
Timestamp-aware chunking for YouTube transcripts.

Raw transcripts arrive as many tiny (2-8 second) caption snippets. Running a
generic `RecursiveCharacterTextSplitter` over the concatenated text would
lose the per-snippet timestamps, which are the whole point of grounding
answers in "watch from here" citations. Instead, this module merges
consecutive snippets into chunks of roughly `chunk_size` characters while
tracking the start time of the first snippet in each chunk.
"""

from __future__ import annotations

from langchain_core.documents import Document


def format_timestamp(seconds: float) -> str:
    """Format seconds as `MM:SS` or `H:MM:SS` for long videos."""
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def chunk_transcript(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """
    Merge per-snippet transcript Documents into larger, timestamp-tagged
    chunks suitable for embedding.

    Each output Document's metadata includes:
      - start: seconds into the video where the chunk begins
      - timestamp: human-readable MM:SS / H:MM:SS string
      - video_id, video_title: carried over from the source snippets
    """
    if not documents:
        return []

    video_id = documents[0].metadata.get("video_id", "")
    video_title = documents[0].metadata.get("video_title", "")

    chunks: list[Document] = []
    buffer_texts: list[str] = []
    buffer_len = 0
    chunk_start: float | None = None

    def flush() -> None:
        nonlocal buffer_texts, buffer_len, chunk_start
        if not buffer_texts or chunk_start is None:
            return
        text = " ".join(buffer_texts).strip()
        if text:
            chunks.append(
                Document(
                    page_content=text,
                    metadata={
                        "video_id": video_id,
                        "video_title": video_title,
                        "start": chunk_start,
                        "timestamp": format_timestamp(chunk_start),
                    },
                )
            )

    for doc in documents:
        text = doc.page_content.strip()
        if not text:
            continue

        if chunk_start is None:
            chunk_start = doc.metadata["start"]

        buffer_texts.append(text)
        buffer_len += len(text) + 1

        if buffer_len >= chunk_size:
            flush()
            # Carry a small text overlap into the next chunk for context continuity.
            joined = " ".join(buffer_texts)
            overlap_text = joined[-chunk_overlap:].strip() if chunk_overlap else ""
            buffer_texts = [overlap_text] if overlap_text else []
            buffer_len = len(overlap_text)
            # The next chunk's timestamp should reflect where new content starts,
            # which is approximately the end of the snippet we just processed.
            chunk_start = doc.metadata["start"] + doc.metadata["duration"]

    flush()
    return chunks
