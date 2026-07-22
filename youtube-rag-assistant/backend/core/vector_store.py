"""
FAISS vector store management, one index per YouTube video, persisted to disk
under `settings.vector_store_dir/<video_id>/`.

Embeddings: `sentence-transformers/all-MiniLM-L6-v2` via `langchain-huggingface`.
This runs fully locally on CPU — no API key, no per-call cost.

Vector store: FAISS via `langchain_community.vectorstores.FAISS`.

Note: as of the LangChain v1.0 restructuring, `langchain-community` is
archived/sunset (no new features), but it is NOT deleted and remains the
canonical home of the FAISS integration — `langchain-classic` (the new home
for other legacy helpers) actually re-exports FAISS from
`langchain-community` under the hood via a lazy shim, and importing it from
there prints a deprecation warning telling you to import from
`langchain_community` directly. So we do that directly here, and simply pin
`langchain-community` as an explicit dependency (see requirements.txt).
"""

from __future__ import annotations

import shutil
from functools import lru_cache
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from backend.config import settings


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the embedding model once per process and reuse it across requests."""
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )


def _index_dir(video_id: str) -> Path:
    return settings.vector_store_dir / video_id


def index_exists(video_id: str) -> bool:
    return (_index_dir(video_id) / "index.faiss").exists()


def build_and_save_index(video_id: str, chunks: list[Document]) -> FAISS:
    """Embed chunks, build a FAISS index, and persist it to disk."""
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    index_dir = _index_dir(video_id)
    index_dir.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_dir))
    return vector_store


def load_index(video_id: str) -> FAISS:
    """Load a previously built FAISS index from disk."""
    if not index_exists(video_id):
        raise FileNotFoundError(f"No index found for video_id={video_id!r}. Process the video first.")

    embeddings = get_embeddings()
    return FAISS.load_local(
        str(_index_dir(video_id)),
        embeddings,
        # Safe here: we only ever load indexes this application wrote itself.
        allow_dangerous_deserialization=True,
    )


def delete_index(video_id: str) -> bool:
    index_dir = _index_dir(video_id)
    if index_dir.exists():
        shutil.rmtree(index_dir)
        return True
    return False
