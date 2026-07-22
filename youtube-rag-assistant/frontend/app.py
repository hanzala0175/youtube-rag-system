"""
Streamlit frontend for the YouTube RAG Assistant.

Run with:
    streamlit run frontend/app.py

Talks to the FastAPI backend over HTTP — set BACKEND_URL if it's not running
on the default localhost:8000.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="YouTube RAG Assistant", page_icon="🎬", layout="wide")


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
def _init_state() -> None:
    defaults = {
        "video_id": None,
        "video_title": None,
        "video_channel": None,
        "thumbnail_url": None,
        "chat_history": [],  # list of {"role": "user"/"assistant", "content": str, "sources": list}
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


_init_state()


# --------------------------------------------------------------------------
# Backend helpers
# --------------------------------------------------------------------------
def backend_healthy() -> bool:
    try:
        r = requests.get(f"{BACKEND_URL}/api/health", timeout=3)
        return r.ok
    except requests.RequestException:
        return False


def process_video(youtube_url: str, force_reprocess: bool = False) -> dict | None:
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/videos/process",
            json={"youtube_url": youtube_url, "force_reprocess": force_reprocess},
            timeout=180,  # first-time embedding can take a while for long videos
        )
    except requests.RequestException as exc:
        st.error(f"Could not reach the backend at {BACKEND_URL}: {exc}")
        return None

    if not response.ok:
        detail = _extract_error(response)
        st.error(f"Failed to process video: {detail}")
        return None

    return response.json()


def ask_question(video_id: str, question: str) -> dict | None:
    try:
        response = requests.post(
            f"{BACKEND_URL}/api/chat",
            json={"video_id": video_id, "question": question},
            timeout=60,
        )
    except requests.RequestException as exc:
        st.error(f"Could not reach the backend at {BACKEND_URL}: {exc}")
        return None

    if not response.ok:
        detail = _extract_error(response)
        st.error(f"Failed to get an answer: {detail}")
        return None

    return response.json()


def _extract_error(response: requests.Response) -> str:
    try:
        return response.json().get("detail", response.text)
    except ValueError:
        return response.text


# --------------------------------------------------------------------------
# Sidebar — video input
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("🎬 YouTube RAG Assistant")
    st.caption("Ask questions about any YouTube video, grounded in its transcript.")

    if not backend_healthy():
        st.error(f"Backend not reachable at `{BACKEND_URL}`.\nStart it with:\n\n`uvicorn backend.main:app --reload`")

    st.divider()
    youtube_url = st.text_input("YouTube URL or video ID", placeholder="https://www.youtube.com/watch?v=...")
    force_reprocess = st.checkbox("Force re-process (rebuild index)", value=False)

    if st.button("Process video", type="primary", use_container_width=True, disabled=not youtube_url):
        with st.spinner("Fetching transcript and building the vector index... this can take a minute for long videos."):
            result = process_video(youtube_url, force_reprocess=force_reprocess)

        if result:
            st.session_state.video_id = result["video_id"]
            st.session_state.video_title = result["title"]
            st.session_state.video_channel = result.get("channel")
            st.session_state.thumbnail_url = result.get("thumbnail_url")
            st.session_state.chat_history = []

            if result["already_indexed"]:
                st.info(f"Loaded existing index ({result['num_chunks']} chunks).")
            else:
                st.success(f"Indexed {result['num_chunks']} transcript chunks.")

    if st.session_state.video_id:
        st.divider()
        st.subheader("Current video")
        if st.session_state.thumbnail_url:
            st.image(st.session_state.thumbnail_url, use_container_width=True)
        st.markdown(f"**{st.session_state.video_title}**")
        if st.session_state.video_channel:
            st.caption(st.session_state.video_channel)
        st.caption(f"Video ID: `{st.session_state.video_id}`")

        if st.button("Clear conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()


# --------------------------------------------------------------------------
# Main panel — chat
# --------------------------------------------------------------------------
st.header("Chat with the video")

if not st.session_state.video_id:
    st.info("👈 Paste a YouTube URL in the sidebar and click **Process video** to get started.")
    st.stop()

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander(f"Sources ({len(message['sources'])})"):
                for source in message["sources"]:
                    st.markdown(f"**[{source['timestamp']}]**({source['youtube_url']})")
                    st.caption(source["text"])

question = st.chat_input("Ask something about this video...")

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask_question(st.session_state.video_id, question)

        if result:
            st.markdown(result["answer"])
            sources = result.get("sources", [])
            if sources:
                with st.expander(f"Sources ({len(sources)})"):
                    for source in sources:
                        st.markdown(f"**[{source['timestamp']}]**({source['youtube_url']})")
                        st.caption(source["text"])
            st.session_state.chat_history.append(
                {"role": "assistant", "content": result["answer"], "sources": sources}
            )
        else:
            st.session_state.chat_history.pop()  # remove the user turn if the call failed
