# YouTube RAG Assistant

Ask questions about any YouTube video and get answers grounded in its actual
transcript, with clickable timestamp citations.

**Stack:** FastAPI (backend) · Streamlit (frontend) · LangChain (RAG orchestration,
LCEL) · FAISS (vector store) · Groq (free-tier LLM) · HuggingFace `sentence-transformers`
(free, local embeddings)

```
┌────────────┐  HTTP   ┌──────────────┐   retrieval   ┌────────────┐
│  Streamlit │ ──────► │   FastAPI    │ ─────────────► │   FAISS    │
│  frontend  │ ◄────── │   backend    │ ◄───────────── │   index    │
└────────────┘         └──────┬───────┘                └────────────┘
                               │
                     ┌─────────┴─────────┐
                     │  Groq (LLM, free) │
                     └───────────────────┘
```

## How it works

1. **Transcript ingestion** — `youtube-transcript-api` pulls the video's
   captions (manual or auto-generated), and YouTube's public oEmbed endpoint
   supplies the title/channel/thumbnail (no API key needed for either).
2. **Timestamp-aware chunking** — raw caption snippets (2–8 seconds each) are
   merged into ~1000-character chunks while preserving the start time of each
   chunk, so every retrieved passage can be cited as `[MM:SS]`.
3. **Embedding + indexing** — chunks are embedded locally with
   `sentence-transformers/all-MiniLM-L6-v2` (CPU, free, no API key) and stored
   in a per-video FAISS index on disk, so re-asking questions about a video
   you've already processed is instant.
4. **Retrieval-augmented generation** — a LangChain Expression Language (LCEL)
   chain retrieves the top-k relevant chunks and asks a free Groq-hosted LLM
   to answer strictly from that context.

## Project layout

```
youtube-rag-assistant/
├── backend/
│   ├── main.py              # FastAPI app + CORS
│   ├── config.py            # pydantic-settings configuration
│   ├── schemas.py           # request/response models
│   ├── api/
│   │   └── routes.py        # /api/videos/process, /api/chat, etc.
│   └── core/
│       ├── youtube_loader.py    # transcript + metadata fetching
│       ├── text_processor.py    # timestamp-aware chunking
│       ├── vector_store.py      # FAISS build/save/load
│       └── rag_pipeline.py      # LCEL retrieval + generation chain
├── frontend/
│   └── app.py                # Streamlit chat UI
├── data/vector_store/         # persisted FAISS indexes (one folder per video)
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

### 1. Get a free Groq API key

Sign up at [console.groq.com](https://console.groq.com/keys) — the free tier
is generous and requires no credit card.

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

`sentence-transformers` pulls in PyTorch, which by default resolves to a
large CUDA-enabled build even on machines without a GPU. If you're running
CPU-only (typical for this project), install the CPU wheel first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Otherwise, just:

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example .env
# then edit .env and paste your GROQ_API_KEY
```

### 5. Run the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

The interactive API docs are then available at `http://localhost:8000/docs`.

### 6. Run the frontend (in a second terminal)

```bash
streamlit run frontend/app.py
```

Open the URL Streamlit prints (typically `http://localhost:8501`), paste a
YouTube URL, click **Process video**, and start asking questions.

## API reference

| Method | Endpoint                     | Description                                  |
|--------|-------------------------------|-----------------------------------------------|
| GET    | `/api/health`                 | Liveness check                                |
| POST   | `/api/videos/process`         | Fetch transcript, chunk, embed, build index   |
| GET    | `/api/videos/{video_id}/status` | Check whether a video is already indexed   |
| DELETE | `/api/videos/{video_id}`      | Remove a video's index from disk              |
| POST   | `/api/chat`                   | Ask a question about a processed video        |

Example `POST /api/videos/process`:

```json
{ "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ" }
```

Example `POST /api/chat`:

```json
{ "video_id": "dQw4w9WgXcQ", "question": "What is the main point made in the first two minutes?" }
```

Response:

```json
{
  "answer": "The video opens by explaining ... [0:42]",
  "sources": [
    { "timestamp": "0:42", "start_seconds": 42.0, "text": "...", "youtube_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s" }
  ]
}
```

## Notes on the LangChain version used here

As of mid-2026, LangChain went through a major restructuring (v1.0): most
legacy chains, retrievers, and hub helpers moved into a new
**`langchain-classic`** package, and `langchain-community` was
archived/sunset (no new releases, but still fully functional and installable).

The FAISS vectorstore integration, specifically, still lives in
`langchain-community` — `langchain-classic` only re-exports it through a
deprecated compatibility shim that prints a warning telling you to import
from `langchain_community` directly. So this project:

- imports FAISS from `langchain_community.vectorstores` directly (verified
  against the installed package — this is the non-deprecated path today),
- builds the actual RAG chain with plain LCEL runnables
  (`langchain_core.prompts`, `langchain_core.output_parsers`,
  `langchain_core.runnables`) instead of the legacy `RetrievalQA` chain
  class, since `langchain-core` has the strongest backwards-compatibility
  guarantees in the ecosystem and this sidesteps the classic/community split
  entirely for the orchestration logic.

If you're adapting this code and hit an `ImportError`, it's almost always
because a tutorial you're looking at predates this restructuring — check
whether the class you need now lives in `langchain_classic` (most legacy
chains/retrievers) or is still in `langchain_community` (most third-party
integrations, including FAISS).

`youtube-transcript-api` also changed its interface in v1.x: the old static
methods (`YouTubeTranscriptApi.get_transcript(...)`) were removed. This
project uses the current instance-based API:
`YouTubeTranscriptApi().fetch(video_id)`.

## Swapping models

- **LLM**: change `GROQ_MODEL` in `.env`. See
  [console.groq.com/docs/models](https://console.groq.com/docs/models) for
  the current free-tier lineup — Groq deprecates specific model IDs
  periodically, so check there if you get a "model decommissioned" error.
- **Embeddings**: change `EMBEDDING_MODEL` in `.env` to any local
  `sentence-transformers` model, e.g. `sentence-transformers/all-mpnet-base-v2`
  for higher quality at the cost of speed.

## Known limitations

- `youtube-transcript-api` scrapes an internal, undocumented YouTube
  endpoint. It has no official support contract, so it can occasionally
  break when YouTube changes its internals — `pip install --upgrade
  youtube-transcript-api` is usually the fix.
- Videos without any captions (manual or auto-generated) cannot be indexed.
- Running from a cloud/datacenter IP can occasionally trigger YouTube rate
  limiting; the library supports residential proxy configuration if you hit
  this at scale (see its own docs).
