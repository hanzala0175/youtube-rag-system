"""
The actual RAG chain: retrieve relevant transcript chunks, then generate an
answer grounded in them using a free Groq-hosted LLM.

This is built with plain LangChain Expression Language (LCEL) runnables
(`langchain_core.runnables`) rather than the legacy `RetrievalQA` chain
class. LCEL lives in `langchain-core`, which has strong backwards-compat
guarantees, so this pipeline is far less likely to break on a future
LangChain upgrade than code built on the older chains API.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_groq import ChatGroq

from backend.config import settings
from backend.core.vector_store import load_index

SYSTEM_PROMPT = """You are a precise assistant that answers questions using ONLY the \
provided excerpts from a YouTube video's transcript.

Rules:
- Answer using only the given context. If the context does not contain the answer, \
say clearly that the video does not appear to cover that.
- Be concise and well-organized. Use short paragraphs or bullet points where helpful.
- When you reference something specific, mention its timestamp in [MM:SS] format, \
taken from the context labels.
- Do not invent timestamps or facts that are not present in the context.

Context from the video transcript:
{context}
"""


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    if not settings.groq_api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
            "and add it to your .env file."
        )
    return ChatGroq(
        model=settings.groq_model,
        temperature=settings.llm_temperature,
        api_key=settings.groq_api_key,
    )


def _format_docs_for_prompt(docs: list[Document]) -> str:
    return "\n\n".join(f"[{d.metadata.get('timestamp', '?:??')}] {d.page_content}" for d in docs)


def get_retriever(video_id: str, top_k: int):
    vector_store = load_index(video_id)
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})


def build_chain(video_id: str, top_k: int):
    retriever = get_retriever(video_id, top_k)
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )

    chain = (
        {
            "context": retriever | RunnableLambda(_format_docs_for_prompt),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def answer_question(video_id: str, question: str, top_k: int | None = None) -> tuple[str, list[Document]]:
    """Run the full RAG pipeline and return (answer_text, source_documents)."""
    k = top_k or settings.default_top_k
    chain, retriever = build_chain(video_id, k)

    answer = chain.invoke(question)
    source_docs = retriever.invoke(question)
    return answer, source_docs
