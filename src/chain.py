"""Conversational RAG chain with strict hallucination guardrails.

Architecture
------------
1. **Retrieval**    : Two-stage hybrid + CrossEncoder pipeline from
                      :mod:`src.retrieval` surfaces the top-k grounded contexts.
2. **Prompt**       : A strict ``SystemMessage`` forbids the LLM from using
                      outside knowledge; conversation history (last k=5 turns)
                      and the retrieved context are injected per request.
3. **LLM**          : Selected dynamically at runtime — ``ChatOpenAI`` when
                      ``OPENAI_API_KEY`` is set, ``ChatGoogleGenerativeAI``
                      when ``GOOGLE_API_KEY`` is set.
4. **Memory**       : :class:`~langchain_classic.memory.ConversationBufferWindowMemory`
                      (``k=5``) tracks the rolling conversation window.
5. **Source citation**: Each response is appended with a deduplicated
                       ``Sources Used`` section built from chunk metadata.

Public API
----------
- :func:`build_memory`     – factory for a fresh ``ConversationBufferWindowMemory``
- :func:`run_rag_chain`    – main entry point; returns ``(answer, sources)``
- :func:`stream_rag_chain` – streaming variant; yields tokens, returns via StopIteration
"""

import os
import threading
import warnings
from typing import Dict, List, Optional, Tuple

from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.config import settings
from src.ingestion import load_documents, split_documents
from src.logger import get_logger
from src.retrieval import get_hybrid_reranked_retriever, retrieve
from src.vectorstore import build_bm25_retriever, load_faiss_vectorstore

logger = get_logger(__name__)

# ── Strict hallucination-prevention system prompt ────────────────────────────

_SYSTEM_PROMPT = (
    "You are an Enterprise HR & IT Policy Assistant. "
    "Answer questions strictly using the provided context. "
    "If the answer cannot be directly derived from the context, explicitly state: "
    "'I am sorry, but I cannot find this information in the provided company documents.' "
    "Do NOT hallucinate or use outside knowledge. "
    "Be concise, precise, and professional in your responses."
)

# ── Input validation constant ─────────────────────────────────────────────────
# Prevents token overflow and unexpected API costs from excessively long queries.
_MAX_QUERY_LENGTH = 2000  # characters

# ── Module-level lazy-initialised singletons ─────────────────────────────────
# Both are built once per process and reused across calls to run_rag_chain.
# FIX 5: Locks ensure thread-safe double-checked initialisation so that
# concurrent Streamlit sessions cannot trigger duplicate builds.

_llm: Optional[BaseChatModel] = None
_retriever: Optional[ContextualCompressionRetriever] = None
_llm_lock = threading.Lock()
_retriever_lock = threading.Lock()


# ── Internal helpers ─────────────────────────────────────────────────────────


def _load_llm() -> BaseChatModel:
    """Instantiate and return the configured chat LLM.

    Selection order:
    1. ``ChatOpenAI``               – when ``OPENAI_API_KEY`` is non-empty.
    2. ``ChatGoogleGenerativeAI``   – when ``GOOGLE_API_KEY`` is non-empty.

    Returns:
        A :class:`~langchain_core.language_models.chat_models.BaseChatModel`.

    Raises:
        RuntimeError: If neither API key is configured in the environment.
    """
    if settings.openai_api_key:
        from langchain_openai import ChatOpenAI

        logger.info("LLM: ChatOpenAI (model=%s)", settings.openai_model)
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.0,
        )

    if settings.google_api_key:
        from langchain_google_genai import ChatGoogleGenerativeAI

        logger.info("LLM: ChatGoogleGenerativeAI (model=%s)", settings.google_model)
        return ChatGoogleGenerativeAI(
            model=settings.google_model,
            google_api_key=settings.google_api_key,
            temperature=0.0,
        )

    raise RuntimeError(
        "No LLM API key found. "
        "Set OPENAI_API_KEY or GOOGLE_API_KEY in your .env file."
    )


def _get_llm() -> BaseChatModel:
    """Return the cached LLM, initialising it on first call (thread-safe).

    Uses double-checked locking: the outer ``if`` avoids acquiring the lock on
    every hot-path call; the inner ``if`` (inside the lock) prevents duplicate
    initialisation when two threads race on the first call.
    """
    global _llm
    if _llm is None:
        with _llm_lock:
            if _llm is None:  # re-check after acquiring the lock
                _llm = _load_llm()
    return _llm


def _get_retriever() -> ContextualCompressionRetriever:
    """Return the cached hybrid retriever, building it on first call (thread-safe).

    Loads the persisted FAISS index, rebuilds the BM25 retriever in memory,
    then assembles the two-stage ``ContextualCompressionRetriever``.

    Uses double-checked locking — safe for concurrent Streamlit sessions.

    Raises:
        FileNotFoundError: If the FAISS index has not been built yet.
    """
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:  # re-check after acquiring the lock
                logger.info("Initialising retrieval pipeline...")
                vectorstore = load_faiss_vectorstore()
                chunks = split_documents(load_documents(str(settings.data_dir)))
                bm25 = build_bm25_retriever(chunks)
                _retriever = get_hybrid_reranked_retriever(vectorstore, bm25)
                logger.info("Retrieval pipeline ready.")
    return _retriever


def _format_context(docs: List[Document]) -> str:
    """Render retrieved documents into a single context string for the prompt.

    Each chunk is prefixed with its source filename so the model can reference
    provenance when answering.

    Args:
        docs: Reranked :class:`~langchain_core.documents.Document` objects.

    Returns:
        A newline-separated context string.
    """
    parts = []
    for doc in docs:
        filename = doc.metadata.get("filename", "Unknown")
        parts.append(f"[{filename}]\n{doc.page_content.strip()}")
    return "\n\n---\n\n".join(parts)


def _format_sources(docs: List[Document]) -> List[Dict[str, str]]:
    """Build a deduplicated list of source citations from chunk metadata.

    For PDF chunks the ``page`` field (0-indexed) is converted to a 1-indexed
    page number.  For DOCX/TXT chunks the ``chunk_index`` is used as a section
    reference.

    Args:
        docs: Reranked :class:`~langchain_core.documents.Document` objects.

    Returns:
        Ordered list of unique dicts with keys ``filename``, ``reference``,
        and ``file_type``.
    """
    seen: set = set()
    sources: List[Dict[str, str]] = []

    for doc in docs:
        filename = doc.metadata.get("filename", "Unknown")
        file_type = doc.metadata.get("file_type", "?")
        page = doc.metadata.get("page")  # present only for PDFs

        if page is not None:
            reference = f"Page {int(page) + 1}"
            dedup_key = (filename, int(page))
        else:
            chunk_idx = doc.metadata.get("chunk_index", 0)
            reference = f"Section {int(chunk_idx) + 1}"
            dedup_key = (filename, chunk_idx)

        if dedup_key not in seen:
            seen.add(dedup_key)
            sources.append(
                {"filename": filename, "reference": reference, "file_type": file_type}
            )

    return sources


def _build_messages(
    query: str,
    context: str,
    memory: ConversationBufferWindowMemory,
) -> list:
    """Assemble the full message list to send to the LLM.

    Structure:
    1. ``SystemMessage`` – strict hallucination guardrails.
    2. Conversation history from *memory* (up to k=5 prior turns).
    3. ``HumanMessage``  – retrieved context + current question.

    Args:
        query:   The user's current question.
        context: Pre-formatted context string from :func:`_format_context`.
        memory:  :class:`~langchain_classic.memory.ConversationBufferWindowMemory`
                 holding prior turns.

    Returns:
        List of :class:`~langchain_core.messages.BaseMessage` objects.
    """
    messages = [SystemMessage(content=_SYSTEM_PROMPT)]

    # Inject rolling conversation history
    history_msgs = memory.chat_memory.messages if memory else []
    messages.extend(history_msgs)

    # Current turn: context + question
    user_turn = (
        f"Use the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )
    messages.append(HumanMessage(content=user_turn))
    return messages


# ── Public API ────────────────────────────────────────────────────────────────


def build_memory(k: int = 5) -> ConversationBufferWindowMemory:
    """Create a fresh :class:`~langchain_classic.memory.ConversationBufferWindowMemory`.

    Args:
        k: Number of conversation turns to retain (default 5).

    Returns:
        An initialised memory object ready to pass to :func:`run_rag_chain`.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return ConversationBufferWindowMemory(
            k=k,
            return_messages=True,
            memory_key="history",
        )


def run_rag_chain(
    query: str,
    history: ConversationBufferWindowMemory,
) -> Tuple[str, List[Dict[str, str]]]:
    """Run the full conversational RAG pipeline for a single user turn.

    Steps performed:

    1. Validate *query* (non-empty, within length limit).
    2. Retrieve top-k grounded context chunks via the hybrid reranking pipeline.
    3. Build the LLM message list (system prompt + conversation history + context).
    4. Invoke the configured LLM (``ChatOpenAI`` or ``ChatGoogleGenerativeAI``).
    5. Append a ``Sources Used`` citation block to the answer.
    6. Persist the turn to *history*.

    Args:
        query:   The user's question string (must be non-empty, ≤ 2 000 chars).
        history: A :class:`~langchain_classic.memory.ConversationBufferWindowMemory`
                 instance created by :func:`build_memory`.  Updated in-place.

    Returns:
        A two-tuple ``(answer, sources)`` where:

        - ``answer``  – The grounded response string with appended citation block.
        - ``sources`` – Deduplicated list of source dicts
                        ``{"filename": ..., "reference": ..., "file_type": ...}``.

    Raises:
        ValueError:   If *query* is empty, whitespace-only, or exceeds
                      ``_MAX_QUERY_LENGTH`` characters.
        RuntimeError: If no LLM API key is configured.
        FileNotFoundError: If the FAISS index has not been built yet.
    """
    # ── FIX 8: Input validation ────────────────────────────────────────────
    if not query or not query.strip():
        raise ValueError("run_rag_chain() requires a non-empty query string.")
    if len(query) > _MAX_QUERY_LENGTH:
        raise ValueError(
            f"Query is too long ({len(query):,} characters). "
            f"Maximum allowed is {_MAX_QUERY_LENGTH:,} characters."
        )

    logger.info("RAG chain invoked | query: %r", query)

    # ── Step 1: retrieve context ───────────────────────────────────────────
    try:
        docs = retrieve(query, _get_retriever())
    except Exception as exc:
        logger.error("Retrieval error for query %r: %s", query, exc, exc_info=True)
        raise

    context = _format_context(docs)
    sources = _format_sources(docs)
    logger.info("Retrieved %d chunk(s) from %d unique source(s).", len(docs), len(sources))

    # ── Step 2: build prompt & call LLM ───────────────────────────────────
    messages = _build_messages(query, context, history)

    try:
        llm = _get_llm()
        response = llm.invoke(messages)
        answer: str = (
            response.content if hasattr(response, "content") else str(response)
        )
    except Exception as exc:
        logger.error("LLM inference error for query %r: %s", query, exc, exc_info=True)
        raise

    # ── Step 3: append source citation block ──────────────────────────────
    if sources:
        citation_parts = [f"{s['filename']} ({s['reference']})" for s in sources]
        answer = answer.rstrip() + (
            f"\n\n**Sources Used:** {', '.join(citation_parts)}"
        )

    logger.info("RAG chain completed | answer length=%d chars", len(answer))

    # ── Step 4: persist turn to memory ────────────────────────────────────
    history.save_context({"input": query}, {"output": answer})

    return answer, sources


def stream_rag_chain(
    query: str,
    history: ConversationBufferWindowMemory,
):
    """Streaming variant of :func:`run_rag_chain`.

    Performs retrieval and context building synchronously, then yields LLM
    response tokens one at a time for use with ``st.write_stream``.
    After the generator is exhausted, the completed turn is saved to *history*.

    The caller is responsible for collecting the yielded tokens into a full
    string and calling ``history.save_context`` if needed, but this function
    handles memory saving internally after all tokens are yielded.

    Args:
        query:   The user's question string (must be non-empty, ≤ 2 000 chars).
        history: Memory instance created by :func:`build_memory`.

    Yields:
        ``str`` – individual LLM response tokens.

    Returns:
        A two-tuple ``(full_answer, sources)`` accessible via ``StopIteration.value``
        or collected externally.

    Raises:
        ValueError:   If *query* is empty, whitespace-only, or exceeds
                      ``_MAX_QUERY_LENGTH`` characters.
        RuntimeError: If no LLM API key is configured or streaming unsupported.
        FileNotFoundError: If the FAISS index is missing.
    """
    # ── FIX 8: Input validation ────────────────────────────────────────────
    if not query or not query.strip():
        raise ValueError("stream_rag_chain() requires a non-empty query string.")
    if len(query) > _MAX_QUERY_LENGTH:
        raise ValueError(
            f"Query is too long ({len(query):,} characters). "
            f"Maximum allowed is {_MAX_QUERY_LENGTH:,} characters."
        )

    logger.info("Streaming RAG chain | query: %r", query)

    docs = retrieve(query, _get_retriever())
    context = _format_context(docs)
    sources = _format_sources(docs)
    messages = _build_messages(query, context, history)

    llm = _get_llm()
    full_answer = ""

    try:
        for chunk in llm.stream(messages):
            token: str = chunk.content if hasattr(chunk, "content") else str(chunk)
            full_answer += token
            yield token
    except Exception as exc:
        logger.error("Streaming LLM error for query %r: %s", query, exc, exc_info=True)
        raise

    # Append citations to the persisted answer (not yielded to keep stream clean)
    persisted_answer = full_answer.rstrip()
    if sources:
        citation_parts = [f"{s['filename']} ({s['reference']})" for s in sources]
        persisted_answer += f"\n\n**Sources Used:** {', '.join(citation_parts)}"

    history.save_context({"input": query}, {"output": persisted_answer})
    logger.info("Streaming RAG chain completed | %d chars, %d source(s)", len(full_answer), len(sources))

    return full_answer, sources
