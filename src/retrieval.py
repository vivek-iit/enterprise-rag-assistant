"""Two-stage hybrid retrieval with CrossEncoder reranking.

Pipeline overview
-----------------
Stage 1 — Hybrid retrieval (breadth)
    :class:`~langchain_classic.retrievers.EnsembleRetriever` combines:

    - **Dense**  : FAISS similarity search (``k=10``)
    - **Sparse** : BM25 keyword search (``k=10``)

    with equal weights ``[0.5, 0.5]``, producing a deduplicated candidate pool.

Stage 2 — CrossEncoder reranking (precision)
    :class:`~langchain_classic.retrievers.document_compressors.CrossEncoderReranker`
    backed by ``HuggingFaceCrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")``
    re-scores every candidate against the query and retains the top ``k`` results
    (default ``settings.top_k = 4``).

The two stages are composed into a single
:class:`~langchain_classic.retrievers.ContextualCompressionRetriever` that is
returned to callers.

Public API
----------
- :func:`get_hybrid_reranked_retriever` – build and return the full pipeline
- :func:`retrieve`                       – run a query with timing + logging
"""

import time
from typing import List

from langchain_classic.retrievers import (
    ContextualCompressionRetriever,
    EnsembleRetriever,
)
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)

# ── public API ────────────────────────────────────────────────────────────────


def get_hybrid_reranked_retriever(
    faiss_vectorstore: FAISS,
    bm25_retriever: BM25Retriever,
) -> ContextualCompressionRetriever:
    """Build a two-stage hybrid + CrossEncoder retrieval pipeline.

    Stage 1 creates an :class:`EnsembleRetriever` that merges dense FAISS
    results and sparse BM25 results with equal weighting.  Stage 2 wraps
    the ensemble in a :class:`ContextualCompressionRetriever` that applies a
    CrossEncoder reranker to select the ``settings.top_k`` most relevant
    documents.

    The CrossEncoder model name is read from
    :attr:`src.config.settings.cross_encoder_model` (default
    ``cross-encoder/ms-marco-MiniLM-L-6-v2``) so it can be overridden via
    the ``CROSS_ENCODER_MODEL`` environment variable without code changes.

    Args:
        faiss_vectorstore: A loaded or freshly built
            :class:`~langchain_community.vectorstores.FAISS` instance.
        bm25_retriever: A :class:`~langchain_community.retrievers.BM25Retriever`
            instance (``k=10`` is expected but not enforced).

    Returns:
        A :class:`~langchain_classic.retrievers.ContextualCompressionRetriever`
        ready to accept ``invoke(query)`` calls.

    Raises:
        ValueError: If either *faiss_vectorstore* or *bm25_retriever* is ``None``.
        RuntimeError: If loading the CrossEncoder model fails (e.g. no internet
            on first run, model name typo).
    """
    if faiss_vectorstore is None:
        raise ValueError("faiss_vectorstore must not be None.")
    if bm25_retriever is None:
        raise ValueError("bm25_retriever must not be None.")

    # ── Stage 1: EnsembleRetriever ─────────────────────────────────────────
    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 10})

    ensemble_retriever = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )
    logger.info(
        "EnsembleRetriever created (FAISS k=10, BM25 k=%d, weights=[0.5, 0.5]).",
        bm25_retriever.k,
    )

    # ── Stage 2: CrossEncoder reranker ────────────────────────────────────
    # FIX 6: model name sourced from settings.cross_encoder_model (previously
    # hardcoded); override via CROSS_ENCODER_MODEL in .env if needed.
    cross_encoder_model_name = settings.cross_encoder_model
    logger.info("Loading CrossEncoder model: '%s' ...", cross_encoder_model_name)
    try:
        cross_encoder = HuggingFaceCrossEncoder(model_name=cross_encoder_model_name)
    except Exception as exc:
        logger.error(
            "Failed to load CrossEncoder model '%s': %s",
            cross_encoder_model_name,
            exc,
            exc_info=True,
        )
        raise RuntimeError(
            f"CrossEncoder model '{cross_encoder_model_name}' could not be loaded. "
            "Check your internet connection or the model name."
        ) from exc

    reranker = CrossEncoderReranker(model=cross_encoder, top_n=settings.top_k)
    logger.info(
        "CrossEncoderReranker ready (model='%s', top_n=%d).",
        cross_encoder_model_name,
        settings.top_k,
    )

    # ── Compose pipeline ───────────────────────────────────────────────────
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=ensemble_retriever,
    )
    logger.info(
        "ContextualCompressionRetriever pipeline assembled "
        "(ensemble → cross-encoder rerank, final top_k=%d).",
        settings.top_k,
    )
    return compression_retriever


def retrieve(
    query: str,
    retriever: ContextualCompressionRetriever,
) -> List[Document]:
    """Execute a retrieval query and return the reranked top-k documents.

    Logs the query, per-stage timing, and the number of documents returned.
    Any exception raised by the underlying retriever is logged at ERROR level
    before being re-raised so callers can handle or surface it.

    Args:
        query: The natural-language question or search string.
        retriever: A :class:`~langchain_classic.retrievers.ContextualCompressionRetriever`
            built by :func:`get_hybrid_reranked_retriever`.

    Returns:
        Ordered list of :class:`~langchain_core.documents.Document` objects,
        most relevant first, length ≤ ``settings.top_k``.

    Raises:
        ValueError: If *query* is empty or whitespace-only.
        Exception: Re-raises any exception thrown by the retriever.
    """
    if not query or not query.strip():
        raise ValueError("retrieve() requires a non-empty query string.")

    logger.info("Retrieval query: %r", query)
    t_start = time.perf_counter()

    try:
        docs = retriever.invoke(query)
    except Exception as exc:
        logger.error(
            "Retrieval failed for query %r after %.3fs: %s",
            query,
            time.perf_counter() - t_start,
            exc,
            exc_info=True,
        )
        raise

    elapsed = time.perf_counter() - t_start
    logger.info(
        "Retrieved %d document(s) in %.3fs for query: %r",
        len(docs),
        elapsed,
        query,
    )
    return docs
