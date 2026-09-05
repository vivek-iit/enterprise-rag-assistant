"""Hybrid search indexing: FAISS dense vector store and BM25 keyword retriever.

FAISS index
-----------
- Embeddings : ``HuggingFaceEmbeddings`` (model from :attr:`src.config.settings.hf_embedding_model`)
- Backend    : ``faiss-cpu``
- Persistence: saved/loaded from ``settings.faiss_index_dir`` (default ``data/faiss_index``)

BM25 retriever
--------------
- Backend : ``rank_bm25`` via :class:`~langchain_community.retrievers.BM25Retriever`
- ``k``   : 10 results

Public API
----------
- :func:`build_vectorstore`     – embed chunks → FAISS, persist to disk, return store
- :func:`save_faiss_vectorstore` – write an existing FAISS store to disk
- :func:`load_faiss_vectorstore` – reload a persisted FAISS store from disk
- :func:`build_bm25_retriever`  – build an in-memory BM25 retriever from chunks
"""

from pathlib import Path
from typing import List

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)

_REQUIRED_FAISS_FILES: tuple = ("index.faiss", "index.pkl")


# ── internal helpers ──────────────────────────────────────────────────────────


def _get_embeddings() -> HuggingFaceEmbeddings:
    """Return a :class:`HuggingFaceEmbeddings` instance using the configured model.

    The model name is read from :attr:`src.config.settings.hf_embedding_model`
    (default: ``sentence-transformers/all-MiniLM-L6-v2``).
    """
    logger.info("Loading embedding model: %s", settings.hf_embedding_model)
    return HuggingFaceEmbeddings(model_name=settings.hf_embedding_model)


def _validate_index_dir(index_dir: Path) -> None:
    """Raise :exc:`FileNotFoundError` if the FAISS index directory or its files are missing.

    Args:
        index_dir: Resolved path to the FAISS index directory.

    Raises:
        FileNotFoundError: If the directory does not exist or required index files
            (``index.faiss``, ``index.pkl``) are absent / zero-byte.
    """
    if not index_dir.exists():
        raise FileNotFoundError(
            f"FAISS index directory not found: '{index_dir}'. "
            "Run build_vectorstore() first to create and persist the index."
        )
    missing = [f for f in _REQUIRED_FAISS_FILES if not (index_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"FAISS index in '{index_dir}' is incomplete. "
            f"Missing files: {missing}. "
            "Delete the directory and re-run build_vectorstore() to rebuild."
        )
    corrupted = [
        f for f in _REQUIRED_FAISS_FILES if (index_dir / f).stat().st_size == 0
    ]
    if corrupted:
        raise FileNotFoundError(
            f"FAISS index file(s) appear corrupted (zero bytes): {corrupted}. "
            "Delete the directory and re-run build_vectorstore() to rebuild."
        )


# ── public API ────────────────────────────────────────────────────────────────


def build_vectorstore(chunks: List[Document]) -> FAISS:
    """Embed *chunks* and build a FAISS vectorstore, then persist it to disk.

    The FAISS index is saved to the directory specified by
    :attr:`src.config.settings.faiss_index_dir` (default ``data/faiss_index``).
    Calling this function a second time will overwrite the existing index.

    Args:
        chunks: Non-empty list of :class:`~langchain_core.documents.Document`
            objects produced by :func:`src.ingestion.split_documents`.

    Returns:
        The in-memory :class:`~langchain_community.vectorstores.FAISS` instance.

    Raises:
        ValueError: If *chunks* is empty.
    """
    if not chunks:
        raise ValueError(
            "build_vectorstore() received an empty chunk list. "
            "Run ingestion.split_documents() first."
        )

    logger.info(
        "Building FAISS vectorstore from %d chunks using '%s'...",
        len(chunks),
        settings.hf_embedding_model,
    )
    embeddings = _get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    save_faiss_vectorstore(vectorstore)
    logger.info("FAISS vectorstore built and saved successfully.")
    return vectorstore


def save_faiss_vectorstore(vectorstore: FAISS) -> None:
    """Persist an in-memory FAISS vectorstore to :attr:`src.config.settings.faiss_index_dir`.

    Creates the target directory (and any parents) if it does not already exist.
    Overwrites any previously saved index at the same location.

    Args:
        vectorstore: A :class:`~langchain_community.vectorstores.FAISS` instance
            to persist.
    """
    index_dir = Path(settings.faiss_index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    logger.info("FAISS index saved to '%s'.", index_dir.resolve())


def load_faiss_vectorstore() -> FAISS:
    """Load a previously persisted FAISS vectorstore from disk.

    Reads from :attr:`src.config.settings.faiss_index_dir` (default
    ``data/faiss_index``).  The same embedding model used during
    :func:`build_vectorstore` must be active (controlled by
    :attr:`src.config.settings.hf_embedding_model`).

    Returns:
        The rehydrated :class:`~langchain_community.vectorstores.FAISS` instance.

    Raises:
        FileNotFoundError: If the index directory is missing, the required files
            are absent, or any index file is zero bytes (corrupted).
    """
    index_dir = Path(settings.faiss_index_dir).resolve()
    _validate_index_dir(index_dir)

    embeddings = _get_embeddings()
    # FIX 7: allow_dangerous_deserialization is required by FAISS because it
    # stores the index metadata as a pickle file (index.pkl).  This is safe
    # here because the index is a locally-generated, trusted artefact written
    # by build_vectorstore() in the same process.
    # WARNING: Never load an index received from an untrusted external source.
    vectorstore = FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    logger.info("FAISS index loaded from '%s'.", index_dir)
    return vectorstore


def build_bm25_retriever(chunks: List[Document]) -> BM25Retriever:
    """Build an in-memory BM25 keyword retriever from *chunks*.

    Uses :class:`~langchain_community.retrievers.BM25Retriever` backed by
    ``rank_bm25``.  The retriever is stateless (not persisted) and must be
    rebuilt from chunks on each application start.

    Args:
        chunks: Non-empty list of :class:`~langchain_core.documents.Document`
            objects to index for keyword search.

    Returns:
        A :class:`~langchain_community.retrievers.BM25Retriever` with ``k=10``.

    Raises:
        ValueError: If *chunks* is empty.
    """
    if not chunks:
        raise ValueError(
            "build_bm25_retriever() received an empty chunk list. "
            "Run ingestion.split_documents() first."
        )

    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = 10
    logger.info(
        "BM25 retriever built from %d chunks, k=%d.",
        len(chunks),
        retriever.k,
    )
    return retriever
