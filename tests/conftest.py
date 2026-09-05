"""Shared pytest fixtures for the Enterprise RAG Assistant test suite."""

import sys
from pathlib import Path

import pytest

# Ensure the project root is on sys.path so `src.*` imports resolve
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

# FIX 9: Use an absolute path derived from this file's location so that tests
# pass regardless of which directory pytest is invoked from.
DATA_DIR = str(_PROJECT_ROOT / "data" / "sample_docs")


@pytest.fixture(scope="session")
def raw_docs():
    """Load all documents from the sample data directory (session-scoped)."""
    from src.ingestion import load_documents
    return load_documents(DATA_DIR)


@pytest.fixture(scope="session")
def chunks(raw_docs):
    """Split raw documents into chunks (session-scoped)."""
    from src.ingestion import split_documents
    return split_documents(raw_docs)


@pytest.fixture(scope="session")
def faiss_vs():
    """Load the persisted FAISS vectorstore (session-scoped)."""
    from src.vectorstore import load_faiss_vectorstore
    return load_faiss_vectorstore()


@pytest.fixture(scope="session")
def bm25(chunks):
    """Build a BM25 retriever from chunks (session-scoped)."""
    from src.vectorstore import build_bm25_retriever
    return build_bm25_retriever(chunks)


@pytest.fixture(scope="session")
def hybrid_retriever(faiss_vs, bm25):
    """Assemble the full hybrid + CrossEncoder retriever (session-scoped)."""
    from src.retrieval import get_hybrid_reranked_retriever
    return get_hybrid_reranked_retriever(faiss_vs, bm25)
