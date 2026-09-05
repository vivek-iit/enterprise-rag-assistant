"""Unit tests for src/vectorstore.py and src/retrieval.py — hybrid search pipeline.

Changes from original
---------------------
- Added ``TestCrossEncoderConfig`` class to verify that
  ``get_hybrid_reranked_retriever`` reads the model name from
  ``settings.cross_encoder_model`` (Fix 6) rather than a hardcoded string.
"""

import pytest
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from unittest.mock import patch, MagicMock

from src.config import settings
from src.retrieval import retrieve
from src.vectorstore import _validate_index_dir

from pathlib import Path


# ── FAISS vectorstore ─────────────────────────────────────────────────────────

class TestFAISSVectorstore:

    def test_loads_as_faiss_instance(self, faiss_vs):
        assert isinstance(faiss_vs, FAISS)

    def test_similarity_search_returns_list(self, faiss_vs):
        results = faiss_vs.similarity_search("annual leave", k=3)
        assert isinstance(results, list)

    def test_similarity_search_returns_documents(self, faiss_vs):
        results = faiss_vs.similarity_search("annual leave", k=3)
        assert all(isinstance(d, Document) for d in results)

    def test_similarity_search_k_respected(self, faiss_vs):
        k = 3
        results = faiss_vs.similarity_search("health insurance", k=k)
        assert len(results) <= k

    def test_leave_policy_retrieved_for_leave_query(self, faiss_vs):
        results = faiss_vs.similarity_search("annual leave carry forward", k=4)
        filenames = [d.metadata.get("filename", "") for d in results]
        assert any("Leave_Policy" in f for f in filenames)

    def test_security_policy_retrieved_for_password_query(self, faiss_vs):
        results = faiss_vs.similarity_search("password expiry MFA", k=4)
        filenames = [d.metadata.get("filename", "") for d in results]
        assert any("IT_Security" in f for f in filenames)

    def test_benefits_retrieved_for_wellness_query(self, faiss_vs):
        results = faiss_vs.similarity_search("wellness allowance gym membership", k=4)
        filenames = [d.metadata.get("filename", "") for d in results]
        assert any("Employee_Benefits" in f for f in filenames)

    def test_missing_index_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            _validate_index_dir(Path("data/nonexistent_index"))


# ── BM25 retriever ────────────────────────────────────────────────────────────

class TestBM25Retriever:

    def test_is_bm25_retriever_instance(self, bm25):
        assert isinstance(bm25, BM25Retriever)

    def test_k_equals_ten(self, bm25):
        assert bm25.k == 10

    def test_invoke_returns_list(self, bm25):
        results = bm25.invoke("sick leave policy")
        assert isinstance(results, list)

    def test_invoke_returns_documents(self, bm25):
        results = bm25.invoke("sick leave policy")
        assert all(isinstance(d, Document) for d in results)

    def test_password_policy_retrieved(self, bm25):
        results = bm25.invoke("password expiry reuse MFA")
        filenames = [d.metadata.get("filename", "") for d in results]
        assert any("IT_Security" in f for f in filenames)

    def test_wellness_allowance_retrieved(self, bm25):
        results = bm25.invoke("wellness allowance $500")
        filenames = [d.metadata.get("filename", "") for d in results]
        assert any("Employee_Benefits" in f for f in filenames)

    def test_empty_chunks_raises_value_error(self):
        from src.vectorstore import build_bm25_retriever
        with pytest.raises(ValueError):
            build_bm25_retriever([])


# ── Hybrid reranking retriever ────────────────────────────────────────────────

class TestHybridReranker:

    def test_is_contextual_compression_retriever(self, hybrid_retriever):
        assert isinstance(hybrid_retriever, ContextualCompressionRetriever)

    def test_retrieve_returns_list(self, hybrid_retriever):
        results = retrieve("how many days annual leave", hybrid_retriever)
        assert isinstance(results, list)

    def test_retrieve_returns_documents(self, hybrid_retriever):
        results = retrieve("VPN usage remote access", hybrid_retriever)
        assert all(isinstance(d, Document) for d in results)

    def test_retrieve_bounded_by_top_k(self, hybrid_retriever):
        results = retrieve("employee health insurance coverage", hybrid_retriever)
        assert len(results) <= settings.top_k

    def test_leave_query_returns_leave_policy_docs(self, hybrid_retriever):
        results = retrieve("annual leave entitlement carry forward", hybrid_retriever)
        filenames = [d.metadata.get("filename", "") for d in results]
        assert any("Leave_Policy" in f for f in filenames)

    def test_security_query_returns_it_policy_docs(self, hybrid_retriever):
        results = retrieve("password policy multi-factor authentication", hybrid_retriever)
        filenames = [d.metadata.get("filename", "") for d in results]
        assert any("IT_Security" in f for f in filenames)

    def test_benefits_query_returns_benefits_docs(self, hybrid_retriever):
        results = retrieve("wellness allowance reimbursement", hybrid_retriever)
        filenames = [d.metadata.get("filename", "") for d in results]
        assert any("Employee_Benefits" in f for f in filenames)

    def test_empty_query_raises_value_error(self, hybrid_retriever):
        with pytest.raises(ValueError):
            retrieve("", hybrid_retriever)

    def test_whitespace_query_raises_value_error(self, hybrid_retriever):
        with pytest.raises(ValueError):
            retrieve("   ", hybrid_retriever)

    def test_retrieved_documents_have_metadata(self, hybrid_retriever):
        results = retrieve("leave policy", hybrid_retriever)
        for doc in results:
            assert "filename" in doc.metadata
            assert "file_type" in doc.metadata


# ── CrossEncoder model configurability (Fix 6) ───────────────────────────────

class TestCrossEncoderConfig:
    """Verify that get_hybrid_reranked_retriever reads the model name from
    ``settings.cross_encoder_model`` instead of a hardcoded string.

    This test monkey-patches ``settings.cross_encoder_model`` and confirms
    that the patched name is passed to ``HuggingFaceCrossEncoder``, proving
    the model is configurable without code changes.
    """

    @patch("src.retrieval.ContextualCompressionRetriever")
    @patch("src.retrieval.CrossEncoderReranker")
    @patch("src.retrieval.HuggingFaceCrossEncoder")
    def test_cross_encoder_model_name_comes_from_settings(
        self, mock_hf_cross_encoder, mock_reranker, mock_compression_retriever, faiss_vs, bm25
    ):
        """Confirm HuggingFaceCrossEncoder is called with settings.cross_encoder_model.

        CrossEncoderReranker and ContextualCompressionRetriever are also mocked
        to prevent pydantic strict type-validation from rejecting the MagicMock
        instances returned by the patched constructors.
        """
        from src.retrieval import get_hybrid_reranked_retriever

        custom_model = "cross-encoder/ms-marco-MiniLM-L-2-v2"
        mock_hf_cross_encoder.return_value = MagicMock()
        mock_reranker.return_value = MagicMock()
        mock_compression_retriever.return_value = MagicMock()

        with patch.object(settings, "cross_encoder_model", custom_model):
            get_hybrid_reranked_retriever(faiss_vs, bm25)

        called_with = mock_hf_cross_encoder.call_args
        assert called_with is not None, "HuggingFaceCrossEncoder was never called"
        assert called_with.kwargs.get("model_name") == custom_model or (
            called_with.args and called_with.args[0] == custom_model
        ), (
            f"Expected model_name='{custom_model}', "
            f"got call args: {called_with}"
        )

    def test_default_cross_encoder_model_is_set_in_settings(self):
        """settings.cross_encoder_model must be non-empty (not hardcoded in retrieval.py)."""
        assert settings.cross_encoder_model
        assert "cross-encoder" in settings.cross_encoder_model
