"""Unit tests for src/ingestion.py — document loading and chunking.

Changes from original
---------------------
- ``DATA_DIR`` is now an absolute path derived from ``__file__`` (Fix 9) so
  tests pass regardless of which directory pytest is invoked from.
"""

import pytest
from pathlib import Path
from langchain_core.documents import Document

from src.ingestion import load_documents, split_documents
from src.config import settings

# FIX 9: Absolute path — tests are now CWD-independent.
DATA_DIR = str(Path(__file__).parent.parent / "data" / "sample_docs")


# ── load_documents ────────────────────────────────────────────────────────────

class TestLoadDocuments:

    def test_returns_list(self, raw_docs):
        assert isinstance(raw_docs, list)

    def test_at_least_three_documents_loaded(self, raw_docs):
        assert len(raw_docs) >= 3

    def test_all_items_are_documents(self, raw_docs):
        assert all(isinstance(d, Document) for d in raw_docs)

    def test_pdf_files_loaded(self, raw_docs):
        pdf_docs = [d for d in raw_docs if d.metadata.get("file_type") == "pdf"]
        assert len(pdf_docs) >= 1, "Expected at least one PDF document"

    def test_txt_files_loaded(self, raw_docs):
        txt_docs = [d for d in raw_docs if d.metadata.get("file_type") == "txt"]
        assert len(txt_docs) >= 1, "Expected at least one TXT document"

    def test_docx_files_loaded(self, raw_docs):
        docx_docs = [d for d in raw_docs if d.metadata.get("file_type") == "docx"]
        assert len(docx_docs) >= 1, "Expected at least one DOCX document"

    def test_metadata_contains_filename(self, raw_docs):
        for doc in raw_docs:
            assert "filename" in doc.metadata, f"Missing 'filename' in {doc.metadata}"

    def test_metadata_contains_file_type(self, raw_docs):
        for doc in raw_docs:
            assert "file_type" in doc.metadata, f"Missing 'file_type' in {doc.metadata}"

    def test_metadata_contains_source(self, raw_docs):
        for doc in raw_docs:
            assert "source" in doc.metadata, f"Missing 'source' in {doc.metadata}"

    def test_pdf_metadata_contains_page(self, raw_docs):
        pdf_docs = [d for d in raw_docs if d.metadata.get("file_type") == "pdf"]
        for doc in pdf_docs:
            assert "page" in doc.metadata, "PDF documents must have 'page' in metadata"

    def test_documents_have_non_empty_content(self, raw_docs):
        for doc in raw_docs:
            assert doc.page_content.strip(), f"Empty content in {doc.metadata.get('filename')}"

    def test_filenames_match_expected_sample_files(self, raw_docs):
        filenames = {d.metadata.get("filename") for d in raw_docs}
        assert "Leave_Policy.pdf" in filenames
        assert "IT_Security_Policy.txt" in filenames
        assert "Employee_Benefits.docx" in filenames

    def test_missing_directory_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_documents("nonexistent/path/to/docs")

    def test_file_type_values_are_valid(self, raw_docs):
        valid_types = {"pdf", "docx", "txt"}
        for doc in raw_docs:
            assert doc.metadata["file_type"] in valid_types


# ── split_documents ───────────────────────────────────────────────────────────

class TestSplitDocuments:

    def test_returns_list(self, chunks):
        assert isinstance(chunks, list)

    def test_all_items_are_documents(self, chunks):
        assert all(isinstance(c, Document) for c in chunks)

    def test_chunk_count_exceeds_document_count(self, raw_docs, chunks):
        assert len(chunks) > len(raw_docs), (
            "Splitting should produce more items than the original document list"
        )

    def test_chunk_count_is_positive(self, chunks):
        assert len(chunks) > 0

    def test_chunk_content_does_not_exceed_size_plus_overlap(self, chunks):
        max_allowed = settings.chunk_size + settings.chunk_overlap
        oversized = [
            c for c in chunks if len(c.page_content) > max_allowed
        ]
        assert not oversized, (
            f"{len(oversized)} chunk(s) exceeded max size "
            f"({settings.chunk_size} + {settings.chunk_overlap})"
        )

    def test_chunks_inherit_filename_metadata(self, chunks):
        for chunk in chunks:
            assert "filename" in chunk.metadata

    def test_chunks_have_chunk_index(self, chunks):
        for chunk in chunks:
            assert "chunk_index" in chunk.metadata

    def test_chunks_have_start_index(self, chunks):
        for chunk in chunks:
            assert "start_index" in chunk.metadata

    def test_chunk_index_starts_at_zero_per_source(self, chunks):
        from collections import defaultdict
        first_seen = defaultdict(lambda: None)
        for chunk in chunks:
            src = chunk.metadata.get("source")
            idx = chunk.metadata.get("chunk_index")
            if first_seen[src] is None:
                first_seen[src] = idx
        for src, first_idx in first_seen.items():
            assert first_idx == 0, f"First chunk_index for {src} was {first_idx}, expected 0"

    def test_empty_input_returns_empty_list(self):
        result = split_documents([])
        assert result == []

    def test_all_file_types_represented_in_chunks(self, chunks):
        file_types = {c.metadata.get("file_type") for c in chunks}
        assert "pdf" in file_types
        assert "txt" in file_types
        assert "docx" in file_types
