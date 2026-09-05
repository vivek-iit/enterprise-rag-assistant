"""Document ingestion module: loads and splits source documents for the RAG pipeline.

Supported file types
--------------------
- ``.pdf``  — via :class:`~langchain_community.document_loaders.PyPDFDirectoryLoader`
- ``.docx`` — via :class:`~langchain_community.document_loaders.DirectoryLoader`
              with :class:`~langchain_community.document_loaders.Docx2txtLoader`
- ``.txt``  — via :class:`~langchain_community.document_loaders.DirectoryLoader`
              with :class:`~langchain_community.document_loaders.TextLoader`

Metadata retained on every :class:`~langchain_core.documents.Document`
-----------------------------------------------------------------------
- ``source``      – absolute path of the originating file
- ``filename``    – basename of the source file
- ``file_type``   – ``"pdf"`` | ``"docx"`` | ``"txt"``
- ``page``        – 0-indexed page number (PDF only)
- ``start_index`` – character offset where the chunk starts within its source (chunks only)
- ``chunk_index`` – sequential index of the chunk within its source file (chunks only)
"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    DirectoryLoader,
    Docx2txtLoader,
    PyPDFDirectoryLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)


# ── public API ────────────────────────────────────────────────────────────────


def load_documents(data_dir: str) -> List[Document]:
    """Load all supported documents from *data_dir* and return them as a flat list.

    Each document's ``metadata`` dictionary is normalised to include at minimum:
    ``source`` (absolute path string), ``filename`` (basename), and ``file_type``.
    PDF documents additionally carry a ``page`` key (0-indexed).

    Args:
        data_dir: Path to the directory containing source documents.  May be
            relative to the current working directory or absolute.

    Returns:
        A list of :class:`~langchain_core.documents.Document` objects, one per
        page (PDF) or one per file (DOCX / TXT).

    Raises:
        FileNotFoundError: If *data_dir* does not exist on disk.
    """
    dir_path = Path(data_dir).resolve()
    if not dir_path.exists():
        raise FileNotFoundError(f"Data directory not found: {dir_path}")

    all_docs: List[Document] = []

    # ── PDF ───────────────────────────────────────────────────────────────────
    pdf_files = list(dir_path.glob("*.pdf"))
    if pdf_files:
        pdf_loader = PyPDFDirectoryLoader(str(dir_path))
        pdf_docs = pdf_loader.load()
        for doc in pdf_docs:
            src = Path(doc.metadata.get("source", ""))
            doc.metadata["source"] = str(src.resolve()) if src.exists() else str(src)
            doc.metadata["filename"] = src.name
            doc.metadata.setdefault("file_type", "pdf")
        all_docs.extend(pdf_docs)
        logger.info(
            "PDF: loaded %d page(s) from %d file(s).",
            len(pdf_docs),
            len(pdf_files),
        )
    else:
        logger.info("PDF: no .pdf files found in %s.", dir_path)

    # ── DOCX ──────────────────────────────────────────────────────────────────
    docx_files = list(dir_path.glob("*.docx"))
    if docx_files:
        docx_loader = DirectoryLoader(
            str(dir_path),
            glob="*.docx",
            loader_cls=Docx2txtLoader,
            show_progress=False,
            use_multithreading=False,
        )
        docx_docs = docx_loader.load()
        for doc in docx_docs:
            src = Path(doc.metadata.get("source", ""))
            doc.metadata["source"] = str(src.resolve()) if src.exists() else str(src)
            doc.metadata["filename"] = src.name
            doc.metadata["file_type"] = "docx"
        all_docs.extend(docx_docs)
        logger.info("DOCX: loaded %d document(s).", len(docx_docs))
    else:
        logger.info("DOCX: no .docx files found in %s.", dir_path)

    # ── TXT ───────────────────────────────────────────────────────────────────
    txt_files = list(dir_path.glob("*.txt"))
    if txt_files:
        txt_loader = DirectoryLoader(
            str(dir_path),
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=False,
            use_multithreading=False,
        )
        txt_docs = txt_loader.load()
        for doc in txt_docs:
            src = Path(doc.metadata.get("source", ""))
            doc.metadata["source"] = str(src.resolve()) if src.exists() else str(src)
            doc.metadata["filename"] = src.name
            doc.metadata["file_type"] = "txt"
        all_docs.extend(txt_docs)
        logger.info("TXT: loaded %d document(s).", len(txt_docs))
    else:
        logger.info("TXT: no .txt files found in %s.", dir_path)

    logger.info(
        "Total documents loaded: %d from '%s'.",
        len(all_docs),
        dir_path,
    )
    return all_docs


def split_documents(docs: List[Document]) -> List[Document]:
    """Split a list of documents into overlapping text chunks.

    Uses :class:`~langchain_text_splitters.RecursiveCharacterTextSplitter`
    with parameters drawn from :mod:`src.config`:

    - ``chunk_size``    (default 500 characters)
    - ``chunk_overlap`` (default 50 characters)

    Each chunk inherits all metadata from its parent document and is enriched
    with two additional fields:

    - ``start_index``  – character offset of the chunk within the source text
    - ``chunk_index``  – sequential 0-based index of the chunk per source file

    Args:
        docs: Documents returned by :func:`load_documents` (or any list of
            :class:`~langchain_core.documents.Document` objects).

    Returns:
        A flat list of chunked :class:`~langchain_core.documents.Document` objects
        ready to be embedded and indexed.
    """
    if not docs:
        logger.warning("split_documents received an empty document list; returning [].")
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        add_start_index=True,
    )

    chunks = splitter.split_documents(docs)

    # Annotate each chunk with its sequential index within its source file
    _counters: dict = {}
    for chunk in chunks:
        src_key = chunk.metadata.get("source", "unknown")
        _counters[src_key] = _counters.get(src_key, -1) + 1
        chunk.metadata["chunk_index"] = _counters[src_key]

    logger.info(
        "Split %d document(s) into %d chunk(s) "
        "(chunk_size=%d, chunk_overlap=%d).",
        len(docs),
        len(chunks),
        settings.chunk_size,
        settings.chunk_overlap,
    )
    return chunks
