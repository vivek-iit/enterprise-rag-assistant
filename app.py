"""Enterprise Knowledge Assistant — Streamlit UI.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict

import streamlit as st

# ── page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── project imports ───────────────────────────────────────────────────────────
from src.config import settings
from src.logger import get_logger
import src.chain as _chain_module
from src.chain import build_memory, run_rag_chain, stream_rag_chain
from src.ingestion import load_documents, split_documents
from src.vectorstore import build_vectorstore, build_bm25_retriever

logger = get_logger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────
_FAISS_INDEX_DIR = Path(settings.faiss_index_dir)
_DATA_DIR = Path(settings.data_dir)
_SUPPORTED_EXTS = ("*.pdf", "*.txt", "*.docx")

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Header gradient */
    .app-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .app-header h1 { color: #e2e2e2; margin: 0; font-size: 1.9rem; }
    .app-header p  { color: #90afc5; margin: 0.3rem 0 0 0; font-size: 0.85rem; }

    /* Sidebar styling */
    section[data-testid="stSidebar"] { background: #0f1117; }
    section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

    /* Source citation expander */
    details { border: 1px solid #2d3748; border-radius: 8px; padding: 4px 8px; }
    details summary { font-weight: 600; color: #63b3ed; cursor: pointer; }

    /* Chat bubbles */
    .stChatMessage { border-radius: 12px; }

    /* Status badges */
    .status-ok   { background:#1a4731; color:#6ee7b7; padding:4px 10px; border-radius:20px; font-size:0.8rem; }
    .status-warn { background:#44300e; color:#fbbf24; padding:4px 10px; border-radius:20px; font-size:0.8rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── helper functions ──────────────────────────────────────────────────────────

def _count_documents() -> int:
    """Return the number of supported documents in the data directory."""
    if not _DATA_DIR.exists():
        return 0
    return sum(len(list(_DATA_DIR.glob(ext))) for ext in _SUPPORTED_EXTS)


def _index_exists() -> bool:
    """Return True if a valid FAISS index is present on disk."""
    return (
        (_FAISS_INDEX_DIR / "index.faiss").exists()
        and (_FAISS_INDEX_DIR / "index.pkl").exists()
    )


def _llm_configured() -> bool:
    """Return True if at least one LLM API key is available."""
    return bool(settings.openai_api_key or settings.google_api_key)


def _llm_provider() -> str:
    """Return a display string identifying the active LLM provider."""
    if settings.openai_api_key:
        return f"OpenAI · {settings.openai_model}"
    if settings.google_api_key:
        return f"Google · {settings.google_model}"
    return "None"


def _strip_sources_block(text: str) -> str:
    """Remove the trailing '**Sources Used:**' block from an answer string."""
    return re.sub(r"\n\n\*\*Sources Used:\*\*.*$", "", text, flags=re.DOTALL).rstrip()


def _rebuild_index() -> tuple[bool, str]:
    """Run full ingestion + vectorstore rebuild.

    Returns:
        (success: bool, message: str)
    """
    doc_count = _count_documents()
    if doc_count == 0:
        return False, (
            "No supported documents found in `data/sample_docs/`. "
            "Run `scripts/generate_sample_data.py` or add PDF/TXT/DOCX files."
        )
    try:
        docs = load_documents(str(_DATA_DIR))
        chunks = split_documents(docs)
        build_vectorstore(chunks)
        # Reset lazy retriever singleton so it reloads on next query
        _chain_module._retriever = None
        return True, (
            f"Index built from **{len(docs)}** document(s) → **{len(chunks)}** chunks."
        )
    except Exception as exc:
        logger.error("Re-indexing failed: %s", exc, exc_info=True)
        return False, f"Indexing error: {exc}"


# ── session state initialisation ─────────────────────────────────────────────

def _init_session() -> None:
    if "messages" not in st.session_state:
        # Each entry: {"role": str, "content": str, "sources": list[dict]}
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = build_memory(k=5)
    if "indexed" not in st.session_state:
        st.session_state.indexed = _index_exists()
    if "doc_count" not in st.session_state:
        st.session_state.doc_count = _count_documents()


_init_session()

# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.divider()

    # ── status panel ─────────────────────────────────────────────────────────
    doc_count = _count_documents()
    st.session_state.doc_count = doc_count

    if st.session_state.indexed:
        st.markdown(
            f'<span class="status-ok">✅ Index ready</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-warn">⚠️ Not indexed</span>',
            unsafe_allow_html=True,
        )

    st.markdown(f"**Documents found:** {doc_count}")
    st.markdown(f"**LLM:** {_llm_provider()}")
    st.markdown(
        f"**Embedding:** `{settings.hf_embedding_model.split('/')[-1]}`"
    )
    st.divider()

    # ── re-index button ───────────────────────────────────────────────────────
    if st.button("🔄 Re-index Documents", use_container_width=True, type="primary"):
        with st.spinner("Ingesting & indexing documents…"):
            ok, msg = _rebuild_index()
        if ok:
            st.session_state.indexed = True
            st.success(msg)
        else:
            st.error(msg)

    st.markdown("")

    # ── clear history button ──────────────────────────────────────────────────
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.memory = build_memory(k=5)
        st.rerun()

    st.divider()
    st.caption(
        "Enterprise RAG Assistant · IIT Patna\n\n"
        "Hybrid FAISS + BM25 + CrossEncoder pipeline.\n"
        "Responses are grounded strictly in indexed documents."
    )


# ── main header ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="app-header">
        <h1>🏢 Enterprise Knowledge Assistant</h1>
        <p>Hybrid RAG · FAISS + BM25 + CrossEncoder Reranking · Hallucination-guarded responses</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── error / info banners ──────────────────────────────────────────────────────

if not _llm_configured():
    st.error(
        "**No LLM API key configured.** "
        "Add `OPENAI_API_KEY` or `GOOGLE_API_KEY` to your `.env` file and restart the app.",
        icon="🔑",
    )

if doc_count == 0:
    st.warning(
        "**No documents found** in `data/sample_docs/`. "
        "Run `python scripts/generate_sample_data.py` to generate sample data, "
        "or copy your own PDF/TXT/DOCX files into that folder.",
        icon="📂",
    )
elif not st.session_state.indexed:
    st.info(
        "Documents are present but **not yet indexed**. "
        "Click **Re-index Documents** in the sidebar to build the search index.",
        icon="ℹ️",
    )

# ── chat history display ──────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🤖"):
        # Display clean answer text (without the appended sources block)
        clean = _strip_sources_block(msg["content"])
        st.markdown(clean)

        # Source citations in collapsible expander
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 View Source Citations"):
                for src in msg["sources"]:
                    icon = {"pdf": "📕", "docx": "📘", "txt": "📄"}.get(
                        src.get("file_type", ""), "📎"
                    )
                    st.markdown(
                        f"{icon} **{src['filename']}** — {src['reference']}"
                    )


# ── chat input & response loop ────────────────────────────────────────────────

prompt = st.chat_input(
    "Ask a question about company policies…",
    disabled=not (_llm_configured() and st.session_state.indexed),
)

if prompt:
    # ── render user message ───────────────────────────────────────────────────
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "sources": []}
    )
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    # ── generate & render assistant response ──────────────────────────────────
    with st.chat_message("assistant", avatar="🤖"):
        answer = ""
        sources: List[Dict] = []
        error_msg = ""

        try:
            # Use streaming for OpenAI; fallback to standard invoke for others
            if settings.openai_api_key:
                # st.write_stream collects and returns the full text automatically
                _sources_container: List[Dict] = []

                def _token_gen():
                    """Yield tokens and capture sources as a side-effect."""
                    gen = stream_rag_chain(prompt, st.session_state.memory)
                    try:
                        while True:
                            token = next(gen)
                            yield token
                    except StopIteration as si:
                        if si.value:
                            _, captured = si.value
                            _sources_container.extend(captured)

                answer = st.write_stream(_token_gen())
                sources = _sources_container

            else:
                # Standard (non-streaming) path for Google / other providers
                with st.spinner("Thinking…"):
                    answer, sources = run_rag_chain(
                        prompt, st.session_state.memory
                    )
                # Display the clean answer (sources shown in expander below)
                st.markdown(_strip_sources_block(answer))

        except ValueError as exc:
            # FIX 8: query length or empty-string validation failure
            error_msg = f"**Invalid input:** {exc}"
            st.error(error_msg, icon="✏️")

        except FileNotFoundError:
            error_msg = (
                "**Index not found.** Please click **Re-index Documents** "
                "in the sidebar before asking questions."
            )
            st.error(error_msg, icon="🗂️")

        except RuntimeError as exc:
            error_msg = f"**LLM error:** {exc}"
            st.error(error_msg, icon="🔑")

        except Exception as exc:
            logger.error("Unexpected app error: %s", exc, exc_info=True)
            error_msg = f"**Unexpected error:** {exc}"
            st.error(error_msg, icon="⚠️")

        # ── source citations expander ─────────────────────────────────────────
        if sources:
            with st.expander("📄 View Source Citations"):
                for src in sources:
                    icon = {"pdf": "📕", "docx": "📘", "txt": "📄"}.get(
                        src.get("file_type", ""), "📎"
                    )
                    st.markdown(
                        f"{icon} **{src['filename']}** — {src['reference']}"
                    )

    # ── persist to session state ──────────────────────────────────────────────
    stored_content = error_msg if error_msg else answer
    st.session_state.messages.append(
        {"role": "assistant", "content": stored_content, "sources": sources}
    )
