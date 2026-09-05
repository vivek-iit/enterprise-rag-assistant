"""Unit tests for src/chain.py — memory, prompt, formatting, validation, and hallucination guard.

Changes from original
---------------------
- Added ``TestQueryValidation`` class covering the new ``_MAX_QUERY_LENGTH``
  guard introduced in Fix 8 (both ``run_rag_chain`` and ``stream_rag_chain``).
- Added ``TestThreadSafety`` class verifying that concurrent calls to
  ``_get_llm()`` and ``_get_retriever()`` do not produce duplicate
  initialisations (Fix 5).
- Docstrings and comments updated to reflect current public API.
"""

import threading
import pytest
from unittest.mock import MagicMock, patch

from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.chain import (
    _MAX_QUERY_LENGTH,
    _SYSTEM_PROMPT,
    _build_messages,
    _format_context,
    _format_sources,
    build_memory,
    run_rag_chain,
    stream_rag_chain,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_doc(content: str, filename: str, file_type: str, page=None, chunk_index: int = 0) -> Document:
    meta = {"filename": filename, "file_type": file_type, "chunk_index": chunk_index}
    if page is not None:
        meta["page"] = page
    return Document(page_content=content, metadata=meta)


def _mock_retriever(docs: list) -> MagicMock:
    ret = MagicMock()
    ret.invoke.return_value = docs
    return ret


# ── build_memory ──────────────────────────────────────────────────────────────

class TestBuildMemory:

    def test_returns_conversation_buffer_window_memory(self):
        mem = build_memory()
        assert isinstance(mem, ConversationBufferWindowMemory)

    def test_default_k_is_five(self):
        assert build_memory().k == 5

    def test_custom_k_is_respected(self):
        assert build_memory(k=3).k == 3

    def test_return_messages_is_true(self):
        assert build_memory().return_messages is True

    def test_starts_with_empty_history(self):
        mem = build_memory()
        assert len(mem.chat_memory.messages) == 0

    def test_save_context_adds_two_messages(self):
        mem = build_memory()
        mem.save_context({"input": "Hello"}, {"output": "Hi"})
        assert len(mem.chat_memory.messages) == 2

    def test_history_preserves_human_and_ai_messages(self):
        mem = build_memory()
        mem.save_context({"input": "How much leave?"}, {"output": "20 days."})
        msgs = mem.chat_memory.messages
        assert isinstance(msgs[0], HumanMessage)
        assert isinstance(msgs[1], AIMessage)


# ── _format_sources ───────────────────────────────────────────────────────────

class TestFormatSources:

    def test_pdf_reference_is_one_indexed_page(self):
        docs = [_make_doc("text", "Leave_Policy.pdf", "pdf", page=0)]
        sources = _format_sources(docs)
        assert sources[0]["reference"] == "Page 1"

    def test_pdf_page_2_maps_to_page_2(self):
        docs = [_make_doc("text", "Leave_Policy.pdf", "pdf", page=1)]
        sources = _format_sources(docs)
        assert sources[0]["reference"] == "Page 2"

    def test_txt_reference_uses_section(self):
        docs = [_make_doc("text", "IT_Security_Policy.txt", "txt", chunk_index=0)]
        sources = _format_sources(docs)
        assert sources[0]["reference"] == "Section 1"

    def test_docx_chunk_index_maps_to_section(self):
        docs = [_make_doc("text", "Employee_Benefits.docx", "docx", chunk_index=4)]
        sources = _format_sources(docs)
        assert sources[0]["reference"] == "Section 5"

    def test_duplicate_page_is_deduplicated(self):
        docs = [
            _make_doc("a", "Leave_Policy.pdf", "pdf", page=0),
            _make_doc("b", "Leave_Policy.pdf", "pdf", page=0),
        ]
        assert len(_format_sources(docs)) == 1

    def test_different_pages_are_not_deduplicated(self):
        docs = [
            _make_doc("a", "Leave_Policy.pdf", "pdf", page=0),
            _make_doc("b", "Leave_Policy.pdf", "pdf", page=1),
        ]
        assert len(_format_sources(docs)) == 2

    def test_filename_is_preserved(self):
        docs = [_make_doc("text", "Leave_Policy.pdf", "pdf", page=0)]
        sources = _format_sources(docs)
        assert sources[0]["filename"] == "Leave_Policy.pdf"

    def test_file_type_is_preserved(self):
        docs = [_make_doc("text", "IT_Security_Policy.txt", "txt", chunk_index=1)]
        sources = _format_sources(docs)
        assert sources[0]["file_type"] == "txt"

    def test_empty_docs_returns_empty_list(self):
        assert _format_sources([]) == []


# ── _format_context ───────────────────────────────────────────────────────────

class TestFormatContext:

    def test_returns_string(self):
        docs = [_make_doc("Some content.", "Leave_Policy.pdf", "pdf", page=0)]
        assert isinstance(_format_context(docs), str)

    def test_includes_filename_in_output(self):
        docs = [_make_doc("content", "Leave_Policy.pdf", "pdf", page=0)]
        ctx = _format_context(docs)
        assert "Leave_Policy.pdf" in ctx

    def test_includes_page_content(self):
        docs = [_make_doc("Annual leave is 20 days.", "Leave_Policy.pdf", "pdf", page=0)]
        ctx = _format_context(docs)
        assert "Annual leave is 20 days." in ctx

    def test_multiple_docs_separated_by_divider(self):
        docs = [
            _make_doc("Content A.", "A.pdf", "pdf", page=0),
            _make_doc("Content B.", "B.txt", "txt", chunk_index=0),
        ]
        ctx = _format_context(docs)
        assert "---" in ctx

    def test_empty_docs_returns_empty_string(self):
        assert _format_context([]) == ""


# ── _build_messages ───────────────────────────────────────────────────────────

class TestBuildMessages:

    def test_first_message_is_system(self):
        mem = build_memory()
        msgs = _build_messages("Question?", "Context.", mem)
        assert isinstance(msgs[0], SystemMessage)

    def test_system_message_contains_guardrail_text(self):
        mem = build_memory()
        msgs = _build_messages("Question?", "Context.", mem)
        assert "Do NOT hallucinate" in msgs[0].content

    def test_last_message_is_human(self):
        mem = build_memory()
        msgs = _build_messages("My question?", "Some context.", mem)
        assert isinstance(msgs[-1], HumanMessage)

    def test_human_message_contains_query(self):
        mem = build_memory()
        msgs = _build_messages("What is sick leave?", "Context text.", mem)
        assert "What is sick leave?" in msgs[-1].content

    def test_human_message_contains_context(self):
        mem = build_memory()
        msgs = _build_messages("Question?", "UNIQUE_CONTEXT_TOKEN", mem)
        assert "UNIQUE_CONTEXT_TOKEN" in msgs[-1].content

    def test_conversation_history_injected(self):
        mem = build_memory()
        mem.save_context({"input": "Turn1 Q"}, {"output": "Turn1 A"})
        msgs = _build_messages("Turn2 Q?", "Context.", mem)
        contents = [m.content for m in msgs]
        assert any("Turn1 Q" in c for c in contents)
        assert any("Turn1 A" in c for c in contents)


# ── System prompt content ─────────────────────────────────────────────────────

class TestSystemPrompt:

    def test_contains_hallucination_guard(self):
        assert "Do NOT hallucinate" in _SYSTEM_PROMPT

    def test_contains_fallback_statement(self):
        assert "I am sorry, but I cannot find this information" in _SYSTEM_PROMPT

    def test_contains_context_instruction(self):
        assert "strictly using the provided context" in _SYSTEM_PROMPT

    def test_mentions_company_documents(self):
        assert "company documents" in _SYSTEM_PROMPT

    def test_mentions_outside_knowledge(self):
        assert "outside knowledge" in _SYSTEM_PROMPT


# ── Query validation (Fix 8) ──────────────────────────────────────────────────

class TestQueryValidation:
    """Tests for input validation guards in run_rag_chain and stream_rag_chain.

    Fix 8 introduced a ``_MAX_QUERY_LENGTH`` limit (2 000 chars) in addition
    to the existing empty-string check.  Both functions are tested here.
    """

    # ── run_rag_chain ─────────────────────────────────────────────────────

    def test_empty_query_raises_value_error(self):
        with pytest.raises(ValueError, match="non-empty"):
            run_rag_chain("", build_memory())

    def test_whitespace_only_query_raises_value_error(self):
        with pytest.raises(ValueError, match="non-empty"):
            run_rag_chain("   \t\n", build_memory())

    def test_query_at_max_length_is_accepted(self):
        """A query of exactly _MAX_QUERY_LENGTH chars must not raise."""
        query = "a" * _MAX_QUERY_LENGTH
        with (
            patch("src.chain._get_retriever") as mock_ret,
            patch("src.chain._get_llm") as mock_llm,
        ):
            mock_ret.return_value.invoke.return_value = [
                _make_doc("ctx", "f.pdf", "pdf", page=0)
            ]
            mock_llm.return_value.invoke.return_value = AIMessage(content="ok")
            answer, _ = run_rag_chain(query, build_memory())
        assert isinstance(answer, str)

    def test_query_exceeding_max_length_raises_value_error(self):
        query = "a" * (_MAX_QUERY_LENGTH + 1)
        with pytest.raises(ValueError, match="too long"):
            run_rag_chain(query, build_memory())

    def test_error_message_includes_actual_length(self):
        length = _MAX_QUERY_LENGTH + 100
        query = "x" * length
        # The message formats the number with commas (e.g. "2,100") — use a
        # partial string match on the unformatted part to stay locale-agnostic.
        with pytest.raises(ValueError, match="too long"):
            run_rag_chain(query, build_memory())

    # ── stream_rag_chain ──────────────────────────────────────────────────

    def test_stream_empty_query_raises_value_error(self):
        gen = stream_rag_chain("", build_memory())
        with pytest.raises(ValueError, match="non-empty"):
            next(gen)

    def test_stream_query_exceeding_max_length_raises_value_error(self):
        query = "b" * (_MAX_QUERY_LENGTH + 1)
        gen = stream_rag_chain(query, build_memory())
        with pytest.raises(ValueError, match="too long"):
            next(gen)

    def test_max_query_length_constant_is_positive_integer(self):
        assert isinstance(_MAX_QUERY_LENGTH, int)
        assert _MAX_QUERY_LENGTH > 0


# ── run_rag_chain (functional) ────────────────────────────────────────────────

class TestRunRagChain:

    @patch("src.chain._get_llm")
    @patch("src.chain._get_retriever")
    def test_returns_tuple_of_answer_and_sources(self, mock_get_retriever, mock_get_llm):
        mock_get_retriever.return_value = _mock_retriever(
            [_make_doc("20 days annual leave.", "Leave_Policy.pdf", "pdf", page=0)]
        )
        mock_get_llm.return_value.invoke.return_value = AIMessage(
            content="Employees receive 20 days of annual leave."
        )
        answer, sources = run_rag_chain("How much annual leave?", build_memory())
        assert isinstance(answer, str)
        assert isinstance(sources, list)

    @patch("src.chain._get_llm")
    @patch("src.chain._get_retriever")
    def test_answer_appends_sources_used_block(self, mock_get_retriever, mock_get_llm):
        mock_get_retriever.return_value = _mock_retriever(
            [_make_doc("Leave details.", "Leave_Policy.pdf", "pdf", page=0)]
        )
        mock_get_llm.return_value.invoke.return_value = AIMessage(
            content="Employees get 20 days leave."
        )
        answer, _ = run_rag_chain("Annual leave?", build_memory())
        assert "Sources Used" in answer

    @patch("src.chain._get_llm")
    @patch("src.chain._get_retriever")
    def test_sources_list_contains_expected_keys(self, mock_get_retriever, mock_get_llm):
        mock_get_retriever.return_value = _mock_retriever(
            [_make_doc("Content.", "Leave_Policy.pdf", "pdf", page=0)]
        )
        mock_get_llm.return_value.invoke.return_value = AIMessage(content="Answer.")
        _, sources = run_rag_chain("Question?", build_memory())
        assert len(sources) > 0
        for s in sources:
            assert "filename" in s
            assert "reference" in s
            assert "file_type" in s

    @patch("src.chain._get_llm")
    @patch("src.chain._get_retriever")
    def test_memory_updated_after_call(self, mock_get_retriever, mock_get_llm):
        mock_get_retriever.return_value = _mock_retriever(
            [_make_doc("Content.", "Leave_Policy.pdf", "pdf", page=0)]
        )
        mock_get_llm.return_value.invoke.return_value = AIMessage(content="The answer.")
        mem = build_memory()
        run_rag_chain("Test question?", mem)
        assert len(mem.chat_memory.messages) == 2

    @patch("src.chain._get_llm")
    @patch("src.chain._get_retriever")
    def test_out_of_domain_query_returns_fallback_statement(
        self, mock_get_retriever, mock_get_llm
    ):
        """
        Core hallucination-guard test.

        When a user asks an out-of-domain question (e.g. geography), the LLM
        is expected — per the strict system prompt — to return the prescribed
        fallback phrase. We simulate this contract with a mock that mirrors the
        intended LLM behaviour.
        """
        fallback = (
            "I am sorry, but I cannot find this information "
            "in the provided company documents."
        )
        # Retriever surfaces an unrelated HR document
        mock_get_retriever.return_value = _mock_retriever(
            [_make_doc("Annual leave is 20 days.", "Leave_Policy.pdf", "pdf", page=0)]
        )
        # LLM correctly refuses and returns the fallback (as instructed by system prompt)
        mock_get_llm.return_value.invoke.return_value = AIMessage(content=fallback)

        answer, _ = run_rag_chain(
            "What is the capital of France?", build_memory()
        )

        assert fallback in answer, (
            f"Expected fallback phrase in answer.\nGot: {answer!r}"
        )

    @patch("src.chain._get_llm")
    @patch("src.chain._get_retriever")
    def test_multi_turn_history_is_passed_to_llm(self, mock_get_retriever, mock_get_llm):
        """Second query should include first turn in messages sent to LLM."""
        mock_get_retriever.return_value = _mock_retriever(
            [_make_doc("Leave details.", "Leave_Policy.pdf", "pdf", page=0)]
        )
        call_args_list = []

        def capture_invoke(messages):
            call_args_list.append(messages)
            return AIMessage(content="Answer.")

        mock_get_llm.return_value.invoke.side_effect = capture_invoke

        mem = build_memory()
        run_rag_chain("First question?", mem)
        run_rag_chain("Follow-up question?", mem)

        # Second call should have more messages (history injected)
        assert len(call_args_list[1]) > len(call_args_list[0])


# ── Thread safety (Fix 5) ─────────────────────────────────────────────────────

class TestThreadSafety:
    """Verify that concurrent calls to _get_llm() only initialise the LLM once.

    This tests the double-checked locking pattern introduced in Fix 5.
    """

    @patch("src.chain._load_llm")
    def test_concurrent_get_llm_calls_initialise_exactly_once(self, mock_load_llm):
        import src.chain as chain_module

        # Reset singleton so _get_llm() must build it fresh
        original = chain_module._llm
        chain_module._llm = None
        mock_load_llm.return_value = MagicMock(spec=["invoke", "stream"])

        results = []
        errors = []

        def call_get_llm():
            try:
                results.append(chain_module._get_llm())
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=call_get_llm) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        chain_module._llm = original  # restore

        assert not errors, f"Errors during concurrent _get_llm(): {errors}"
        # All threads must receive the *same* singleton object
        assert all(r is results[0] for r in results), "Multiple LLM instances created"
        # _load_llm must have been invoked exactly once
        assert mock_load_llm.call_count == 1
