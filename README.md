# 🏢 Enterprise RAG Assistant

> A production-grade **Conversational Retrieval-Augmented Generation (RAG)** system for enterprise policy documents, built with LangChain, FAISS, BM25, CrossEncoder reranking, and Streamlit.
>
> **Developed for IIT Patna — Enterprise AI Systems Project**

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Solution Overview](#solution-overview)
3. [Architecture Diagram](#architecture-diagram)
4. [Technology Stack](#technology-stack)
5. [Project Structure](#project-structure)
6. [Setup Instructions](#setup-instructions)
7. [Environment Variable Requirements](#environment-variable-requirements)
8. [How to Run the Application](#how-to-run-the-application)
9. [Running the Unit Test Suite](#running-the-unit-test-suite)
10. [Sample Inputs & Outputs](#sample-inputs--outputs)
11. [Key Design Decisions](#key-design-decisions)
12. [Limitations](#limitations)

---

## Problem Statement

Large enterprises maintain hundreds of internal policy documents — HR handbooks, IT security policies, employee benefit guides, compliance manuals — spread across PDF, Word, and plain-text formats. Employees and HR teams routinely spend significant time manually searching through these documents to answer routine questions such as:

- *"How many days of annual leave am I entitled to?"*
- *"What is the company's MFA policy for remote access?"*
- *"How do I claim the wellness allowance?"*

**Core challenges this system addresses:**

1. **Information fragmentation** — Relevant content is scattered across multiple file formats and documents with no unified search interface.
2. **Hallucination risk** — Generic LLM responses often fabricate plausible-sounding but incorrect policy details, creating legal and compliance risk.
3. **Lack of traceability** — Employees cannot verify which document an answer came from, eroding trust.
4. **Context blindness** — Single-turn Q&A systems forget prior questions, forcing users to repeat context in every message.
5. **Retrieval precision** — Keyword search alone misses semantic variants; pure vector search misses exact terminology (e.g., policy codes, dollar amounts).

---

## Solution Overview

The Enterprise RAG Assistant is a **two-stage hybrid retrieval + reranking conversational AI** that:

- **Ingests** PDF, DOCX, and TXT documents using format-specific LangChain loaders, enriching every chunk with provenance metadata (filename, page, section).
- **Indexes** documents using a dual index — a **FAISS dense vector store** (semantic similarity) and an **in-memory BM25 retriever** (exact keyword match) — to maximise recall across both semantic and lexical dimensions.
- **Reranks** the merged candidate pool using a **CrossEncoder model** (`ms-marco-MiniLM-L-6-v2`) that evaluates each candidate jointly with the query to select the most relevant passages.
- **Generates** grounded answers via a **strict hallucination-prevention prompt** that instructs the LLM to use only retrieved context. If the answer is absent from documents, the model returns a prescribed fallback phrase rather than hallucinating.
- **Cites sources** automatically — every response includes a `Sources Used:` block with filename and page/section reference.
- **Maintains conversational context** via a rolling `ConversationBufferWindowMemory` (last 5 turns), enabling coherent multi-turn dialogues.
- **Presents** everything in a clean **Streamlit** chat UI with sidebar controls for re-indexing and history management.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Streamlit UI  (app.py)                       │
│   ┌────────────┐   ┌──────────────────┐   ┌─────────────────────┐  │
│   │ Chat Input │   │  Re-index Button │   │ Clear History Button │  │
│   └─────┬──────┘   └────────┬─────────┘   └──────────┬──────────┘  │
└─────────┼────────────────────┼───────────────────────┼─────────────┘
          │                    │                        │
          ▼                    ▼                        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     chain.py  — run_rag_chain()                     │
│                                                                     │
│   ConversationBufferWindowMemory (k=5 rolling turns)                │
│                                                                     │
│   ┌─────────────────────────────────────────────────────────────┐   │
│   │               Two-Stage Retrieval Pipeline                  │   │
│   │                                                             │   │
│   │  Stage 1 — EnsembleRetriever [weights: 0.5 | 0.5]          │   │
│   │  ┌──────────────────────┐  ┌──────────────────────────────┐│   │
│   │  │  FAISS Dense Search  │  │     BM25 Keyword Search      ││   │
│   │  │  (all-MiniLM-L6-v2)  │  │  (rank_bm25, k=10)           ││   │
│   │  │  k=10 candidates     │  │  k=10 candidates             ││   │
│   │  └──────────┬───────────┘  └────────────┬─────────────────┘│   │
│   │             └──────────────┬─────────────┘                  │   │
│   │                            ▼                                 │   │
│   │         Stage 2 — CrossEncoder Reranker                     │   │
│   │         (cross-encoder/ms-marco-MiniLM-L-6-v2)              │   │
│   │                   top_k = 4 results                         │   │
│   └──────────────────────────┬──────────────────────────────────┘   │
│                              ▼                                      │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │  ChatPromptTemplate                                          │  │
│   │  ┌────────────────────────────────────────────────────────┐  │  │
│   │  │ SystemMessage  — strict hallucination guardrails       │  │  │
│   │  │ [History]      — last k=5 conversation turns           │  │  │
│   │  │ HumanMessage   — retrieved context + user question     │  │  │
│   │  └────────────────────────────────────────────────────────┘  │  │
│   └──────────────────────────┬───────────────────────────────────┘  │
│                              ▼                                      │
│              LLM (ChatOpenAI │ ChatGoogleGenerativeAI)              │
│                              ▼                                      │
│           Grounded Answer  +  Sources Used: [File, Page]           │
└─────────────────────────────────────────────────────────────────────┘

Document Ingestion Pipeline (scripts/generate_sample_data.py + src/ingestion.py)
─────────────────────────────────────────────────────────────────────────────────
  data/sample_docs/
  ├── Leave_Policy.pdf          PyPDFDirectoryLoader → page-level Documents
  ├── IT_Security_Policy.txt    DirectoryLoader + TextLoader
  └── Employee_Benefits.docx   DirectoryLoader + Docx2txtLoader
                 │
                 ▼
  RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)
                 │
                 ▼
  ┌─────────────────────────┐
  │  FAISS Index            │  ←  HuggingFaceEmbeddings (all-MiniLM-L6-v2)
  │  data/faiss_index/      │     saved as index.faiss + index.pkl
  └─────────────────────────┘
  BM25Retriever (in-memory, rebuilt on each app start)
```

---

## Technology Stack

| Layer | Technology | Version | Purpose |
|---|---|---|---|
| **Orchestration** | LangChain | 1.3.16 | High-level chain composition |
| **Orchestration** | langchain-classic | 1.0.8 | `ConversationBufferWindowMemory`, `EnsembleRetriever`, `CrossEncoderReranker` |
| **Orchestration** | langchain-community | 0.4.2 | BM25Retriever, FAISS, HuggingFace cross-encoders |
| **Orchestration** | langchain-text-splitters | 1.1.2 | `RecursiveCharacterTextSplitter` |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | via langchain-huggingface 1.2.2 | Dense vector encoding of document chunks |
| **Dense Index** | FAISS (`faiss-cpu`) | 1.15.0 | Approximate nearest-neighbour vector search |
| **Sparse Index** | BM25 (`rank-bm25`) | 0.2.2 | TF-IDF keyword-based retrieval |
| **Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | via sentence-transformers 6.0.0 | Precision reranking of candidate passages (configurable via `CROSS_ENCODER_MODEL`) |
| **LLM (OpenAI)** | ChatOpenAI | via langchain-openai 1.6.0 | Answer generation (GPT-3.5-Turbo or GPT-4) |
| **LLM (Google)** | ChatGoogleGenerativeAI | via langchain-google-genai 4.3.5 | Answer generation (Gemini 2.5 Flash / Pro) |
| **UI** | Streamlit | 1.62.0 | Conversational web interface |
| **PDF Parsing** | pypdf | 6.16.1 | Page-level document loading |
| **DOCX Parsing** | docx2txt | 0.9 | Word document text extraction |
| **PDF Generation** | fpdf2 | 2.8.8 | Sample document creation |
| **DOCX Generation** | python-docx | 1.2.0 | Sample document creation |
| **Config** | pydantic-settings v2 | 2.15.0 | Centralised, `.env`-backed settings (v2 `SettingsConfigDict` syntax) |
| **Logging** | Python `logging` | stdlib | Dual-handler file + stdout logging |
| **Testing** | pytest + unittest.mock | 9.1.1 | Unit testing and LLM mocking |
| **Environment** | python-dotenv | 1.2.3 | `.env` file loading |

---

## Project Structure

```
Enterprise Rag Assistant/
├── app.py                        # Streamlit UI
├── requirements.txt              # Pinned dependencies
├── pytest.ini                    # Test runner config
├── .env.example                  # API key template
├── README.md
│
├── src/
│   ├── config.py                 # Centralised settings (pydantic-settings)
│   ├── logger.py                 # Dual-handler logging (file + stdout)
│   ├── ingestion.py              # Document loading & chunking
│   ├── vectorstore.py            # FAISS build/save/load + BM25
│   ├── retrieval.py              # Hybrid EnsembleRetriever + CrossEncoder
│   └── chain.py                  # Conversational RAG chain + memory
│
├── scripts/
│   └── generate_sample_data.py  # Generates Leave_Policy.pdf, IT_Security_Policy.txt,
│                                 # Employee_Benefits.docx in data/sample_docs/
│
├── tests/
│   ├── conftest.py               # Shared session-scoped fixtures
│   ├── test_ingestion.py         # Loading + chunking tests (20 tests)
│   ├── test_retrieval.py         # FAISS / BM25 / reranker tests (20 tests)
│   └── test_chain.py             # Memory / prompt / hallucination guard tests (25 tests)
│
├── data/
│   ├── sample_docs/              # Source documents
│   └── faiss_index/              # Persisted FAISS index (index.faiss + index.pkl)
│
└── logs/
    └── app.log                   # Rotating application log
```

---

## Setup Instructions

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11 | https://python.org/downloads — tick "Add to PATH" |
| pip | >= 23 | Bundled with Python 3.11 |
| Internet | — | First run downloads embedding + reranker models (~90 MB) |

### Step-by-step Installation (Windows)

```powershell
# 1. Navigate to the project root
cd "E:\AI Projects\IIT Patna Project\Enterprise Rag Assistant"

# 2. Create an isolated virtual environment
python -m venv venv

# 3. Activate the virtual environment (Windows)
venv\Scripts\activate
# Your prompt should now show (venv) at the start

# 4. Upgrade pip
python -m pip install --upgrade pip

# 5. Install all pinned dependencies
pip install -r requirements.txt

# 6. Create your local .env from the template
copy .env.example .env
# Open .env and set at least one API key (see next section)
```

> To deactivate later, run `deactivate`.

---

## Environment Variable Requirements

Copy `.env.example` to `.env` and fill in values before starting the app.

```ini
# .env — DO NOT commit to version control (protected by .gitignore)

# LLM API Keys (at least one required)
OPENAI_API_KEY=sk-...           # https://platform.openai.com/api-keys
GOOGLE_API_KEY=AIza...          # https://aistudio.google.com/apikey

# Optional overrides (defaults shown)
OPENAI_MODEL=gpt-3.5-turbo
GOOGLE_MODEL=gemini-2.5-flash   # "gemini-1.5-flash" unavailable on current API; "gemini-pro" deprecated Feb 2025
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
CHUNK_SIZE=500
CHUNK_OVERLAP=50
TOP_K=4
DATA_DIR=data/sample_docs
FAISS_INDEX_DIR=data/faiss_index
LOGS_DIR=logs
```

**Selection logic:** `OPENAI_API_KEY` takes priority (streaming enabled). If absent, `GOOGLE_API_KEY` is used (standard display). If neither is set, the app shows an error banner and disables chat input.

### Full configuration reference

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(empty)* | OpenAI API key |
| `GOOGLE_API_KEY` | *(empty)* | Google Generative AI key |
| `OPENAI_MODEL` | `gpt-3.5-turbo` | OpenAI model name |
| `GOOGLE_MODEL` | `gemini-2.5-flash` | Google model name (`gemini-1.5-flash` unavailable on current API; `gemini-pro` deprecated Feb 2025) |
| `HF_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `CROSS_ENCODER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model (overridable without code changes) |
| `CHUNK_SIZE` | `500` | Max characters per text chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between adjacent chunks |
| `TOP_K` | `4` | Final passages returned after reranking |
| `FAISS_INDEX_DIR` | `data/faiss_index` | Persisted FAISS index directory |
| `DATA_DIR` | `data/sample_docs` | Source document directory |
| `LOGS_DIR` | `logs` | Log file directory |

---

## How to Run the Application

### Step 1 — Generate sample policy documents *(first time only)*

```powershell
python scripts/generate_sample_data.py
```

Expected output:
```
Writing sample documents to: ...\data\sample_docs

[OK] Created Leave_Policy.pdf
[OK] Created IT_Security_Policy.txt
[OK] Created Employee_Benefits.docx
```

> Skip this step if you are adding your own `.pdf`, `.txt`, or `.docx` files to `data/sample_docs/`.

### Step 2 — Build the search index

**Option A — Via the Streamlit UI (recommended)**

Start the app (Step 3) and click **Re-index Documents** in the sidebar.

**Option B — Via command line**

```powershell
python -c "
from src.ingestion import load_documents, split_documents
from src.vectorstore import build_vectorstore
chunks = split_documents(load_documents('data/sample_docs'))
build_vectorstore(chunks)
print('Index built:', len(chunks), 'chunks')
"
```

### Step 3 — Launch the Streamlit app

```powershell
streamlit run app.py
```

Opens at **http://localhost:8502**

| UI Control | Location | Action |
|---|---|---|
| Chat input | Bottom of page | Type and submit a question |
| **Re-index Documents** | Sidebar | Rebuild FAISS index from `data/sample_docs/` |
| **Clear Chat History** | Sidebar | Reset conversation memory and UI |
| Status panel | Sidebar | Index status, document count, active LLM |
| **View Source Citations** | Below each answer | Expandable source file + page/section list |

---

## Running the Unit Test Suite

```powershell
# Run the full suite (90 tests)
python -m pytest

# Run individual files
python -m pytest tests/test_ingestion.py -v
python -m pytest tests/test_retrieval.py -v
python -m pytest tests/test_chain.py -v

# With coverage (requires pip install pytest-cov)
python -m pytest --cov=src --cov-report=term-missing
```

### Test suite overview

| File | Tests | Scope |
|---|---|---|
| `tests/test_ingestion.py` | 25 | PDF/TXT/DOCX loading, metadata fields, chunk size bounds, empty-input edge cases |
| `tests/test_retrieval.py` | 27 | FAISS relevance, BM25 precision, reranker type checks, `top_k` bound, error handling, **cross-encoder model configurability** (Fix 6) |
| `tests/test_chain.py` | 47 | Memory factory, source/context formatters, message builder, system prompt content, hallucination guard, multi-turn history, **query length validation** (Fix 8), **thread safety** (Fix 5) |

**Result:** `99 passed, 0 warnings` (session-scoped fixtures avoid repeated model loading; all deprecation warnings suppressed by targeted `pytest.ini` filters)

---

## Sample Inputs & Outputs

### Query 1 — Annual leave entitlement ✅

**Input**
```
How many days of annual leave are employees entitled to?
```
**Output**
```
All permanent employees are entitled to 20 days of paid annual leave per calendar year.
Leave accrues at 1.67 days per month of service and must be requested at least 5 working
days in advance via the HR portal. Unused leave may be carried forward up to a maximum
of 5 days, which must be used by 31 March of the following year.
```
**Sources Used:** `Leave_Policy.pdf (Page 1)`

---

### Query 2 — Wellness allowance eligible expenses ✅

**Input**
```
What expenses can I claim under the wellness allowance?
```
**Output**
```
The annual Wellness Allowance of $500 covers:
- Gym memberships and fitness classes (yoga, pilates, CrossFit, swimming)
- Sports equipment purchases (up to $200 per item)
- Mental health and counselling sessions
- Nutrition and dietitian consultations
- Mindfulness app subscriptions (Calm, Headspace, etc.)
- Health screening packages not covered by group insurance

Claims must be submitted via the HR portal with valid receipts within 60 days.
Unused wellness allowance does not carry forward to the next year.
```
**Sources Used:** `Employee_Benefits.docx (Section 4)`

---

### Query 3 — IT password policy ✅

**Input**
```
What are the password requirements and how often do they expire?
```
**Output**
```
Passwords must be at least 12 characters and include uppercase letters, lowercase
letters, digits, and special characters. They expire every 90 days, with a 10-day
advance reminder. The last 10 passwords cannot be reused. Multi-Factor Authentication
(MFA) is mandatory for all corporate systems and remote VPN access.
```
**Sources Used:** `IT_Security_Policy.txt (Section 2)`

---

### Query 4 — Out-of-domain query (hallucination guard activated) ❌

**Input**
```
What is the capital of France?
```
**Output**
```
I am sorry, but I cannot find this information in the provided company documents.
```

> The strict system prompt prohibits outside knowledge. Since no retrieved chunk
> contains this answer, the model returns the prescribed fallback phrase rather
> than hallucinating "Paris". This behaviour is verified by
> `test_out_of_domain_query_returns_fallback_statement` in `tests/test_chain.py`.

---

### Query 5 — Multi-turn conversation 💬

```
Turn 1
  User      : How many sick leave days do employees get per year?
  Assistant : Employees are entitled to 10 days of paid sick leave per year.
              Absence exceeding 2 consecutive days requires a medical certificate.
              Sources Used: Leave_Policy.pdf (Page 1)

Turn 2
  User      : Do unused sick days carry forward?
  Assistant : No. Sick leave does not carry forward to the following year and is
              not encashable at any point.
              Sources Used: Leave_Policy.pdf (Page 1)

Turn 3
  User      : What about annual leave — how many days can carry forward?
  Assistant : Up to 5 days of unused annual leave may be carried forward, but must
              be utilised by 31 March of the subsequent year. Any remaining balance
              is forfeited.
              Sources Used: Leave_Policy.pdf (Page 1)
```

> `ConversationBufferWindowMemory(k=5)` injects the prior two turns into every
> subsequent prompt, enabling coherent follow-up without context repetition.

---

## Key Design Decisions

### 1. Hybrid FAISS + BM25 retrieval over vector-only search
Pure dense retrieval excels at semantic similarity but can miss exact-match terms (policy codes, dollar amounts, role titles). Pure BM25 misses paraphrased queries. Combining both with equal weights `[0.5, 0.5]` in `EnsembleRetriever` provides high recall at the candidate stage before precision reranking.

### 2. CrossEncoder as a second reranking stage
Bi-encoder embeddings encode query and passage independently, losing cross-attention signals. A CrossEncoder scores each `(query, passage)` pair jointly, dramatically improving precision. Running it only on the merged top-20 pool (10 FAISS + 10 BM25) keeps CPU latency at ~0.6–0.7 s per query.

### 3. Strict hallucination guardrails via the system prompt
The system prompt explicitly prohibits outside knowledge and specifies the exact fallback phrase the model must output when the answer is absent from context. This constraint is deterministic, verifiable, and testable — unlike relying on an LLM's inherent tendencies.

### 4. Source citation derived from chunk metadata, not LLM output
Every chunk carries `filename`, `file_type`, `page` (PDFs), and `chunk_index` metadata at ingestion time. The `_format_sources()` function extracts and deduplicates these programmatically — the LLM is never asked to generate citations, eliminating citation hallucination entirely.

### 5. Thread-safe lazy singleton pattern for the retriever and LLM
Building the full pipeline (FAISS load + BM25 rebuild + CrossEncoder load) costs ~3–5 s. Module-level `_retriever` and `_llm` singletons are initialised once on first query and reused across all subsequent calls. **Double-checked locking** (`threading.Lock`) guards each singleton: the outer `if` avoids acquiring the lock on every hot-path call; the inner `if` (inside the lock) prevents duplicate initialisation when two sessions race on the very first request. Singletons are reset to `None` after re-indexing so the next query loads the updated index.

### 6. Persistent FAISS index to avoid re-embedding on restart
`FAISS.save_local()` writes `index.faiss` + `index.pkl` after each `build_vectorstore()` call. Subsequent app starts reload in ~1 s via `FAISS.load_local()`, avoiding costly re-embedding of the entire corpus each time. The `allow_dangerous_deserialization=True` flag required by FAISS is documented inline with an explicit security note — the flag is safe here because the index is a locally-generated, trusted artefact.

### 7. `ConversationBufferWindowMemory(k=5)` over unbounded history
Unlimited history inflates prompts, increasing token cost and latency. A rolling 5-turn window is sufficient for typical policy Q&A follow-ups while keeping prompts bounded. The memory object lives in `st.session_state` and is passed by reference into each chain call.

### 8. `pydantic-settings` v2 for centralised, validated configuration
All tunable parameters — including the new `CROSS_ENCODER_MODEL` — are declared as typed fields in a single `Settings` class using the v2 `SettingsConfigDict` API (no deprecated `Field(env=...)` syntax). pydantic-settings v2 automatically maps each field name to its uppercase environment variable (`chunk_size` → `CHUNK_SIZE`, `cross_encoder_model` → `CROSS_ENCODER_MODEL`, etc.), providing type validation, `.env` file support, and environment variable overrides without scattered `os.getenv()` calls throughout the codebase.

### 9. Dynamic LLM selection at runtime
`_load_llm()` checks `OPENAI_API_KEY` first (streaming support), then `GOOGLE_API_KEY`, and raises a descriptive `RuntimeError` if neither is available — making the dependency explicit rather than failing silently at inference time. The default Google model is `gemini-2.5-flash` (`gemini-1.5-flash` was found unavailable on the current Google AI Studio API key; `gemini-pro` was deprecated by Google in February 2025).

### 10. `RecursiveCharacterTextSplitter` with `add_start_index=True`
The recursive splitter respects natural text boundaries (paragraphs, sentences) before falling back to characters, producing more semantically coherent chunks than fixed-size splitting. `add_start_index=True` adds a character-offset key to each chunk's metadata for precise location tracking.

### 11. All model names are fully configurable via environment variables
The CrossEncoder reranker model — previously a hardcoded module-level constant — is now exposed as `CROSS_ENCODER_MODEL` in `.env`. This means swapping to a lighter model (e.g., `ms-marco-MiniLM-L-2-v2` for lower CPU latency) or a heavier one requires only a one-line change in `.env`, with no code modifications.

### 12. Query length guard prevents token overflow
Both `run_rag_chain()` and `stream_rag_chain()` reject queries exceeding 2 000 characters with a descriptive `ValueError`. This prevents accidental token-limit errors from the LLM API and protects against runaway API costs from adversarially long inputs.

---

## Limitations

### 1. CPU-bound inference latency
The CrossEncoder and embedding model run on CPU by default. Typical query latency is **0.6–0.7 s** per request. High-concurrency production deployments would require GPU instances or a lighter reranker model (e.g., `ms-marco-MiniLM-L-2-v2`). The reranker model is now configurable via `CROSS_ENCODER_MODEL` in `.env` without code changes.

### 2. BM25 is not persisted — rebuilt on every app start
BM25 indices are in-memory only. For large corpora (thousands of documents), rebuilding BM25 from all chunks at startup could add several seconds to cold-start time. A serialisation mechanism (e.g., `pickle`) would address this.

### 3. FAISS flat index does not scale to millions of vectors
`FAISS.from_documents()` uses an exact `IndexFlatL2`, which is O(n) at search time. For corpora exceeding ~100,000 chunks, an approximate index (`IVFFlat`, `HNSW`) with clustering should replace it.

### 4. Single flat directory ingestion only
`load_documents()` scans one flat directory for `.pdf`, `.txt`, and `.docx` files. Nested subdirectories, SharePoint libraries, cloud storage (S3, GCS), or live database feeds are not supported without extending the ingestion module.

### 5. No authentication or access control
The Streamlit app has no user authentication. In a real enterprise deployment, documents with different sensitivity classifications would require RBAC to prevent unauthorised employees from querying restricted content.

### 6. Memory is session-local and non-persistent
`ConversationBufferWindowMemory` is held in `st.session_state` and lost when the browser tab is closed or the server restarts. Persistent cross-session history would require a database-backed store (e.g., LangGraph `Store` API with PostgreSQL).

### 7. Context window capped at `top_k=4` chunks
Reranking retains only 4 passages by default. For complex multi-section queries, relevant context may be discarded. Increasing `TOP_K` in `.env` improves recall at the cost of higher token consumption and LLM latency.

### 8. No incremental index updates
Adding a new document requires clicking **Re-index Documents**, which rebuilds the entire FAISS index from scratch. An incremental `vectorstore.add_documents()` pattern would allow adding files without full re-embedding.

### 9. LLM dependency on third-party API keys
The system cannot generate answers without a valid `OPENAI_API_KEY` or `GOOGLE_API_KEY`. Fully offline operation would require a locally-hosted LLM (e.g., Ollama + LLaMA 3), which is a planned future enhancement.

### 10. Chunk size may split mid-sentence in long paragraphs
`CHUNK_SIZE=500` characters can occasionally bisect a sentence when a paragraph exceeds the limit without a natural break. Increasing `CHUNK_SIZE` or adding a sentence-aware splitter would improve chunk coherence for dense policy text.

### 11. Query length is capped at 2 000 characters *(mitigated)*
Queries exceeding 2 000 characters are now rejected at the API boundary with a descriptive `ValueError` before any LLM call is made, preventing token overflow errors. Users requiring longer inputs can raise `_MAX_QUERY_LENGTH` in `src/chain.py`.
