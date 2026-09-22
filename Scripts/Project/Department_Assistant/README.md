# 🎓 Department Knowledge Assistant

A RAG (Retrieval-Augmented Generation) pipeline built with **LangGraph** that enables intelligent Q&A capabilities for IIT Bombay's Chemical Engineering Department website. The system scrapes department information (web pages & PDFs), processes it into a vector database, and provides accurate answers to user queries through a Streamlit chat interface.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup Instructions](#setup-instructions)
- [Configuration](#configuration)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Data Pipeline](#data-pipeline)
- [Technologies Used](#technologies-used)
- [Limitations](#limitations)

---

## 🎯 Overview

This project implements a RAG pipeline orchestrated by **LangGraph** that:

1. **Scrapes** the Chemical Engineering department website (https://www.che.iitb.ac.in/)
2. **Downloads** course curriculum PDFs containing detailed course information
3. **Processes** all content into chunks and generates embeddings
4. **Stores** everything in a Chroma vector database
5. **Retrieves** relevant information for user queries
6. **Generates** accurate, context-aware responses using Groq's LLM, with light conversation memory across a chat session

The dataset size (pages scraped, PDFs, chunks stored) depends on whatever the last run of `rebuild.py` found - see [Usage](#usage) for how to rebuild it, and check the summary it prints at the end for current numbers rather than relying on any number written here.

---

## 🏗️ Architecture

### LangGraph Workflow

The project uses **two separate LangGraph StateGraphs**:

#### 1. Data Collection Graph (run via `rebuild.py`)
```
START → Scraper Node → Processor Node → END
```

#### 2. Query Graph (run for each query)
```
START → Retriever Node → Responder Node → END
```

Each call to the query graph builds a brand-new state - the graph itself has no memory between calls. Conversation continuity comes from the caller (`app.py`) passing in recent chat history as part of that fresh state; see [Conversation Memory](#conversation-memory) below. Retrieval only ever searches on the current question's text - chat history is used by the responder alone, to understand what the current question refers to.

### Agent Flow Diagram

```
┌─────────────────────────────────────────────────────────┐
│                   ORCHESTRATOR AGENT                    │
│                    (LangGraph)                          │
└─────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
        ▼                                   ▼
┌───────────────┐                   ┌───────────────┐
│  Data Collect │                   │  Query Flow   │
│    Workflow   │                   │   Workflow    │
└───────────────┘                   └───────────────┘
        │                                   │
        ▼                                   ▼
┌───────────────┐                   ┌───────────────┐
│ Scraper Agent │                   │Retriever Agent│
│  - Web pages  │                   │ - MMR search  │
│  - PDF files  │                   │ - Top-k docs  │
└───────────────┘                   └───────────────┘
        │                                   │
        ▼                                   ▼
┌───────────────┐                   ┌───────────────┐
│Processor Agent│                   │Responder Agent│
│  - Chunking   │                   │ - LLM prompt  │
│  - Embedding  │                   │ - Chat history│
│  - Storage    │                   │ - Generation  │
└───────────────┘                   └───────────────┘
        │                                   │
        ▼                                   ▼
┌───────────────┐                   ┌───────────────┐
│ Vector Store  │◄──────────────────│    User       │
│  (Chroma DB)  │                   │  (Streamlit)  │
└───────────────┘                   └───────────────┘
```

---

## ✨ Features

### Intelligent Web Scraping
- **BFS (Breadth-First Search)** algorithm for systematic crawling
- **Domain filtering** to stay within allowed domain
- **Duplicate detection** to avoid re-scraping pages
- **Polite crawling** with configurable delays, and a `robots.txt` check before each page
- **Retries** transient failures (timeouts, connection errors, 5xx) a couple of times before giving up on a page; permanent failures (403/404) are not retried
- **Boilerplate stripping** - `<nav>`, `<header>`, `<footer>` are removed before extracting text, so the same site-wide menu/footer text doesn't get duplicated into every single page's chunks
- **PDF downloading** for curriculum and policy documents

### PDF Processing
- Automatic extraction of text from downloaded PDFs
- Curriculum files with course-instructor mappings
- Faculty advisor lists
- Policy documents
- Exam schedules

### Smart Text Processing
- **Recursive text chunking** with overlap for context preservation
- **HuggingFace embeddings** (sentence-transformers/all-MiniLM-L6-v2)
- **Metadata tracking** for source URLs and document types
- **Chroma vector database** for efficient similarity search

### Powerful Query System
- **MMR (Maximal Marginal Relevance) search** - balances relevance against diversity of retrieved chunks, not just raw similarity
- **Top-k retrieval** (k=10, fetch_k=30 candidates before MMR filtering - set directly in `retriever_node.py`'s `retrieve_documents()`, not a `config.py` constant)
- **Cached embedding model + DB connection** - loaded once per process (first query only), not reloaded on every question
- **Context-aware response generation** using Groq LLM
- **Clean source citations** - real `Source:` URLs are listed in a `Sources:` section at the end of an answer, only when actually used; the model is explicitly told not to fabricate inline citation brackets or footnote-style markers
- **Conversation memory** - see [Conversation Memory](#conversation-memory) below
- **Minimal hallucination** (temperature=0)
- **Retrieval/generation errors surface as errors** - if a node fails internally, `run_query()` raises rather than silently falling through to a generic "I don't have that information" answer

### Conversation Memory
- `app.py` passes the Streamlit chat history into `run_query(query, chat_history=...)` on every question
- The responder converts the last `CHAT_HISTORY_TURNS` exchanges (3 by default, in `config.py`) into real prior conversation turns for the LLM, so follow-ups like *"what about the other one?"* resolve correctly
- This is memory for understanding the question only - every factual claim in an answer must still come from the current question's retrieved context, not from what was said in earlier turns, so the model doesn't start treating its own prior (possibly context-free) answers as a source
- Retrieval itself is **not** history-aware - it searches on the current question's raw text only, so a follow-up that depends entirely on prior turns with no restated keywords may retrieve less relevant documents even though the response text correctly understands what's being asked (see [Limitations](#limitations))

### User-Friendly Interface
- **Streamlit chat interface** with conversation history
- **Example questions** to guide users
- **Database statistics** display
- **Clear pointer to `rebuild.py`** when the database isn't built yet, instead of triggering a long scrape from inside the Streamlit session itself

---

## 📁 Project Structure

```
Department_Assistant/
│
├── config.py                   # Configuration (API keys, models, parameters)
├── state.py                    # PipelineState TypedDict for LangGraph (incl. chat_history)
├── main.py                     # Orchestrator with both graph workflows
├── app.py                      # Streamlit frontend
├── rebuild.py                  # Standalone script: backs up old data, then re-scrapes + rebuilds
├── requirements.txt            # Pinned dependencies for this project specifically
├── .env                        # Environment variables (GROQ_API_KEY)
│
├── nodes/                      # Individual agent nodes
│   ├── scraper_node.py        # Web scraping & PDF downloading
│   ├── processor_node.py      # Chunking, embedding, storage
│   ├── retriever_node.py      # Vector similarity search
│   └── responder_node.py      # LLM response generation + chat history handling
│
├── downloaded_pdfs/            # Downloaded curriculum PDFs
│
└── README.md                   # This file

C:\department_vector_db\        # Chroma database storage - lives OUTSIDE this
                                 # folder deliberately, see Configuration below
```

`__pycache__/`, one-off debug scripts, and old `rebuild.py`-generated backup folders (`*_backup_<timestamp>/`, `department_vector_db_old_backup/`) are not part of the project structure - they're either regenerated automatically or were manually cleaned up once the current setup was confirmed working. `.gitignore` keeps them (and the database/PDFs) out of git regardless.

---

## 🚀 Setup Instructions

### Prerequisites

- **Python 3.8+**
- **Virtual environment** (recommended)
- **Groq API Key** (free tier available at https://console.groq.com/)

### Installation

1. **Navigate to the project directory**
   ```bash
   cd "Scripts/Project/Department_Assistant"
   ```

2. **Create and activate virtual environment** (if not already done)
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1  # Windows PowerShell
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   This file is pinned to exact versions, deliberately - unpinned `>=` requirements are how this project ended up several major versions ahead of what it was built against (LangChain 0.3→1.2, LangGraph 0.2→1.0, Chroma 0.5→1.4) without anyone changing anything on purpose. Re-verify deliberately before bumping any of these versions.

   Key packages:
   - `langgraph` / `langchain-core` / `langchain-community` - Workflow orchestration & LLM framework
   - `langchain-groq` - Groq integration
   - `langchain-huggingface` / `sentence-transformers` - HuggingFace embeddings
   - `langchain-chroma` / `chromadb` - Vector database
   - `beautifulsoup4` - Web scraping
   - `pypdf` - PDF processing
   - `streamlit` - Web interface
   - `requests` - HTTP client
   - `python-dotenv` - Environment management

4. **Set up environment variables**

   Create a `.env` file in the project directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

   Get your free Groq API key from: https://console.groq.com/

---

## ⚙️ Configuration

All configuration is centralized in [`config.py`](config.py):

### API Configuration
```python
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
```

### Models
```python
# LLM Model
LLM_MODEL = "openai/gpt-oss-120b"   # Groq model - llama-3.3-70b-versatile was
                                     # deprecated/decommissioned by Groq (shutdown
                                     # 08/16/26), switched to Groq's recommended
                                     # replacement
LLM_TEMPERATURE = 0                 # Minimal hallucination
MAX_TOKENS = 1500                   # Response length cap

# Embedding Model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # HuggingFace
```

### Scraper Parameters
```python
BASE_URL = "https://www.che.iitb.ac.in/"
ALLOWED_DOMAIN = "che.iitb.ac.in"
MAX_PAGES = 350                  # Maximum pages to scrape
SCRAPER_DELAY = 1.5              # Seconds between requests
REQUEST_TIMEOUT = 10             # HTTP timeout in seconds
SCRAPER_MAX_RETRIES = 2          # Retries for a transient failure before giving up on a page
```

### Processing Parameters
```python
CHUNK_SIZE = 1500                # Characters per chunk
CHUNK_OVERLAP = 300              # Overlap for context
VECTOR_DB_PATH = r"C:\department_vector_db"   # See note below
COLLECTION_NAME = "che_department"
# Retrieval k (10) is set directly in retriever_node.py's retrieve_documents(),
# not here - there is no RETRIEVAL_K constant in this file.
```

### Conversation Memory Parameters
```python
CHAT_HISTORY_TURNS = 3           # How many prior user/assistant exchanges the
                                  # responder is shown, so follow-up questions can
                                  # be resolved. Kept small on purpose - it's for
                                  # following the thread of a conversation, not a
                                  # substitute for the retrieved context.
```

**Why `VECTOR_DB_PATH` points outside this folder:** this project lives inside a OneDrive-synced folder, but Chroma's storage engine does file locking / memory-mapped I/O that clashes with OneDrive's sync client - this threw `disk I/O error`s when the database lived inside the synced folder. The database now lives at a plain local path instead. If you ever move the project, update this path (and re-run `rebuild.py`, or move the DB folder manually) - it will *not* auto-follow the project folder.

### Customization

To scrape a different department:
1. Change `BASE_URL` in `config.py`
2. Update `ALLOWED_DOMAIN` accordingly
3. Delete (or let `rebuild.py` back up) the folder at `VECTOR_DB_PATH`
4. Run `rebuild.py` again

---

## 📖 Usage

### Rebuilding the Data (Scrape + Process)

Use `rebuild.py` - it backs up whatever's currently at `VECTOR_DB_PATH` and in `downloaded_pdfs/` to timestamped folders first, then runs the full scrape + process pipeline fresh:

```bash
cd "Scripts\Project\Department_Assistant"
& "..\..\..\venv\Scripts\python.exe" rebuild.py
```

Expect this to take well over half an hour (350 pages at a 1.5s per-request delay, plus PDF downloads and embedding generation) - that's expected, not a hang. It's the same `collect_data()` pipeline the old "Initialize Database" Streamlit button used to call; it's now run standalone specifically so a 30+ minute request doesn't sit inside (and risk timing out) a Streamlit session.

It prints a summary at the end: pages scraped, PDFs downloaded, chunks stored. The timestamped backup folders it creates (`department_vector_db_backup_<timestamp>/`-style, `downloaded_pdfs_backup_<timestamp>/`) are meant to be temporary safety nets - once you've confirmed the new rebuild works, it's fine to delete them.

### Running the Chat Interface

```bash
streamlit run app.py
```

Or with the project venv:
```bash
& "..\..\..\venv\Scripts\python.exe" -m streamlit run app.py
```

Then open your browser to: **http://localhost:8501**

If the database hasn't been built yet, the sidebar shows the `rebuild.py` command to run instead of trying to build it from inside the app.

Note that if you edit any file under `nodes/` (or `config.py`, `main.py`, etc.) while the app is already running, Streamlit's "rerun on save" only re-executes `app.py` - it does **not** re-import already-cached modules. Fully stop (Ctrl+C) and restart `streamlit run app.py` to pick up changes to any file other than `app.py` itself.

### Testing Queries via CLI

```python
# In main.py
response = run_query("What are the research areas in the department?")
print(response)

# With conversation history, so a follow-up can refer back to the previous turn:
history = [
    {"role": "user", "content": "What are the research areas in the department?"},
    {"role": "assistant", "content": "..."},
]
response = run_query("Which of those focuses on polymers and colloids?", chat_history=history)
print(response)
```

### Example Queries

- "What are the research areas in the department?"
- "Who teaches CH 101?" (requires curriculum PDF data)
- "List all faculty members in the department"
- "What is the UG curriculum structure?"
- "Which faculty works on polymer research?"
- "What are the PhD admission requirements?"

---

## 🔧 How It Works

### 1. Scraper Node (`scraper_node.py`)

**Function**: Crawl website and download PDFs

**Algorithm**: BFS (Breadth-First Search)
```python
1. Load robots.txt once; fail open (proceed without restriction) if it can't be fetched
2. Start from BASE_URL
3. Maintain visited set and to_visit queue
4. For each URL:
   - Skip if robots.txt disallows it
   - If PDF: Download to downloaded_pdfs/ (retrying transient failures)
   - If web page: Strip nav/header/footer, extract text and links (retrying transient failures)
   - Add new valid links to queue
   - Mark as visited
5. Continue until MAX_PAGES reached or queue empty
```

**Output**:
```python
{
    "scraped_pages": [{"url": "...", "content": "...", "type": "web"}, ...],
    "pdf_files": [{"url": "...", "file_path": "..."}, ...]
}
```

### 2. Processor Node (`processor_node.py`)

**Function**: Chunk, embed, and store documents

**Process**:
```python
1. Create Document objects from web pages
2. Extract text from PDFs using PyPDFLoader
3. Chunk all documents with RecursiveCharacterTextSplitter
4. Generate embeddings using HuggingFace model
5. Store in Chroma vector database with metadata
```

**Chunking Strategy**:
- Chunk size: 1500 characters
- Overlap: 300 characters (preserves context across chunks)
- Separator hierarchy: `\n\n` → `\n` → ` ` → `""`

### 3. Retriever Node (`retriever_node.py`)

**Function**: Find relevant documents for query

**Process**:
```python
1. Get cached embeddings model + Chroma connection (created once per process, reused after)
2. Convert query to embedding vector
3. MMR search: fetch 30 candidates, then select the top 10 balancing
   relevance against diversity (lambda_mult=0.7)
4. Return top-k most relevant, non-redundant chunks
```

Note: this node only ever looks at `state["query"]` - it does not use `chat_history`. A follow-up question is resolved by the *responder*, not by re-running retrieval with reformulated search terms.

**Output**:
```python
{
    "retrieved_docs": [
        {"content": "...", "metadata": {"source": "https://...", ...}},
        ...
    ]
}
```

### 4. Responder Node (`responder_node.py`)

**Function**: Generate answer using LLM with context and recent conversation history

**Process**:
```python
1. Format retrieved documents as context
2. Create system prompt with instructions, including how to use chat history
   and how (and when) to cite sources
3. Convert the last CHAT_HISTORY_TURNS exchanges of chat_history into actual
   prior HumanMessage/AIMessage turns
4. Call Groq LLM with: system prompt + prior turns + current question/context
5. Return generated response
```

**System prompt covers**:
- Answering only from the provided context documents
- Using any included chat history to resolve what the current question refers to, without treating earlier answers as a source of facts
- Writing the answer as plain prose - no inline `(Source: ...)` or bracketed markers mid-sentence
- Appending a plain `Sources:` list of the real URLs used, only when the context actually contained one, and omitting the section entirely otherwise
- Never inventing citation IDs, footnote markers, or line-number-style annotations

**LLM Settings**:
- Model: `openai/gpt-oss-120b`
- Temperature: 0 (deterministic, factual)
- Max tokens: 1500

**Output**:
```python
{
    "response": "The department has 6 research areas: ...\n\nSources:\nhttps://..."
}
```

If either the retriever or responder node fails internally, `main.py`'s `run_query()` raises rather than returning a plausible-looking answer - `app.py` catches that and shows it as a real error in the chat.

---

## 🔄 Data Pipeline

### State Flow Through LangGraph (illustrative example)

```python
# Initial State (Data Collection)
{
    "scraped_pages": None,
    "pdf_files": None,
    "chunks": None,
    "stored_doc_ids": None,
    "query": None,
    "chat_history": None,
    "retrieved_docs": None,
    "response": None,
    "status": None,
    "error": None
}

# After Scraper
{
    "scraped_pages": [... pages],
    "pdf_files": [... PDFs],
    ...
}

# After Processor
{
    ...
    "chunks": [...],
    "stored_doc_ids": [...],
    "status": "Successfully processed N chunks"
}

# Query State (built fresh by run_query() on every call - no memory persists
# inside the graph; chat_history is passed in by the caller each time)
{
    "query": "Which of those focuses on polymers and colloids?",
    "chat_history": [
        {"role": "user", "content": "What are the research areas in the department?"},
        {"role": "assistant", "content": "..."}
    ],
    ...
}

# After Retriever
{
    ...
    "retrieved_docs": [10 relevant docs]
}

# After Responder
{
    ...
    "response": "The department has 6 research areas: ..."
}
```

---

## 🛠️ Technologies Used

Pinned exactly in [`requirements.txt`](requirements.txt) - these are the versions actually installed and verified working, not aspirational minimums.

### Core Framework
- **LangGraph** 1.0.7 - State graph orchestration
- **LangChain** 1.2.0 / **LangChain Core** 1.2.7 / **LangChain Community** 0.4.1

### LLM & Embeddings
- **Groq API** - Fast LLM inference (`openai/gpt-oss-120b`)
- **HuggingFace Transformers** / **Sentence Transformers** 5.2.0 - Sentence embeddings (all-MiniLM-L6-v2)
- **PyTorch** 2.9.1 (CPU)

### Vector Database
- **Chroma** 1.4.1 / **langchain-chroma** 1.1.0 - Vector storage and similarity search

### Web Scraping
- **BeautifulSoup4** 4.14.3 - HTML parsing
- **Requests** 2.32.5 - HTTP client

### PDF Processing
- **PyPDF** 6.6.2 - PDF text extraction

### Frontend
- **Streamlit** 1.54.0 - Interactive web interface

### Utilities
- **python-dotenv** 1.2.1 - Environment management
- **Pydantic** 2.12.5

---

## ⚠️ Limitations

### Current Limitations

1. **Scraping Scope**
   - Limited to che.iitb.ac.in domain
   - Max 350 pages (configurable)
   - Some faculty pages may be forbidden (403) or blocked by robots.txt
   - Dynamic JavaScript content not captured

2. **PDF Processing**
   - Text-based PDFs only (no OCR)
   - May have parsing errors for complex layouts
   - Image content is not extracted

3. **Embedding Model**
   - CPU-based (no GPU acceleration)
   - Limited to 384-dimensional vectors
   - English language optimized

4. **Response Quality**
   - Depends on chunk quality and retrieval accuracy
   - No query expansion or re-ranking

5. **Conversation Memory Is Shallow**
   - Only the responder sees chat history - retrieval always searches on just the current question's raw text
   - A follow-up whose subject depends entirely on prior turns, with none of the relevant keywords restated (e.g. "tell me more" with no noun to search on), can retrieve weaker documents even though the response text correctly understands what's being asked
   - History is capped at `CHAT_HISTORY_TURNS` (3) exchanges - older context in a long conversation is simply dropped, not summarized

6. **Performance**
   - Scraping is slow due to polite delays (1.5s per page) - budget well over half an hour for a full rebuild with MAX_PAGES=350
   - Embedding generation is CPU-intensive
   - The embedding model is now loaded once per process and cached (see Retriever Node above), so only the *first* query after starting the app is slow from model loading - subsequent queries reuse it

### Potential Improvements

- [ ] Add multi-modal processing (images, tables)
- [ ] Implement query expansion/rephrasing (including making retrieval itself history-aware, not just the responder)
- [ ] Add re-ranking for better retrieval
- [ ] Enable GPU acceleration for embeddings
- [ ] Add caching for frequent queries
- [ ] Implement incremental updates (delta scraping) instead of a full rebuild each time
- [ ] Summarize/compress older conversation turns instead of a hard cutoff at CHAT_HISTORY_TURNS
- [ ] Support for multiple departments
- [ ] Export chat history feature
- [ ] Consider a stronger embedding model than all-MiniLM-L6-v2 if retrieval quality still needs work after the nav/footer-stripping fix

---

## 🎓 Academic Context

**Project**: WIDS (Winter in Data Science) Project
**Institution**: IIT Bombay
**Department**: Chemical Engineering
**Year**: 2nd Year (2025-26)

**Learning Outcomes**:
- LangGraph state management
- RAG pipeline architecture
- Vector database operations
- Web scraping best practices
- LLM prompt engineering
- Full-stack AI application development

---

## 📝 Notes

- This is a personal project for learning purposes
- Not intended for production deployment
- Database and PDFs are not tracked in git
- `.gitignore` already covers `.env`, `department_vector_db/` (legacy in-folder path), `downloaded_pdfs/`, and other local artifacts
- The project folder was cleaned up (removed `__pycache__/`, a stray `_final_test.py` debug script, and superseded database/PDF backup folders) once the rebuilt database and current code were confirmed working - see git history around September 22, 2026 if any of the old paths are referenced somewhere unexpected

---

## 🤝 Acknowledgments

- **LangChain** & **LangGraph** communities for excellent documentation
- **Groq** for free, fast LLM API access
- **HuggingFace** for open-source embedding models
- **Streamlit** for rapid prototyping
- IIT Bombay Chemical Engineering Department for public web resources

---

## 📞 Support

For issues or questions about this project:
1. Check the configuration in `config.py`
2. Verify API keys in `.env`
3. Ensure all dependencies are installed (`pip install -r requirements.txt`)
4. Check terminal output for specific errors - real pipeline errors now surface as actual errors rather than a generic "I don't have that information" answer
5. If you just edited a file under `nodes/` (or `config.py`/`main.py`) and don't see the change, fully restart `streamlit run app.py` - see the note in [Usage](#usage)

---

**Last Updated**: September 22, 2026
**Status**: Rebuilt after a scrape targeting a deprecated Groq model, an unpinned-dependency version drift, and a OneDrive/Chroma file-locking issue. Since the rebuild: fixed the responder to stop fabricating citation-style brackets (now a plain end-of-answer `Sources:` list, real URLs only), added lightweight conversation memory (`chat_history`, capped at `CHAT_HISTORY_TURNS`), and cleaned up leftover build artifacts and superseded backups from the project folder. See git history / conversation notes for the full diagnosis if any of this resurfaces.
