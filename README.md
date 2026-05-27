<div align="center">

# ⟡ RAG Retrieval-Chatbot




A multi-agent AI pipeline that ingests your documents, indexes them into a semantic vector database, and retrieves precise context in real time — delivering grounded answers with zero hallucination and zero manual prompt engineering.

</div>

---

## Overview

When you ask a standard LLM a question about your private documents, it cannot answer — it only knows what it was trained on. Uploading sensitive documents to a third-party AI service introduces data privacy risks and still produces hallucinated responses when the model guesses instead of retrieves.

The **RAG Knowledge Assistant** solves this by acting as an intelligent semantic retrieval layer between your documents and the language model. The moment you ask a question, the system embeds your query, searches a local vector database for the most semantically relevant document chunks, and injects only verified, factual context into the LLM prompt — forcing grounded, citation-backed responses with no fabrication.

---

## Key Features

- **Universal Document Ingestion** — Accepts PDF, DOCX, TXT, MD, CSV, XLSX, and HTML files through a drag-and-drop UI with zero CLI interaction required
- **Semantic Vector Retrieval** — Embeds all document content using MiniLM-L6-v2 locally, performs cosine similarity search across ChromaDB to surface the top-5 most relevant chunks per query
- **Streaming AI Responses** — Answers stream token by token in real time via Python async generators, eliminating waiting delays on long outputs
- **In-App Upload Pipeline** — Full document ingestion, chunking, embedding, and indexing happens live inside the running interface — no separate terminal commands needed
- **Multi-Document Cross-Search** — Upload multiple files across different topics; the retriever searches all indexed content simultaneously and synthesizes a unified answer
- **Zero Embedding API Cost** — All embedding computation runs locally on CPU using `sentence-transformers`, removing OpenAI embedding fees entirely
- **Model Flexible** — Swap between any of Featherless AI's 500+ hosted open models by changing a single `.env` line — no code changes required

---

## How The Pipeline Works

```
Your Documents  (PDF / DOCX / TXT / MD / CSV / XLSX / HTML)
        │
        ▼
  ┌─────────────────────────────────┐
  │   Document Chunker              │  Splits content into 1500-char
  │   RecursiveCharacterTextSplitter│  overlapping segments (200-char overlap)
  └─────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────┐
  │   Local Embedding Engine        │  sentence-transformers/all-MiniLM-L6-v2
  │   (runs on CPU, fully offline)  │  Produces normalized 384-dim vectors
  └─────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────┐
  │   ChromaDB Vector Store         │  Persists embeddings locally
  │   (local, no cloud dependency)  │  Fast cosine similarity lookup
  └─────────────────────────────────┘
        │
   User asks a question
        │
        ▼
  ┌─────────────────────────────────┐
  │   Query Embedder                │  Same MiniLM model embeds the question
  └─────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────┐
  │   Semantic Retriever            │  Top-5 chunks by similarity score
  │   (similarity threshold search) │  injected directly into LLM prompt
  └─────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────┐
  │   Featherless AI LLM            │  Generates grounded answer strictly
  │   (streaming, OpenAI-compatible)│  from retrieved document context
  └─────────────────────────────────┘
        │
        ▼
   Streamed answer rendered live in the Gradio UI
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM API** | [Featherless AI](https://featherless.ai) | OpenAI-compatible inference, 500+ open models |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | Local CPU embedding, zero API cost |
| **Vector Database** | [ChromaDB](https://trychroma.com) | Local semantic similarity search |
| **Orchestration** | [LangChain](https://langchain.com) | Document loaders, text splitters, retriever chain |
| **UI** | [Gradio 6](https://gradio.app) + Custom CSS | Streaming chat interface with dark premium theme |
| **PDF Parser** | `PyPDFLoader` | Page-by-page PDF text extraction |
| **DOCX Parser** | `Docx2txtLoader` | Word document content extraction |
| **Sheet Parser** | `UnstructuredExcelLoader` | CSV and Excel structured data ingestion |

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/rag-chatbot.git
cd rag-chatbot
```

```bash
# Create virtual environment using Python 3.11
py -3.11 -m venv venv

# Activate — Windows
venv\Scripts\activate

# Activate — Mac / Linux
source venv/bin/activate
```

```bash
# Install all dependencies
pip install -r requirements.txt
```

```bash
# Add your API key to .env
# Get a free key at https://featherless.ai
FEATHERLESS_API_KEY=your_key_here
FEATHERLESS_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
```

```bash
# Launch the app
python chatbot.py
```

Open your browser at: `http://localhost:7860`

---

## Usage

**Step 1** — Go to the **Upload Documents** tab
**Step 2** — Drag and drop any supported file (PDF, DOCX, TXT, MD, CSV, XLSX, HTML)
**Step 3** — Click **Index Documents** and wait for the confirmation message
**Step 4** — Switch to the **Chat** tab
**Step 5** — Ask anything — answers come strictly from your uploaded content

---


## What Makes This Special

- **True Semantic Retrieval** — Unlike keyword search that matches exact strings, this system understands meaning. Asking "what are his qualifications?" correctly retrieves content about "education and skills" even without those exact words appearing in the question
- **Character-Aware Chunking** — Chunks are sized at 1500 characters (not words) to map safely within MiniLM's 512-token hard limit, preventing silent text truncation that destroys retrieval accuracy
- **Mathematically Optimized Similarity** — Embeddings are produced with `normalize_embeddings=True`, making all vectors unit-length. This reduces cosine similarity computation to a pure dot product, eliminating redundant square-root magnitude calculations
- **Fully Local Embedding Layer** — The embedding model downloads once and runs entirely on CPU thereafter. No embedding API calls, no per-token costs, no data leaving your machine during indexing
- **Production-Ready Codebase** — Modular separation between ingestion logic, retrieval logic, LLM calls, and UI rendering. Clean `.gitignore` rules protecting `.env` credentials from accidental exposure

---

## Project Structure

```
rag-chatbot/
├── chatbot.py              ← Main application — run this
├── ingest_database.py      ← Optional CLI bulk ingestion script
├── requirements.txt        ← All Python dependencies
├── .env                    ← API key configuration (never committed)
├── .gitignore              ← Protects credentials and local data
├── README.md
├── data/                   ← Uploaded documents stored here
└── chroma_db/              ← Auto-generated local vector database
```

---

## Future Roadmap

- **Conversational Memory** — Maintaining multi-turn context across a full session so follow-up questions reference previous answers naturally
- **Source Citations** — Displaying the exact document name and page number alongside every answer so users can verify retrieved content directly
- **Re-ranking Layer** — Adding a cross-encoder reranker pass after initial retrieval to further improve chunk relevance precision before LLM injection
- **Multi-User Sessions** — Isolated per-user vector collections enabling separate knowledge bases for different users on the same deployment
- **REST API Layer** — Exposing a FastAPI endpoint so external applications can query the RAG pipeline programmatically without the Gradio UI

---

Built for AI/ML internship portfolio · MIT License
