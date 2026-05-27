# ⟡ RAG Knowledge Chatbot

A clean Python RAG chatbot — ask questions, get answers from **your own documents**.  
Built following Tom's Tech Academy architecture, powered by **Featherless AI** (free tier).

---

## How it works

```
Your documents (PDF/TXT/MD/DOCX)
        ↓  ingest_database.py
    Split into chunks (1500 chars)
        ↓
    Embed with MiniLM-L6-v2 (local, free)
        ↓
    Store in ChromaDB
        ↓  chatbot.py
    User asks a question
        ↓
    Retrieve top-5 relevant chunks
        ↓
    Send chunks + question to Featherless AI
        ↓
    Stream the answer
```

---

## Setup (5 minutes)

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Get your Featherless API key

1. Go to [featherless.ai](https://featherless.ai) → Sign up (free)
2. Copy your API key from the dashboard

### 3. Set your API key

Open `.env` and paste your key:

```env
FEATHERLESS_API_KEY=your_actual_key_here
FEATHERLESS_MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
```

**Model recommendations:**
| Model | Speed | Quality |
|---|---|---|
| `meta-llama/Meta-Llama-3.1-8B-Instruct` | Fast | Good — use for testing |
| `meta-llama/Meta-Llama-3.1-70B-Instruct` | Moderate | Best — use for demo |
| `Qwen/Qwen2.5-7B-Instruct` | Fast | Very good for RAG |

### 4. Add your documents

Drop any of these into the `data/` folder:
- `.pdf` — PDFs (resumes, papers, books)
- `.txt` — plain text files
- `.md` — markdown files
- `.docx` — Word documents

### 5. Ingest (run once)

```bash
python ingest_database.py
```

This embeds your documents and saves them to `chroma_db/`.  
You only need to run this again if you add new documents.

### 6. Chat!

```bash
python chatbot.py
```

Opens at **http://localhost:7860**

---

## Project structure

```
rag-chatbot/
├── data/                  ← Drop your documents here
├── chroma_db/             ← Auto-created by ingest_database.py
├── ingest_database.py     ← Run once to index documents
├── chatbot.py             ← The chatbot (run anytime)
├── requirements.txt
├── .env                   ← Your API key goes here
└── README.md
```

---

## Re-ingesting documents

If you add new documents, run ingest again:

```bash
# Optional: wipe old DB first (recommended if replacing documents)
rm -rf chroma_db/

python ingest_database.py
python chatbot.py
```

On Windows:
```
rmdir /s /q chroma_db
python ingest_database.py
```

---

## Tech stack

| Component | Technology |
|---|---|
| LLM API | Featherless AI (OpenAI-compatible) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, free) |
| Vector DB | ChromaDB (local) |
| Orchestration | LangChain |
| UI | Gradio with custom CSS |
| PDF parsing | PyPDFLoader (LangChain) |
| DOCX parsing | Docx2txtLoader (LangChain) |

---

Built for AI/ML internship portfolio · MIT License
