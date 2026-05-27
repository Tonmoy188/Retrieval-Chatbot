"""
chatbot.py  —  RAG Chatbot, fully working
"""

import os
import shutil
from pathlib import Path
from uuid import uuid4

from openai import OpenAI
from dotenv import load_dotenv
import gradio as gr
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH           = "data"
CHROMA_PATH         = "chroma_db"
FEATHERLESS_API_KEY = os.getenv("FEATHERLESS_API_KEY", "")
FEATHERLESS_MODEL   = os.getenv("FEATHERLESS_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
NUM_RESULTS         = 5

if not FEATHERLESS_API_KEY:
    raise ValueError("FEATHERLESS_API_KEY not set in .env file.")

llm_client = OpenAI(
    api_key=FEATHERLESS_API_KEY,
    base_url="https://api.featherless.ai/v1",
)

# ── Direct SentenceTransformer wrapper (bypasses langchain_huggingface bug) ───
print("Loading embedding model...")
_st_model = SentenceTransformer("all-MiniLM-L6-v2")

def _to_str(val) -> str:
    """Safely extract a plain string from whatever Gradio passes in."""
    if isinstance(val, str):
        return val
    if isinstance(val, dict):
        return val.get("text") or val.get("content") or str(val)
    if isinstance(val, list):
        parts = []
        for item in val:
            parts.append(_to_str(item))
        return " ".join(parts)
    return str(val)

class DirectEmbeddings(Embeddings):
    """Wraps SentenceTransformer directly — avoids langchain_huggingface bugs."""
    def embed_documents(self, texts):
        clean = [_to_str(t) for t in texts]
        return _st_model.encode(clean, normalize_embeddings=True).tolist()
    def embed_query(self, text):
        clean = _to_str(text)
        return _st_model.encode(clean, normalize_embeddings=True).tolist()

embeddings_model = DirectEmbeddings()

# ── Vector store ───────────────────────────────────────────────────────────────
vector_store = Chroma(
    collection_name="rag_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)
retriever = vector_store.as_retriever(search_kwargs={"k": NUM_RESULTS})

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200,
    length_function=len, is_separator_regex=False,
)

SUPPORTED = {".pdf", ".docx", ".doc", ".txt", ".md", ".csv"}

def load_file(path: str):
    ext = Path(path).suffix.lower()
    if ext == ".pdf":              return PyPDFLoader(path).load()
    elif ext in (".docx", ".doc"): return Docx2txtLoader(path).load()
    elif ext in (".txt", ".md"):   return TextLoader(path, encoding="utf-8").load()
    elif ext == ".csv":            return CSVLoader(path, encoding="utf-8").load()
    return []

# ── Upload handler ─────────────────────────────────────────────────────────────
def upload_and_ingest(files):
    if not files:
        return "No files selected."
    Path(DATA_PATH).mkdir(exist_ok=True)
    lines = []
    for f in files:
        src = Path(f.name)
        if src.suffix.lower() not in SUPPORTED:
            lines.append(f"⚠ Skipped {src.name} — unsupported format")
            continue
        try:
            dest = Path(DATA_PATH) / src.name
            shutil.copy(f.name, dest)
            docs = load_file(str(dest))
            if not docs:
                lines.append(f"⚠ No text found in {src.name}")
                continue
            chunks = text_splitter.split_documents(docs)
            uuids  = [str(uuid4()) for _ in chunks]
            vector_store.add_documents(documents=chunks, ids=uuids)
            lines.append(f"✓ {src.name}  →  {len(chunks)} chunks indexed")
        except Exception as e:
            lines.append(f"✗ {src.name}: {e}")
    return "\n".join(lines)

# ── RAG stream ─────────────────────────────────────────────────────────────────
def stream_response(message, history):
    docs = retriever.invoke(message)
    knowledge = "\n\n".join(doc.page_content for doc in docs)

    rag_prompt = f"""You are an assistant which answers questions based on knowledge which is provided to you.
While answering, you don't use your internal knowledge,
but solely the information in the "The knowledge" section.
You don't mention anything to the user about the provided knowledge.

The question: {message}

Conversation history: {history}

The knowledge: {knowledge}
"""
    partial_message = ""
    stream = llm_client.chat.completions.create(
        model=FEATHERLESS_MODEL,
        messages=[{"role": "user", "content": rag_prompt}],
        stream=True,
        temperature=0.5,
        max_tokens=1024,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta is None:
            continue
        partial_message += delta
        yield partial_message

# ── Chat handlers ──────────────────────────────────────────────────────────────
def user_submit(user_message, history):
    if not user_message.strip():
        return "", history or []
    history = history or []
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": "▋"})
    return "", history

def bot_respond(history):
    if not history or len(history) < 2:
        yield history
        return
    user_message = history[-2]["content"]
    history[-1]["content"] = ""
    for partial in stream_response(user_message, history[:-2]):
        history[-1]["content"] = partial + " ▋"
        yield history
    history[-1]["content"] = history[-1]["content"].replace(" ▋", "").replace("▋", "")
    yield history

# ── CSS ────────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container {
    background: #06060a !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    color: rgba(255,255,255,0.92) !important;
    min-height: 100vh;
}
.gradio-container::before {
    content: ''; position: fixed; top: -250px; left: -250px;
    width: 650px; height: 650px; border-radius: 50%;
    background: radial-gradient(circle, rgba(16,185,129,0.13) 0%, transparent 65%);
    animation: orb1 18s ease-in-out infinite;
    pointer-events: none; z-index: 0;
}
.gradio-container::after {
    content: ''; position: fixed; bottom: -220px; right: -220px;
    width: 750px; height: 750px; border-radius: 50%;
    background: radial-gradient(circle, rgba(99,102,241,0.13) 0%, transparent 65%);
    animation: orb2 24s ease-in-out infinite;
    pointer-events: none; z-index: 0;
}
@keyframes orb1 {
    0%,100%{transform:translate(0,0) scale(1);}
    33%{transform:translate(50px,-40px) scale(1.07);}
    66%{transform:translate(-25px,50px) scale(0.93);}
}
@keyframes orb2 {
    0%,100%{transform:translate(0,0) scale(1);}
    50%{transform:translate(-45px,55px) scale(1.12);}
}
.app-header {
    text-align: center; padding: 40px 20px 16px;
    position: relative; z-index: 1;
}
.app-header h1 {
    font-size: 2.1rem; font-weight: 700;
    letter-spacing: -0.03em; margin: 0 0 8px;
    background: linear-gradient(135deg, #ffffff 0%, rgba(255,255,255,0.45) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.app-header p { color: rgba(255,255,255,0.35); font-size: 0.9rem; margin: 0; font-weight: 300; }
.badge {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 5px 14px; margin-top: 12px; border-radius: 20px;
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    background: rgba(99,102,241,0.10); border: 1px solid rgba(99,102,241,0.22);
    color: rgba(167,139,250,0.8);
}
.tab-nav button {
    background: transparent !important; border: none !important;
    color: rgba(255,255,255,0.4) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important; font-weight: 500 !important;
    padding: 10px 20px !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s !important;
}
.tab-nav button.selected {
    color: rgba(255,255,255,0.92) !important;
    border-bottom: 2px solid #10b981 !important;
}
#chatbox {
    background: rgba(255,255,255,0.018) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 18px !important;
}
#msg-input textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 14px !important;
    color: rgba(255,255,255,0.92) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.9rem !important; padding: 12px 16px !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    resize: none !important;
}
#msg-input textarea:focus {
    border-color: rgba(16,185,129,0.35) !important;
    box-shadow: 0 0 0 3px rgba(16,185,129,0.06) !important;
    outline: none !important;
}
#msg-input textarea::placeholder { color: rgba(255,255,255,0.25) !important; }
#send-btn {
    background: linear-gradient(135deg, rgba(16,185,129,0.25), rgba(16,185,129,0.15)) !important;
    border: 1px solid rgba(16,185,129,0.35) !important;
    border-radius: 12px !important; color: #34d399 !important;
    font-weight: 600 !important; transition: all 0.2s !important; min-width: 80px !important;
}
#send-btn:hover {
    background: linear-gradient(135deg, rgba(16,185,129,0.35), rgba(16,185,129,0.25)) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(16,185,129,0.15) !important;
}
#clear-btn {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important; color: rgba(255,255,255,0.35) !important;
    font-weight: 500 !important; transition: all 0.2s !important; min-width: 70px !important;
}
#clear-btn:hover {
    color: rgba(255,255,255,0.75) !important;
    border-color: rgba(255,255,255,0.15) !important;
    background: rgba(255,255,255,0.06) !important;
}
#upload-box {
    background: rgba(255,255,255,0.02) !important;
    border: 1.5px dashed rgba(255,255,255,0.10) !important;
    border-radius: 16px !important; transition: border-color 0.2s !important;
}
#upload-box:hover { border-color: rgba(99,102,241,0.35) !important; }
#ingest-btn {
    background: linear-gradient(135deg, rgba(99,102,241,0.22), rgba(99,102,241,0.12)) !important;
    border: 1px solid rgba(99,102,241,0.32) !important;
    border-radius: 12px !important; color: rgba(167,139,250,0.92) !important;
    font-weight: 600 !important; transition: all 0.2s !important;
}
#ingest-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(99,102,241,0.15) !important;
}
#result-box {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 12px !important; color: rgba(255,255,255,0.60) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important; line-height: 1.7 !important;
}
.info-bar {
    display: flex; justify-content: center;
    gap: 12px; padding: 12px 0 4px; flex-wrap: wrap;
}
.pill {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 11px; color: rgba(255,255,255,0.32);
    padding: 4px 12px; background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06); border-radius: 20px;
}
.dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #10b981; display: inline-block;
    animation: pulse 2.2s ease-in-out infinite;
}
@keyframes pulse {
    0%,100%{opacity:1;transform:scale(1);}
    50%{opacity:0.3;transform:scale(0.75);}
}
footer { display: none !important; }
"""

# ── UI ─────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="RAG Chatbot") as app:

    gr.HTML(f"""
    <div class="app-header">
        <h1>RAG Knowledge Assistant</h1>
        <p>Upload your documents · Ask questions · Answers come only from your files</p>
        <div class="badge">{FEATHERLESS_MODEL.split('/')[-1]} &nbsp;·&nbsp; ChromaDB &nbsp;·&nbsp; MiniLM-L6-v2</div>
    </div>
    """)

    with gr.Tabs():

        with gr.Tab("💬  Chat"):
            chatbot_box = gr.Chatbot(
                elem_id="chatbox",
                height=480,
                show_label=False,
                render_markdown=True,
                avatar_images=(None, None),
            )
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask anything about your uploaded documents...",
                    show_label=False, container=False,
                    scale=7, elem_id="msg-input",
                    lines=1, max_lines=5, autofocus=True,
                )
                send_btn  = gr.Button("Send",  scale=1, elem_id="send-btn",  variant="primary", min_width=80)
                clear_btn = gr.Button("Clear", scale=0, elem_id="clear-btn", min_width=70)

            gr.HTML("""
            <div class="info-bar">
                <div class="pill"><span class="dot"></span>&nbsp;RAG active</div>
                <div class="pill">Enter to send &nbsp;·&nbsp; Shift+Enter for new line</div>
                <div class="pill">Powered by Featherless AI</div>
            </div>
            """)

        with gr.Tab("📁  Upload Documents"):
            gr.HTML("""
            <div style="text-align:center;padding:22px 0 10px;color:rgba(255,255,255,0.38);font-size:0.875rem;line-height:1.8">
                Drop your files below and click <b style="color:rgba(255,255,255,0.7)">Index Documents</b>.<br>
                They are chunked and added to the knowledge base immediately.<br>
                <span style="color:rgba(255,255,255,0.55);font-weight:500">PDF &nbsp;·&nbsp; DOCX &nbsp;·&nbsp; TXT &nbsp;·&nbsp; MD &nbsp;·&nbsp; CSV</span>
            </div>
            """)
            upload_box = gr.File(
                label="Drop files here or click to browse",
                file_count="multiple", elem_id="upload-box",
                file_types=[".pdf",".docx",".doc",".txt",".md",".csv"],
            )
            ingest_btn = gr.Button("⚡  Index Documents", elem_id="ingest-btn", variant="primary")
            result_box = gr.Textbox(
                label="Indexing Result", interactive=False,
                elem_id="result-box", lines=5,
                placeholder="Results will appear here after indexing...",
            )
            ingest_btn.click(fn=upload_and_ingest, inputs=[upload_box], outputs=[result_box])

    msg_input.submit(user_submit, [msg_input, chatbot_box], [msg_input, chatbot_box], queue=False).then(
                     bot_respond,  [chatbot_box], [chatbot_box])
    send_btn.click(  user_submit, [msg_input, chatbot_box], [msg_input, chatbot_box], queue=False).then(
                     bot_respond,  [chatbot_box], [chatbot_box])
    clear_btn.click(lambda: [], outputs=[chatbot_box])

if __name__ == "__main__":
    print(f"\n🚀 RAG Chatbot starting...")
    print(f"   Model : {FEATHERLESS_MODEL}")
    print(f"   DB    : {CHROMA_PATH}/\n")
    app.queue()
    app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)), inbrowser=False, css=CSS)
