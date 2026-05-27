"""
ingest_database.py  —  Run once to build the ChromaDB vector database
"""

from pathlib import Path
from uuid import uuid4

from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

DATA_PATH   = "data"
CHROMA_PATH = "chroma_db"
SUPPORTED   = {".pdf", ".docx", ".doc", ".txt", ".md", ".csv"}

print("Loading embedding model (downloads once, then cached)...")
_st_model = SentenceTransformer("all-MiniLM-L6-v2")

def _to_str(val) -> str:
    if isinstance(val, str): return val
    if isinstance(val, dict): return val.get("text") or val.get("content") or str(val)
    if isinstance(val, list): return " ".join(_to_str(i) for i in val)
    return str(val)

class DirectEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return _st_model.encode([_to_str(t) for t in texts], normalize_embeddings=True).tolist()
    def embed_query(self, text):
        return _st_model.encode(_to_str(text), normalize_embeddings=True).tolist()

vector_store = Chroma(
    collection_name="rag_collection",
    embedding_function=DirectEmbeddings(),
    persist_directory=CHROMA_PATH,
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200,
    length_function=len, is_separator_regex=False,
)

def load_file(path: str):
    ext = Path(path).suffix.lower()
    if ext == ".pdf":              return PyPDFLoader(path).load()
    elif ext in (".docx", ".doc"): return Docx2txtLoader(path).load()
    elif ext in (".txt", ".md"):   return TextLoader(path, encoding="utf-8").load()
    elif ext == ".csv":            return CSVLoader(path, encoding="utf-8").load()
    return []

data_dir = Path(DATA_PATH)
if not data_dir.exists():
    print(f"ERROR: '{DATA_PATH}/' folder not found. Create it and add your files.")
    exit(1)

files = [f for f in data_dir.iterdir() if f.suffix.lower() in SUPPORTED]
if not files:
    print(f"No supported files in '{DATA_PATH}/'. Add PDF, DOCX, TXT, MD, or CSV files.")
    exit(1)

print(f"\nFound {len(files)} file(s):\n")
all_chunks = []
for f in files:
    print(f"  Loading: {f.name}")
    docs = load_file(str(f))
    if not docs:
        print(f"  ⚠ No text from {f.name}")
        continue
    chunks = text_splitter.split_documents(docs)
    all_chunks.extend(chunks)
    print(f"  ✓ {len(chunks)} chunks created")

print(f"\nEmbedding and storing {len(all_chunks)} chunks in ChromaDB...")
uuids = [str(uuid4()) for _ in all_chunks]
vector_store.add_documents(documents=all_chunks, ids=uuids)
print(f"\n✅ Done! Database saved to '{CHROMA_PATH}/'")
print("Now run:  python chatbot.py")