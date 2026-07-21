"""
Ingest raw documents into ChromaDB vector store using OpenAI embeddings.
Run this once (or re-run to refresh the index).

Uses a custom text splitter to avoid langchain_text_splitters pulling in
transformers/PyTorch, which causes initialization deadlocks on some systems.
"""
import os
import re
import json
import shutil
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

RAW_FILE = "data/raw_documents.json"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "bsu_sfs"


# ── Minimal text splitter (no heavy ML deps) ─────────────────────────────────
def _split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> list[str]:
    """
    Split text into overlapping chunks using a priority list of separators.
    Prefers splitting on paragraph breaks, then newlines, then sentences, then words.
    """
    separators = ["\n\n", "\n", ". ", " "]

    def _merge_splits(splits: list[str], sep: str) -> list[str]:
        chunks: list[str] = []
        current_parts: list[str] = []
        current_len = 0

        for s in splits:
            s_len = len(s)
            add_len = s_len + (len(sep) if current_parts else 0)
            if current_len + add_len > chunk_size and current_parts:
                chunk = sep.join(current_parts).strip()
                if chunk:
                    chunks.append(chunk)
                # Keep overlap: drop parts from the front until we're within overlap
                while current_parts and current_len > chunk_overlap:
                    removed = current_parts.pop(0)
                    current_len -= len(removed) + len(sep)
                current_len = max(0, current_len)
            current_parts.append(s)
            current_len += add_len

        if current_parts:
            chunk = sep.join(current_parts).strip()
            if chunk:
                chunks.append(chunk)
        return chunks

    def _recursive_split(t: str, seps: list[str]) -> list[str]:
        if not t.strip():
            return []
        if len(t) <= chunk_size:
            return [t.strip()]
        if not seps:
            # Hard cut at chunk_size
            return [t[i:i + chunk_size] for i in range(0, len(t), chunk_size - chunk_overlap)]

        sep = seps[0]
        parts = t.split(sep)
        good: list[str] = []
        bad: list[str] = []
        for p in parts:
            if len(p) > chunk_size:
                bad.append(p)
            else:
                good.append(p)

        # Sub-split any oversized parts
        final_parts: list[str] = []
        for p in parts:
            if len(p) > chunk_size:
                final_parts.extend(_recursive_split(p, seps[1:]))
            else:
                final_parts.append(p)

        return _merge_splits(final_parts, sep)

    return _recursive_split(text, separators)


def load_raw_documents() -> list[Document]:
    if not os.path.exists(RAW_FILE):
        raise FileNotFoundError(
            f"{RAW_FILE} not found. Run `python crawler.py` first."
        )
    with open(RAW_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    docs = []
    for item in raw:
        docs.append(
            Document(
                page_content=item["content"],
                metadata={"source": item["url"]},
            )
        )
    print(f"Loaded {len(docs)} raw documents")
    return docs


def split_documents(docs: list[Document], chunk_size: int = 800, chunk_overlap: int = 100) -> list[Document]:
    chunks = []
    for doc in docs:
        for chunk_text in _split_text(doc.page_content, chunk_size, chunk_overlap):
            chunks.append(Document(page_content=chunk_text, metadata=doc.metadata))
    print(f"Split into {len(chunks)} chunks")
    return chunks


def build_vectorstore(chunks: list[Document]) -> Chroma:
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
    )
    print(f"Vector store built at '{CHROMA_DIR}/' with {len(chunks)} vectors")
    return vectorstore


def main():
    docs = load_raw_documents()
    chunks = split_documents(docs)
    build_vectorstore(chunks)
    print("\nIngestion complete. Run: streamlit run app.py")


if __name__ == "__main__":
    main()
