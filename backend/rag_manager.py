"""Document layer: PDF text extraction, page-aware chunking and semantic search with ChromaDB."""
from __future__ import annotations

import io
import re
from pathlib import Path

import chromadb
from chromadb.config import Settings
from pypdf import PdfReader

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHROMA_DIR = DATA_DIR / "chroma"
COLLECTION = "documents"


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    """Split text into ~size-char chunks, preferring sentence boundaries, with overlap."""
    text = re.sub(r"\s+", " ", text or "").strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = text.rfind(". ", start + size // 2, end)
            if cut != -1:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


class RAGManager:
    def __init__(self, path: Path = CHROMA_DIR):
        path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))
        self.collection = self._get_collection()

    def _get_collection(self):
        return self.client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    def index_pdf(self, data: bytes, name: str) -> int:
        """Index a PDF (bytes). Re-indexing the same file name replaces old chunks. Returns chunk count."""
        reader = PdfReader(io.BytesIO(data))
        ids, docs, metas = [], [], []
        for page_no, page in enumerate(reader.pages, start=1):
            for i, chunk in enumerate(chunk_text(page.extract_text() or "")):
                ids.append(f"{name}::p{page_no}::c{i}")
                docs.append(chunk)
                metas.append({"source": name, "page": page_no, "chunk": i})
        self.collection.delete(where={"source": name})
        for b in range(0, len(docs), 100):
            self.collection.upsert(ids=ids[b:b + 100], documents=docs[b:b + 100], metadatas=metas[b:b + 100])
        return len(docs)

    def search(self, query: str, k: int = 4) -> list[dict]:
        count = self.collection.count()
        if count == 0:
            return []
        res = self.collection.query(query_texts=[query], n_results=min(k, count),
                                    include=["documents", "metadatas", "distances"])
        return [
            {"source": m["source"], "page": int(m["page"]), "text": d, "score": round(1 - dist, 3)}
            for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    def list_documents(self) -> list[dict]:
        stats: dict[str, dict] = {}
        for m in self.collection.get(include=["metadatas"])["metadatas"] or []:
            s = stats.setdefault(m["source"], {"pages": set(), "chunks": 0})
            s["pages"].add(m["page"])
            s["chunks"] += 1
        return [{"name": n, "pages": len(s["pages"]), "chunks": s["chunks"]} for n, s in sorted(stats.items())]

    def delete_document(self, name: str) -> None:
        self.collection.delete(where={"source": name})

    def reset(self) -> None:
        try:
            self.client.delete_collection(COLLECTION)
        except Exception:
            pass
        self.collection = self._get_collection()
