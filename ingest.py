from pathlib import Path
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer

DATA_DIR = Path("data")
CHUNK_WORDS = 250
OVERLAP = 50


def chunk_text(text, size=CHUNK_WORDS, overlap=OVERLAP):
    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if len(piece) >= 30:
            chunks.append(" ".join(piece))
    return chunks


def main():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path="chroma_db")
    try:
        client.delete_collection("policies")
    except Exception:
        pass
    col = client.create_collection("policies", metadata={"hnsw:space": "cosine"})

    total = 0
    for pdf in sorted(DATA_DIR.glob("*.pdf")):
        reader = PdfReader(str(pdf))
        ids, docs, metas = [], [], []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            for i, chunk in enumerate(chunk_text(text)):
                ids.append(f"{pdf.stem}-p{page_num}-c{i}")
                docs.append(chunk)
                metas.append({"source": pdf.name, "page": page_num})
        if docs:
            embeddings = model.encode(docs).tolist()
            col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
        print(f"{pdf.name}: {len(docs)} chunks")
        total += len(docs)
    print(f"Done. {total} chunks stored.")


if __name__ == "__main__":
    main()
