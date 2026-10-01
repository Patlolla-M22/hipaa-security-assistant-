import sys
import chromadb
from sentence_transformers import SentenceTransformer

question = " ".join(sys.argv[1:]) or "What is incident response?"
model = SentenceTransformer("all-MiniLM-L6-v2")
col = chromadb.PersistentClient(path="chroma_db").get_collection("policies")

result = col.query(query_embeddings=model.encode([question]).tolist(), n_results=4)
for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
    print(f"\n[{meta['source']} p.{meta['page']}]  distance={dist:.3f}")
    print(doc[:400], "...")
