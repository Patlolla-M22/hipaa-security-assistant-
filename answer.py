import sys
from dotenv import load_dotenv
import anthropic
import chromadb
from sentence_transformers import SentenceTransformer

load_dotenv()
MODEL = "claude-haiku-4-5-20251001"

embedder = SentenceTransformer("all-MiniLM-L6-v2")
col = chromadb.PersistentClient(path="chroma_db").get_collection("policies")
client = anthropic.Anthropic()

SYSTEM = (
    "You answer questions about HIPAA and security incident handling using ONLY "
    "the numbered context passages provided. Cite sources like [1] or [2] after "
    "the claims they support. If the passages do not contain the answer, say "
    "'I don't have that in the documents.' Do not use outside knowledge."
)


def retrieve(question, k=4):
    res = col.query(query_embeddings=embedder.encode([question]).tolist(), n_results=k)
    return [
        {"text": d, "source": m["source"], "page": m["page"]}
        for d, m in zip(res["documents"][0], res["metadatas"][0])
    ]


def answer(question):
    hits = retrieve(question)
    context = "\n\n".join(
        f"[{i + 1}] ({h['source']} p.{h['page']})\n{h['text']}" for i, h in enumerate(hits)
    )
    msg = client.messages.create(
        model=MODEL,
        max_tokens=600,
        system=SYSTEM,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
    )
    return msg.content[0].text, hits, msg.usage


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What are the phases of the incident response life cycle?"
    text, hits, usage = answer(q)
    print("\nANSWER:\n" + text)
    print("\nSOURCES:")
    for i, h in enumerate(hits, 1):
        print(f"[{i}] {h['source']} p.{h['page']}")
    print(f"\nTokens: {usage.input_tokens} in / {usage.output_tokens} out")
