import time
import uuid
from fastapi import FastAPI
from pydantic import BaseModel

from answer import retrieve, client, MODEL, SYSTEM

app = FastAPI(title="HIPAA & Security Policy Assistant")

sessions = {}      # session_id -> list of past messages
MAX_TURNS = 6      # how many past exchanges to remember


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    start = time.time()
    sid = req.session_id or str(uuid.uuid4())
    history = sessions.setdefault(sid, [])

    hits = retrieve(req.question)
    context = "\n\n".join(
        f"[{i + 1}] ({h['source']} p.{h['page']})\n{h['text']}" for i, h in enumerate(hits)
    )
    messages = history[-MAX_TURNS * 2:] + [
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.question}"}
    ]
    msg = client.messages.create(
        model=MODEL, max_tokens=600, system=SYSTEM, messages=messages
    )
    text = msg.content[0].text

    history.append({"role": "user", "content": req.question})
    history.append({"role": "assistant", "content": text})

    return {
        "session_id": sid,
        "answer": text,
        "sources": [{"source": h["source"], "page": h["page"]} for h in hits],
        "latency_seconds": round(time.time() - start, 2),
        "tokens": {"input": msg.usage.input_tokens, "output": msg.usage.output_tokens},
    }
