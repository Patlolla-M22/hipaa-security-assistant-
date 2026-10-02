import time
import uuid
from fastapi import FastAPI
from pydantic import BaseModel

from graph import run

app = FastAPI(title="HIPAA & Security Policy Assistant")

sessions = {}
MAX_TURNS = 6


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

    out = run(req.question, history[-MAX_TURNS * 2:])

    history.append({"role": "user", "content": req.question})
    history.append({"role": "assistant", "content": out["answer"]})

    return {
        "session_id": sid,
        "answer": out["answer"],
        "sources": [{"source": h["source"], "page": h["page"]} for h in out["hits"]],
        "trace": out["trace"],
        "latency_seconds": round(time.time() - start, 2),
        "tokens": {"input": out["tokens_in"], "output": out["tokens_out"]},
    }
