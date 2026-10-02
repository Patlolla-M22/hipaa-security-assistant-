import sys
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from answer import retrieve, client, MODEL, SYSTEM

MAX_ATTEMPTS = 2


class State(TypedDict, total=False):
    question: str
    history: list
    query: str
    hits: list
    good: bool
    attempts: int
    answer: str
    tokens_in: int
    tokens_out: int
    trace: list


def ask(prompt, max_tokens=80):
    msg = client.messages.create(
        model=MODEL, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text.strip(), msg.usage


def add_usage(state, usage):
    return {
        "tokens_in": state.get("tokens_in", 0) + usage.input_tokens,
        "tokens_out": state.get("tokens_out", 0) + usage.output_tokens,
    }


def rewrite(state):
    attempts = state.get("attempts", 0)
    history = state.get("history", [])
    trace = list(state.get("trace", []))
    if attempts == 0 and not history:
        trace.append("rewrite: first question, used as-is")
        return {"query": state["question"], "trace": trace}
    if attempts == 0:
        hist = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in history[-4:])
        prompt = (
            f"Conversation so far:\n{hist}\n\n"
            f"Rewrite the last question as a standalone search query. "
            f"Output only the query.\nQuestion: {state['question']}"
        )
    else:
        prompt = (
            f"The search query '{state['query']}' returned poor results. "
            f"Write a different, broader search query for this question. "
            f"Output only the query.\nQuestion: {state['question']}"
        )
    query, usage = ask(prompt)
    trace.append(f"rewrite: {query}")
    return {"query": query, "trace": trace, **add_usage(state, usage)}


def retrieve_node(state):
    hits = retrieve(state["query"])
    trace = list(state.get("trace", []))
    trace.append(f"retrieve: {len(hits)} passages")
    return {"hits": hits, "trace": trace}


def grade(state):
    passages = "\n\n".join(h["text"][:500] for h in state["hits"])
    prompt = (
        f"Question: {state['question']}\n\nPassages:\n{passages}\n\n"
        "Do these passages contain enough information to answer the question? "
        "Reply with only YES or NO."
    )
    verdict, usage = ask(prompt, max_tokens=5)
    good = verdict.upper().startswith("YES")
    trace = list(state.get("trace", []))
    trace.append(f"grade: {'good' if good else 'weak'}")
    return {
        "good": good,
        "attempts": state.get("attempts", 0) + 1,
        "trace": trace,
        **add_usage(state, usage),
    }


def route(state):
    if state["good"] or state["attempts"] >= MAX_ATTEMPTS:
        return "answer"
    return "retry"


def answer_node(state):
    hits = state["hits"]
    context = "\n\n".join(
        f"[{i + 1}] ({h['source']} p.{h['page']})\n{h['text']}" for i, h in enumerate(hits)
    )
    messages = state.get("history", [])[-12:] + [
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {state['question']}"}
    ]
    msg = client.messages.create(
        model=MODEL, max_tokens=600, system=SYSTEM, messages=messages
    )
    trace = list(state.get("trace", []))
    trace.append("answer: done")
    return {"answer": msg.content[0].text, "trace": trace, **add_usage(state, msg.usage)}


builder = StateGraph(State)
builder.add_node("rewrite", rewrite)
builder.add_node("retrieve", retrieve_node)
builder.add_node("grade", grade)
builder.add_node("answer", answer_node)
builder.add_edge(START, "rewrite")
builder.add_edge("rewrite", "retrieve")
builder.add_edge("retrieve", "grade")
builder.add_conditional_edges("grade", route, {"answer": "answer", "retry": "rewrite"})
builder.add_edge("answer", END)
graph = builder.compile()


def run(question, history=None):
    return graph.invoke(
        {"question": question, "history": history or [], "attempts": 0, "trace": []}
    )


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "What are the phases of the incident response life cycle?"
    out = run(q)
    print("\nTRACE:")
    for step in out["trace"]:
        print(" -", step)
    print("\nANSWER:\n" + out["answer"])
    print(f"\nTokens: {out['tokens_in']} in / {out['tokens_out']} out")
