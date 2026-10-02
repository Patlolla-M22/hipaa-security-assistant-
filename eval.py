import json
import time
from graph import run

# (question, should_answer, keywords: at least one must appear in the answer)
TESTS = [
    ("What are the phases of the incident response life cycle?", True, ["preparation", "containment"]),
    ("What should an incident response policy include?", True, ["management", "metrics", "scope"]),
    ("What are the recoverability effort categories?", True, ["supplemented", "extended"]),
    ("What is a containment strategy and why is it important?", True, ["resources", "damage"]),
    ("What is the purpose of a lessons learned meeting after an incident?", True, ["lessons learned"]),
    ("What is the difference between a precursor and an indicator?", True, ["precursor", "indicator"]),
    ("How soon must a business associate notify the covered entity of a breach?", True, ["60"]),
    ("What factors are used in a breach risk assessment?", True, ["nature and extent", "acquired or viewed", "mitigated"]),
    ("Who must be notified after a breach of unsecured protected health information?", True, ["individual", "secretary"]),
    ("What are the categories of HIPAA Security Rule safeguards?", True, ["administrative", "physical", "technical"]),
    ("What is the capital of France?", False, []),
    ("What are the penalty amounts for GDPR violations?", False, []),
    ("How do I configure a Cisco firewall?", False, []),
    ("Who won the 2022 World Cup?", False, []),
    ("What does PCI DSS require for storing card data?", False, []),
]

PRICE_IN = 1.00 / 1_000_000   # dollars per input token (check console.anthropic.com pricing)
PRICE_OUT = 5.00 / 1_000_000  # dollars per output token


def refused(text):
    t = text.lower().replace("\u2019", "'")
    return "have that in the documents" in t


def main():
    results = []
    for q, should_answer, keywords in TESTS:
        start = time.time()
        out = run(q)
        latency = time.time() - start
        text = out["answer"]
        low = text.lower()
        if should_answer:
            ok = (not refused(text)) and any(k in low for k in keywords) and "[" in text
        else:
            ok = refused(text)
        cost = out["tokens_in"] * PRICE_IN + out["tokens_out"] * PRICE_OUT
        weak = sum(1 for s in out["trace"] if s.startswith("grade: weak"))
        results.append({
            "question": q, "should_answer": should_answer, "pass": ok,
            "latency": round(latency, 2), "tokens_in": out["tokens_in"],
            "tokens_out": out["tokens_out"], "cost": cost, "weak_grades": weak,
            "answer": text,
        })
        print(("PASS" if ok else "FAIL"), f"{latency:5.1f}s", q[:70])

    ans = [r for r in results if r["should_answer"]]
    ref = [r for r in results if not r["should_answer"]]
    n = len(results)
    print("\n===== SUMMARY =====")
    print(f"Answerable questions correct: {sum(r['pass'] for r in ans)}/{len(ans)}")
    print(f"Out-of-scope correctly refused: {sum(r['pass'] for r in ref)}/{len(ref)}")
    print(f"Avg latency: {sum(r['latency'] for r in results) / n:.2f}s")
    print(f"Avg tokens: {sum(r['tokens_in'] for r in results) / n:.0f} in / {sum(r['tokens_out'] for r in results) / n:.0f} out")
    print(f"Avg cost per query: ${sum(r['cost'] for r in results) / n:.5f}")
    print(f"Queries that needed a retry: {sum(1 for r in results if r['weak_grades'] > 0)}/{n}")
    with open("eval_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
