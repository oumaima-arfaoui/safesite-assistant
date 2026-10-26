import json
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from rag_chain import ask

REFUSAL = "I could not find this in the safety documents."

EVAL_DIR = Path(__file__).parent
questions = json.loads((EVAL_DIR / "questions.json").read_text(encoding="utf-8"))

results = []
for q in questions:
    start = time.perf_counter()
    answer, _ = ask(q["question"])
    latency_ms = int((time.perf_counter() - start) * 1000)
    results.append({**q, "answer": answer, "latency_ms": latency_ms})
    print(f"Q{q['id']} done ({latency_ms} ms)")

(EVAL_DIR / "results.json").write_text(
    json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
)

refusals = [r for r in results if not r["answerable"]]
correct_refusals = sum(REFUSAL in r["answer"] for r in refusals)
print(f"\nCorrect refusals: {correct_refusals}/{len(refusals)}")
print(f"Average latency: {sum(r['latency_ms'] for r in results) // len(results)} ms")