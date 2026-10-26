import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from retriever import get_retriever

QUESTIONS = [
    "What is the maximum height before a guardrail is mandatory on a scaffold?",
    "What must a worker do in the first 24 hours after an incident?",
    "Which PPE is required for concrete cutting?",
]

for q in QUESTIONS:
    print("\n" + "=" * 80)
    print("Q:", q)
    for search_type in ("similarity", "mmr"):
        print(f"\n--- {search_type.upper()} ---")
        for d in get_retriever(search_type).invoke(q):
            snippet = d.page_content[:90].replace("\n", " ")
            print(f"[{d.metadata['source']}, p.{d.metadata['page']}] {snippet}...")