from pipeline.retrieval.retriever import retrieve
from pipeline.generation.generate import generate_answer

query = "who is eligible for PM Poshan"
chunks = retrieve(query, language="en", top_k=4)

result = generate_answer(query, chunks, language="en")

print("=== ANSWER ===")
print(result["answer"])
print("\n=== CITED CHUNKS ===")
for c in result["cited_chunks"][:2]:
    print(f"- {c['scheme_id']} | {c['section']}")