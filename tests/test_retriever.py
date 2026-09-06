from pipeline.retrieval.retriever import retrieve

print("=== English ===")
results = retrieve("who is eligible for PM Poshan", language="en", top_k=3)
for r in results:
    print(f"[{r['scheme_id']}] [{r['section']}] score={r['score']:.4f}")
    print(r["text"][:150], "...\n")

print("\n=== Hindi ===")
results = retrieve("पीएम पोषण के लिए कौन पात्र है", language="hi", top_k=3)
for r in results:
    print(f"[{r['scheme_id']}] [{r['section']}] score={r['score']:.4f}")
    print(r["text"][:150], "...\n")

print("\n=== Marathi ===")
results = retrieve("पीएम पोषण साठी कोण पात्र आहे", language="mr", top_k=3)
for r in results:
    print(f"[{r['scheme_id']}] [{r['section']}] score={r['score']:.4f}")
    print(r["text"][:150], "...\n")