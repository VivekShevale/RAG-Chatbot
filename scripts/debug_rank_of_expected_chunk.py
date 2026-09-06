"""
scripts/debug_rank_of_expected_chunk.py

For each failing case, finds the EXACT rank and distance of the chunk we
expect to be retrieved (e.g. the tablet_assistance documents_required chunk
in Hindi), even if it falls outside the top 5. Also confirms the chunk
actually exists in the index at all.

Run from project root:
    python scripts/debug_rank_of_expected_chunk.py
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

CHROMA_PATH = Path("data/processed/chroma_db")
COLLECTION_NAME = "scheme_chunks"
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"

CASES = [
    {
        "label": "HI - tablet_assistance documents_required",
        "query": "टैबलेट सहायता योजना के लिए कौन से दस्तावेज चाहिए?",
        "language": "hi",
        "expected_scheme_id": "scheme_05_tablet_assistance",
        "expected_section": "documents_required",
    },
    {
        "label": "MR - tablet_assistance documents_required",
        "query": "टॅब्लेट सहाय्य योजनेसाठी कोणती कागदपत्रे लागतात?",
        "language": "mr",
        "expected_scheme_id": "scheme_05_tablet_assistance",
        "expected_section": "documents_required",
    },
    {
        "label": "HI - vahan_aksmat eligibility",
        "query": "वाहन अक्षमत सहाय योजना किसे मिलती है?",
        "language": "hi",
        "expected_scheme_id": "scheme_02_vahan_aksmat_sahay",
        "expected_section": "eligibility",
    },
    {
        "label": "MR - vahan_aksmat eligibility",
        "query": "वाहन अक्षम सहाय योजना कोणाला मिळते?",
        "language": "mr",
        "expected_scheme_id": "scheme_02_vahan_aksmat_sahay",
        "expected_section": "eligibility",
    },
]

print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL)
client = PersistentClient(path=str(CHROMA_PATH))
collection = client.get_collection(name=COLLECTION_NAME)

print(f"Total chunks in collection: {collection.count()}\n")

for case in CASES:
    print("=" * 90)
    print(f"CASE: {case['label']}")
    print(f"Query: {case['query']}")

    # Step 1: does the expected chunk even exist in the index?
    existing = collection.get(
        where={
            "$and": [
                {"scheme_id": case["expected_scheme_id"]},
                {"section": case["expected_section"]},
                {"language": case["language"]},
            ]
        },
        include=["documents", "metadatas"],
    )

    if not existing["ids"]:
        print(f"  >>> PROBLEM: expected chunk NOT FOUND in index at all "
              f"(scheme_id={case['expected_scheme_id']}, section={case['expected_section']}, "
              f"language={case['language']}) <<<")
        print("  This means the chunk never made it into chunks.json or never got embedded.")
        continue

    print(f"  Expected chunk EXISTS in index. id={existing['ids'][0]}")
    print(f"  Stored text preview: {existing['documents'][0][:100]}...")

    # Step 2: rank it against ALL chunks in that language (not just top 5)
    query_embedding = model.encode([f"query: {case['query']}"], normalize_embeddings=True).tolist()

    all_in_lang = collection.query(
        query_embeddings=query_embedding,
        n_results=collection.count(),  # get everything, so we can find the true rank
        where={"language": case["language"]},
        include=["documents", "metadatas", "distances"],
    )

    target_id = existing["ids"][0]
    found_rank = None
    for rank, chunk_id in enumerate(all_in_lang["ids"][0], 1):
        if chunk_id == target_id:
            found_rank = rank
            found_distance = all_in_lang["distances"][0][rank - 1]
            break

    total_in_lang = len(all_in_lang["ids"][0])
    if found_rank:
        print(f"  Expected chunk's TRUE RANK: {found_rank} out of {total_in_lang} "
              f"(distance={found_distance:.4f})")
    else:
        print(f"  >>> PROBLEM: expected chunk exists but was NOT RETURNED by query() at all <<<")

    print()

print("=" * 90)