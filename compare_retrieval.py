from retrieval import retrieve
from semantic_retrieval import build_index, semantic_search

queries = [
    "What happens when a payment provider times out?",
    "How should we handle authorization when the gateway stops responding?",
    "How do we prevent duplicate refunds?",
    "What if settlement records don't match?",
]

# Build once and reuse for all queries.
index = build_index()

for query in queries:
    print("\nQUESTION:", query)

    print("\nKeyword:")
    for result in retrieve(query):
        print(result["source"], result["score"])

    print("\nSemantic:")
    for result in semantic_search(query, index):
        print(result["source"], round(result["score"], 4))

    print("-" * 50)
