from semantic_retrieval import cosine_similarity, embed_text
from trusted_retrieval import approved_documents

RELEVANCE_THRESHOLD = 0.35


def retrieve_with_gate(query, top_k=3):
    documents = list(approved_documents().values())

    if not documents:
        return {"status": "NO_EVIDENCE", "results": []}

    query_vector = embed_text(query)
    candidates = []

    for doc in documents:
        score = cosine_similarity(query_vector, doc["embedding"])

        candidates.append(
            {
                "chunk_id": doc["chunk_id"],
                "source": doc["source"],
                "content": doc["content"],
                "similarity": score,
            }
        )

    candidates.sort(key=lambda item: item["similarity"], reverse=True)

    relevant = [
        item for item in candidates if item["similarity"] >= RELEVANCE_THRESHOLD
    ]

    if not relevant:
        return {"status": "NO_EVIDENCE", "results": []}

    return {"status": "EVIDENCE_FOUND", "results": relevant[:top_k]}


if __name__ == "__main__":
    questions = [
        "What happens when a payment provider times out?",
        "What is AtlasPay's maternity leave policy?",
        "How do we prevent duplicate refunds?",
    ]

    for question in questions:
        result = retrieve_with_gate(question)

        print(f"\nQuestion: {question}")
        print(f"Status: {result['status']}")

        for item in result["results"]:
            print(item["chunk_id"], round(item["similarity"], 4))
