from persistent_index import load_index
from semantic_retrieval import cosine_similarity, embed_text

test_cases = [
    {"query": "What happens when a payment provider times out?", "answerable": True},
    {
        "query": "How should we handle authorization when the gateway stops responding?",
        "answerable": True,
    },
    {"query": "What does PENDING_VERIFICATION mean?", "answerable": True},
    {"query": "How do we prevent duplicate refunds?", "answerable": True},
    {"query": "What if settlement records don't match?", "answerable": True},
    {
        "query": "What is AtlasPay's employee maternity leave policy?",
        "answerable": False,
    },
    {"query": "What is the company's annual leave allowance?", "answerable": False},
    {"query": "How do employees claim medical insurance?", "answerable": False},
]


def evaluate():
    documents = list(load_index().values())
    results = []

    if not documents:
        raise RuntimeError("Build the index first")

    for case in test_cases:
        query_vector = embed_text(case["query"])

        ranked = []

        for doc in documents:
            score = cosine_similarity(query_vector, doc["embedding"])

            ranked.append((doc["chunk_id"], score))

        ranked.sort(key=lambda x: x[1], reverse=True)

        top_chunk, top_score = ranked[0]
        results.append(
            {
                "query": case["query"],
                "score": top_score,
                "answerable": case["answerable"],
            }
        )

        print(f"\nQuery: {case['query']}")
        print(f"Answerable: {case['answerable']}")
        print(f"Top chunk: {top_chunk}")
        print(f"Similarity: {top_score:.4f}")

    print("\nTHRESHOLD EVALUATION")

    for threshold in [0.20, 0.30, 0.35, 0.40, 0.45, 0.50]:
        metrics = evaluate_threshold(results, threshold)
        print(metrics)


def evaluate_threshold(results, threshold):
    tp = tn = fp = fn = 0

    for result in results:
        predicted = result["score"] >= threshold
        actual = result["answerable"]

        if predicted and actual:
            tp += 1
        elif not predicted and not actual:
            tn += 1
        elif predicted and not actual:
            fp += 1
        else:
            fn += 1

    accuracy = (tp + tn) / len(results)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "false_positives": fp,
        "false_negatives": fn,
    }


if __name__ == "__main__":
    evaluate()
