import json
from pathlib import Path

from hybrid_retrieval import hybrid_search

BASE_DIR = Path(__file__).resolve().parent


def load_dataset():
    path = BASE_DIR / "rag_eval_dataset.json"

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def evaluate_case(case, retrieved):
    expected = set(case["relevant_chunks"])

    retrieved_ids = [item["chunk_id"] for item in retrieved]

    # Handle questions without a relevant document
    # separately from retrieval-recall metrics.
    if not expected:
        return {
            "id": case["id"],
            "hit": None,
            "recall": None,
            "reciprocal_rank": None,
            "retrieved": retrieved_ids,
            "expected": [],
        }

    found = expected.intersection(retrieved_ids)

    hit = 1.0 if found else 0.0
    recall = len(found) / len(expected)

    reciprocal_rank = 0.0

    for position, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in expected:
            reciprocal_rank = 1.0 / position
            break

    return {
        "id": case["id"],
        "hit": hit,
        "recall": recall,
        "reciprocal_rank": reciprocal_rank,
        "retrieved": retrieved_ids,
        "expected": sorted(expected),
    }


def main():
    dataset = load_dataset()
    results = []

    for case in dataset:
        retrieved = hybrid_search(case["query"], top_k=3)

        result = evaluate_case(case, retrieved)
        results.append(result)

        print(f"\n{case['id']}: {case['query']}")
        print(f"Expected:  {result['expected']}")
        print(f"Retrieved: {result['retrieved']}")
        print(f"Hit@3:     {result['hit']}")
        print(f"Recall@3:  {result['recall']}")
        print(f"RR:        {result['reciprocal_rank']}")

    answerable = [result for result in results if result["hit"] is not None]

    count = len(answerable)

    if count:
        mean_hit = sum(r["hit"] for r in answerable) / count

        mean_recall = sum(r["recall"] for r in answerable) / count

        mrr = sum(r["reciprocal_rank"] for r in answerable) / count

        print("\n========== SUMMARY ==========")
        print(f"Hit@3:       {mean_hit:.2%}")
        print(f"Mean Recall@3: {mean_recall:.2%}")
        print(f"MRR@3:       {mrr:.4f}")

    print("\nNote: Unanswerable questions require a separate abstention evaluation.")


if __name__ == "__main__":
    main()
