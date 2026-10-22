import json
from datetime import datetime, timezone, UTC
from pathlib import Path

from hybrid_retrieval import hybrid_search
from relevance_gate import retrieve_with_gate
from trusted_retrieval import approved_documents, load_registry

BASE = Path(__file__).resolve().parent


def load_json(filename):
    return json.loads((BASE / filename).read_text(encoding="utf-8"))


def retrieval_assessment():
    cases = load_json("rag_eval_dataset.json")
    evaluated = []

    for case in cases:
        expected = set(case["relevant_chunks"])

        if not expected:
            continue

        retrieved = hybrid_search(case["query"], top_k=3)

        ids = [x["chunk_id"] for x in retrieved]
        hits = expected.intersection(ids)

        rr = next(
            (1 / rank for rank, chunk_id in enumerate(ids, 1) if chunk_id in expected),
            0.0,
        )

        evaluated.append(
            {
                "id": case["id"],
                "hit": int(bool(hits)),
                "recall": len(hits) / len(expected),
                "reciprocal_rank": rr,
            }
        )

    if not evaluated:
        return {"status": "NOT_TESTED"}

    n = len(evaluated)

    return {
        "status": "MEASURED",
        "cases": n,
        "hit_at_3": sum(x["hit"] for x in evaluated) / n,
        "mean_recall_at_3": sum(x["recall"] for x in evaluated) / n,
        "mrr_at_3": sum(x["reciprocal_rank"] for x in evaluated) / n,
    }


def abstention_assessment():
    questions = [
        "What is AtlasPay's employee maternity leave policy?",
        "What is the company's annual leave allowance?",
        "How do employees claim medical insurance?",
    ]

    results = []

    for question in questions:
        result = retrieve_with_gate(question)

        results.append(
            {
                "question": question,
                "status": result["status"],
                "correct": result["status"] == "NO_EVIDENCE",
            }
        )

    return {
        "status": "MEASURED",
        "cases": results,
        "passed": sum(x["correct"] for x in results),
        "total": len(results),
    }


def authority_assessment():
    index = approved_documents()
    registry = load_registry()

    excluded = [
        name
        for name, metadata in registry.items()
        if metadata.get("status") == "superseded"
    ]

    active_sources = {chunk["source"] for chunk in index.values()}

    return {
        "status": "MEASURED",
        "superseded_sources": excluded,
        "excluded_from_approved_index": all(
            source not in active_sources for source in excluded
        ),
        "tested_superseded_sources": len(excluded),
    }


def main():
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "retrieval": retrieval_assessment(),
        "abstention": abstention_assessment(),
        "document_authority": authority_assessment(),
        "answer_faithfulness": {
            "status": "NOT_TESTED",
            "reason": "Requires captured answers and evidence",
        },
        "conflict_enforcement": {"status": "NOT_TESTED"},
        "action_safety": {"status": "NOT_TESTED"},
    }

    output = BASE / "rag_reliability_report.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nReport written to: {output}")


if __name__ == "__main__":
    main()
