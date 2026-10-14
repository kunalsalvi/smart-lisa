from hybrid_retrieval import hybrid_search
from persistent_index import load_index


def neighboring_ids(chunk_id, radius=1):
    document_id, chunk_number = chunk_id.rsplit(":", 1)
    position = int(chunk_number)

    return [
        f"{document_id}:{i}"
        for i in range(max(0, position - radius), position + radius + 1)
    ]


def expand_chunks(retrieved, index, radius=1):
    expanded = []
    seen = set()

    for result in retrieved:
        chunk_id = result["chunk_id"]

        for neighbor_id in neighboring_ids(chunk_id, radius):
            if neighbor_id not in index:
                continue

            if neighbor_id in seen:
                continue

            seen.add(neighbor_id)

            doc = index[neighbor_id]

            expanded.append(
                {
                    "chunk_id": neighbor_id,
                    "source": doc["source"],
                    "content": doc["content"],
                    "origin": ("retrieved" if neighbor_id == chunk_id else "expanded"),
                }
            )

    return expanded


def retrieve_with_context(query, top_k=1):
    index = load_index()

    candidates = hybrid_search(query, top_k=top_k)

    return expand_chunks(candidates, index, radius=1)


if __name__ == "__main__":
    query = "What happens when a payment provider times out?"

    results = retrieve_with_context(query, top_k=1)

    for result in results:
        print(result["chunk_id"], result["origin"])
        print(result["content"])
        print("-" * 50)

    results = hybrid_search("What happens when a payment provider times out?", top_k=1)

    print([r["chunk_id"] for r in results])
