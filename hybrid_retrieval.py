from persistent_index import load_index
from retrieval import tokenize
from semantic_retrieval import cosine_similarity, embed_text


def keyword_rank(query, documents):
    query_tokens = tokenize(query)
    results = []

    for doc in documents:
        score = len(query_tokens & tokenize(doc["content"]))

        if score > 0:
            results.append((doc["chunk_id"], score))

    return sorted(results, key=lambda item: (-item[1], item[0]))


def semantic_rank(query, documents):
    query_vector = embed_text(query)
    results = []

    for doc in documents:
        score = cosine_similarity(query_vector, doc["embedding"])

        results.append((doc["chunk_id"], score))

    return sorted(results, key=lambda item: (-item[1], item[0]))


def reciprocal_rank_fusion(rankings, k=60):
    fused = {}

    for ranking in rankings:
        for position, (chunk_id, _) in enumerate(ranking, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + position)

    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


def hybrid_search(query, top_k=3):
    index = load_index()
    documents = list(index.values())

    if not documents:
        return []

    keyword_results = keyword_rank(query, documents)
    # print_search_results("Keyword", 2, index, keyword_results)

    semantic_results = semantic_rank(query, documents)
    # print_search_results("Semantic", 2, index, semantic_results)

    fused_results = reciprocal_rank_fusion([keyword_results, semantic_results])
    # print_search_results("Fused", 2, index, fused_results)

    results = []

    for chunk_id, score in fused_results[:top_k]:
        document = index[chunk_id]

        results.append(
            {
                "chunk_id": chunk_id,
                "source": document["source"],
                "content": document["content"],
                "score": score,
            }
        )

    return results


def print_search_results(search_type, top_k, index, search_results):
    print(f"Top {top_k} Results ({search_type}):")
    for chunk_id, score in search_results[:top_k]:
        document = index[chunk_id]
        print(f"  {chunk_id}: {document['content']}")


if __name__ == "__main__":
    # query = "What should happen when a payment gateway stops responding?"
    query = "What happens when a payment provider times out?"
    # query = "How should we handle authorization when the gateway stops responding?"
    # query = "What does PENDING_VERIFICATION mean?"
    # query = "How do we prevent duplicate refunds?"
    # query = "What if settlement records don't match?"
    # query = "What is AtlasPay's employee maternity leave policy?"
    print(f"Query : {query}\n")
    for result in hybrid_search(query):
        print(result["chunk_id"], round(result["score"], 5), "\n")
