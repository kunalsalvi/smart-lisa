import os
from math import sqrt

from dotenv import load_dotenv
from openai import OpenAI

from retrieval import load_documents

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

EMBEDDING_MODEL = "text-embedding-3-small"


def embed_text(text):
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))

    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def build_index():
    documents = load_documents()
    index = []

    for document in documents:
        vector = embed_text(document["content"])

        index.append(
            {
                "source": document["source"],
                "content": document["content"],
                "embedding": vector,
            }
        )

    return index


def semantic_search(query, index, top_k=2):
    query_vector = embed_text(query)
    results = []

    for document in index:
        score = cosine_similarity(query_vector, document["embedding"])

        results.append(
            {
                "source": document["source"],
                "content": document["content"],
                "score": score,
            }
        )

    results.sort(key=lambda result: result["score"], reverse=True)

    return results[:top_k]


if __name__ == "__main__":
    index = build_index()

    query = "How should we handle authorization when the gateway stops responding?"

    for result in semantic_search(query, index):
        print(result["source"], round(result["score"], 4))
