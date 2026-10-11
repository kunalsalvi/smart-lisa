import re
from pathlib import Path

KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"


def tokenize(text):
    return set(re.findall(r"\b[a-z0-9_]+\b", text.lower()))


def load_documents():
    documents = []

    for path in sorted(KNOWLEDGE_DIR.glob("*.txt")):
        documents.append(
            {"source": path.name, "content": path.read_text(encoding="utf-8")}
        )

    return documents


def retrieve(query, top_k=2):
    query_tokens = tokenize(query)
    results = []

    for doc in load_documents():
        document_tokens = tokenize(doc["content"])

        score = len(query_tokens & document_tokens)

        if score > 0:
            results.append(
                {"source": doc["source"], "content": doc["content"], "score": score}
            )

    results.sort(key=lambda item: item["score"], reverse=True)

    return results[:top_k]


if __name__ == "__main__":
    question = "What if payment provider times out?"

    for result in retrieve(question):
        print(f"\nSource: {result['source']}")
        print(f"Score: {result['score']}")
        print(result["content"])
