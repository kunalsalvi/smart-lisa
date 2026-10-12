from pathlib import Path

KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"


def chunk_text(text, chunk_size=50, overlap=10):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be nonnegative and smaller than chunk_size")

    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])

        chunks.append({"content": chunk, "start_word": start, "end_word": end})

        if end == len(words):
            break

    return chunks


def load_chunks():
    results = []

    for path in sorted(KNOWLEDGE_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        chunks = chunk_text(text)

        for index, chunk in enumerate(chunks):
            results.append(
                {"chunk_id": f"{path.stem}:{index}", "source": path.name, **chunk}
            )

    return results


if __name__ == "__main__":
    chunks = load_chunks()

    for chunk in chunks:
        print(f"\n[{chunk['chunk_id']}] Source: {chunk['source']}")
        print(chunk["content"])
        print("-" * 50)
