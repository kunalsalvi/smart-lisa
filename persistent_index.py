import hashlib
import json
import os
from pathlib import Path

from chunking import load_chunks
from llm_interface import getOpenAIAPIClient

client = getOpenAIAPIClient()

MODEL = "text-embedding-3-small"
INDEX_FILE = Path(__file__).resolve().parent / "embedding_index.json"


def fingerprint(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_index():
    if not INDEX_FILE.exists():
        return {}

    with INDEX_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_index(index):
    # Write a temporary file, then atomically replace
    # the target on the same filesystem.
    temp_file = INDEX_FILE.with_suffix(".json.tmp")

    with temp_file.open("w", encoding="utf-8") as file:
        json.dump(index, file, indent=2)
        file.flush()
        os.fsync(file.fileno())

    os.replace(temp_file, INDEX_FILE)


def build_index():
    existing = load_index()
    updated = {}

    generated = 0
    reused = 0

    for chunk in load_chunks():
        chunk_id = chunk["chunk_id"]
        text = chunk["content"]
        hash_value = fingerprint(text)

        cached = existing.get(chunk_id)

        if cached and cached["hash"] == hash_value and cached["model"] == MODEL:
            embedding = cached["embedding"]
            reused += 1
        else:
            response = client.embeddings.create(model=MODEL, input=text)

            embedding = response.data[0].embedding
            generated += 1

        updated[chunk_id] = {
            **chunk,
            "hash": hash_value,
            "model": MODEL,
            "embedding": embedding,
        }

    # This also removes chunks no longer present.
    save_index(updated)

    print(f"Generated: {generated}")
    print(f"Reused: {reused}")
    print(f"Total chunks: {len(updated)}")

    return list(updated.values())


if __name__ == "__main__":
    build_index()
