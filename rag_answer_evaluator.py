import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def load_cases():
    path = BASE_DIR / "rag_answer_dataset.json"
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_answer(answer, case):
    normalized = " ".join(answer.lower().split())

    missing_terms = [
        term for term in case["must_contain"] if term.lower() not in normalized
    ]

    missing_citations = [
        source for source in case["must_cite"] if source.lower() not in normalized
    ]

    alternatives = case.get("must_contain_any", [])
    alternative_passed = not alternatives or any(
        phrase.lower() in normalized for phrase in alternatives
    )

    return {
        "id": case["id"],
        "passed": (not missing_terms and not missing_citations and alternative_passed),
        "missing_terms": missing_terms,
        "missing_citations": missing_citations,
    }


if __name__ == "__main__":
    for case in load_cases():
        print(f"\nTest {case['id']}")
        print(f"Question: {case['question']}")

        print("Paste answer. Finish with END:")
        lines = []

        while True:
            line = input()

            if line == "END":
                break

            lines.append(line)

        answer = "\n".join(lines)
        result = evaluate_answer(answer, case)

        print(result)
