import json
from pathlib import Path

from persistent_index import load_index

BASE_DIR = Path(__file__).resolve().parent


def load_registry():
    path = BASE_DIR / "policy_registry.json"
    return json.loads(path.read_text(encoding="utf-8"))


def approved_documents():
    registry = load_registry()
    index = load_index()
    approved = {}

    for chunk_id, chunk in index.items():
        metadata = registry.get(chunk["source"])

        # Fail closed when approval metadata is missing.
        if metadata is None:
            continue

        if metadata.get("status") != "approved":
            continue

        approved[chunk_id] = {
            **chunk,
            "policy_id": metadata["policy_id"],
            "version": metadata["version"],
            "authority": metadata["authority"],
        }
    detected_conflicts = detect_version_conflicts(registry)
    if detected_conflicts:
        print("Detected version conflicts:", detected_conflicts)
        # In a real system, you might raise an exception or log this as an error.
        # For this example, we will just print a warning.
        print(
            "Warning: Version conflicts detected in the policy registry. "
            "This may lead to inconsistent retrieval results."
        )
        raise RuntimeError("Version conflicts detected in the policy registry.")
    return approved


CONSEQUENTIAL_TOOLS = {"send_payment"}


def execute_tool_safely(tool_name, arguments, approved_action=None):
    if tool_name in CONSEQUENTIAL_TOOLS:
        if approved_action is None:
            return {"status": "BLOCKED", "reason": "Independent approval required"}

        if (
            approved_action.get("tool_name") != tool_name
            or approved_action.get("arguments") != arguments
            or approved_action.get("authorized") is not True
        ):
            return {"status": "BLOCKED", "reason": "Approval does not match action"}

        # The real executor is deliberately omitted.
        # The training environment must never send money.
        return {"status": "SIMULATED_APPROVED", "action": tool_name}

    return {"status": "NOT_HANDLED", "reason": "Use the existing tool dispatcher"}


def detect_version_conflicts(registry):
    grouped = {}

    for filename, metadata in registry.items():
        if metadata["status"] != "approved":
            continue

        policy_id = metadata["policy_id"]

        grouped.setdefault(policy_id, []).append(
            {"source": filename, "version": metadata["version"]}
        )

    return {
        policy_id: entries for policy_id, entries in grouped.items() if len(entries) > 1
    }


if __name__ == "__main__":
    for chunk_id, document in approved_documents().items():
        print(
            chunk_id, document["policy_id"], document["version"], document["authority"]
        )

    result = execute_tool_safely(
        "send_payment", {"amount": 10000, "recipient": "test-beneficiary"}
    )

    print(result)

    test_registry = {
        "policy_v2.txt": {
            "policy_id": "PAYMENT_TIMEOUT",
            "version": 2,
            "status": "approved",
        },
        "policy_v3.txt": {
            "policy_id": "PAYMENT_TIMEOUT",
            "version": 3,
            "status": "approved",
        },
    }

    conflicts = detect_version_conflicts(test_registry)
    print("Detected version conflicts:", conflicts)
    assert "PAYMENT_TIMEOUT" in conflicts

    # detected_conflicts = detect_version_conflicts(load_registry())
    # print("Detected version conflicts:", detected_conflicts)
