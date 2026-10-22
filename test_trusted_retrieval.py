import contextlib
import io
import sys
import types
import unittest
from unittest.mock import patch


class OfflineOpenAI:
    def __init__(self, **kwargs):
        self.embeddings = None


def load_dotenv():
    return None


offline_dotenv = types.ModuleType("dotenv")
offline_dotenv.load_dotenv = load_dotenv
offline_openai = types.ModuleType("openai")
offline_openai.OpenAI = OfflineOpenAI

with patch.dict(
    sys.modules, {"dotenv": offline_dotenv, "openai": offline_openai}
):
    import persistent_index
    import relevance_gate
    import trusted_retrieval


class PolicyIntegrationTests(unittest.TestCase):
    def test_valid_policy_query_returns_current_evidence(self):
        index = persistent_index.load_index()
        policy_embedding = next(
            document["embedding"]
            for document in index.values()
            if document["source"] == "payment_timeout.txt"
        )

        with patch.object(relevance_gate, "embed_text", return_value=policy_embedding):
            result = relevance_gate.retrieve_with_gate(
                "What happens when a payment provider times out?"
            )

        self.assertEqual(result["status"], "EVIDENCE_FOUND")
        self.assertTrue(
            any(
                item["source"] == "payment_timeout.txt"
                and "PENDING_VERIFICATION" in item["content"]
                for item in result["results"]
            )
        )

    def test_out_of_domain_query_returns_no_evidence(self):
        index = persistent_index.load_index()
        vector_size = len(next(iter(index.values()))["embedding"])
        unrelated_embedding = [0.0] * vector_size

        with patch.object(
            relevance_gate, "embed_text", return_value=unrelated_embedding
        ):
            result = relevance_gate.retrieve_with_gate(
                "What is the company's parental leave policy?"
            )

        self.assertEqual(result, {"status": "NO_EVIDENCE", "results": []})

    def test_superseded_document_is_not_retrievable(self):
        approved = trusted_retrieval.approved_documents()
        sources = {document["source"] for document in approved.values()}

        self.assertIn("payment_timeout.txt", sources)
        self.assertNotIn("payment_timeout_v1.txt", sources)

    def test_conflicting_approved_policies_fail_closed(self):
        conflicting_registry = {
            "payment_timeout_v2.txt": {
                "policy_id": "PAYMENT_TIMEOUT",
                "version": 2,
                "status": "approved",
                "authority": "payments-risk",
            },
            "payment_timeout_v3.txt": {
                "policy_id": "PAYMENT_TIMEOUT",
                "version": 3,
                "status": "approved",
                "authority": "payments-risk",
            },
        }

        with (
            patch.object(
                trusted_retrieval, "load_registry", return_value=conflicting_registry
            ),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaisesRegex(RuntimeError, "Version conflicts"),
        ):
            trusted_retrieval.approved_documents()

    def test_unauthorized_payment_is_blocked(self):
        result = trusted_retrieval.execute_tool_safely(
            "send_payment",
            {"amount": 10000, "recipient": "test-beneficiary"},
        )

        self.assertEqual(
            result,
            {"status": "BLOCKED", "reason": "Independent approval required"},
        )


if __name__ == "__main__":
    unittest.main()
