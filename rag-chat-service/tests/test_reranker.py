import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.config import settings
from app.reranker import normalize_rerank_score, rerank_chunks
from app.schemas import RetrievedChunk


def parse_sse_text(sse_raw_response: str) -> str:
    """Helper to reconstruct text stream from SSE response lines."""
    tokens = []
    for line in sse_raw_response.splitlines():
        if line.startswith("data:"):
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                continue
            try:
                payload = json.loads(data_str)
                if isinstance(payload, dict) and "token" in payload:
                    tokens.append(payload["token"])
            except Exception:
                pass
    return "".join(tokens)



def test_normalize_rerank_score():
    """Verify calibrated score for positive and negative logits."""
    assert normalize_rerank_score(0.0) > 0.80
    assert normalize_rerank_score(-2.8) > 0.50
    assert normalize_rerank_score(-11.0) < 0.05



def test_reranker_reorders_chunks_by_relevance():
    """
    Verify that Cross-Encoder accurately promotes the genuinely relevant chunk
    to rank #1 even if another chunk had a higher initial vector score.
    """
    query = "What is the policy for medical health insurance claim submission?"

    chunk_unrelated = RetrievedChunk(
        chunk_id="chunk_unrelated",
        content="Employees must submit travel expense receipts within 30 days of the trip.",
        source_document="Travel_Expense_Policy.pdf",
        score=0.75,  # High vector score due to keywords 'policy', 'submission'
    )

    chunk_relevant = RetrievedChunk(
        chunk_id="chunk_relevant",
        content="All medical insurance claims must be submitted to HR with original hospital bills within 14 days.",
        source_document="Health_Insurance_Handbook.pdf",
        score=0.60,  # Lower vector score
    )

    reranked = rerank_chunks(query=query, chunks=[chunk_unrelated, chunk_relevant], top_k=2)

    assert len(reranked) == 2
    # The health insurance chunk must be re-ranked to index 0!
    assert reranked[0].chunk_id == "chunk_relevant"
    assert reranked[0].rerank_score is not None
    assert reranked[1].chunk_id == "chunk_unrelated"
    assert reranked[0].rerank_score > reranked[1].rerank_score


def test_reranker_truncates_to_top_k():
    """Verify that reranker correctly slices candidate pool down to requested top_k."""
    query = "Annual leave entitlement"
    chunks = [
        RetrievedChunk(chunk_id=f"c_{i}", content=f"Policy chunk {i} description", source_document="doc.pdf", score=0.5)
        for i in range(8)
    ]

    reranked = rerank_chunks(query=query, chunks=chunks, top_k=3)
    assert len(reranked) == 3
    # Verify sorted descending
    scores = [c.rerank_score for c in reranked]
    assert scores == sorted(scores, reverse=True)


def test_reranker_disabled_fallback():
    """Verify clean fallback to vector similarity score when RERANKER_ENABLED is False."""
    query = "Office timings"
    chunks = [
        RetrievedChunk(chunk_id="c_low", content="timing A", source_document="doc.pdf", score=0.40),
        RetrievedChunk(chunk_id="c_high", content="timing B", source_document="doc.pdf", score=0.88),
    ]

    with patch.object(settings, "RERANKER_ENABLED", False):
        reranked = rerank_chunks(query=query, chunks=chunks, top_k=2)
        assert len(reranked) == 2
        assert reranked[0].chunk_id == "c_high"
        assert reranked[0].score == 0.88


def test_reranker_empty_or_blank_edge_cases():
    """Verify reranker handles empty inputs gracefully without exceptions."""
    assert rerank_chunks(query="test", chunks=[]) == []
    chunks = [RetrievedChunk(chunk_id="c1", content="some content", source_document="doc.pdf", score=0.5)]
    assert len(rerank_chunks(query="", chunks=chunks)) == 1


@pytest.mark.asyncio
async def test_reranker_confidence_threshold_skips_llm(client, sample_user_branch_1):
    """
    Verify that when retrieved chunks have low cross-encoder rerank relevance scores,
    the confidence threshold triggers and the LLM call is SKIPPED.
    """
    irrelevant_chunk = RetrievedChunk(
        chunk_id="c_irrelevant",
        content="Office cafeteria provides coffee and tea on weekdays from 9am to 5pm.",
        source_document="Cafeteria_Rules.pdf",
        score=0.45,
    )

    mock_llm = MagicMock()

    with patch("app.routers.chat.validate_user_context", AsyncMock(return_value=sample_user_branch_1)), \
         patch("app.routers.chat.get_scope_ingestion_version", AsyncMock(return_value="v1")), \
         patch("app.routers.chat.search_knowledge_base", AsyncMock(return_value=[irrelevant_chunk])), \
         patch("app.rag_engine.stream_llm_completion", mock_llm):

        # Ask a totally unrelated query about deep space astrophysics
        res = client.post(
            "/api/v1/chat",
            data={"message": "What is the Schwarzschild radius of a supermassive black hole?"},
            headers={"Authorization": "Bearer fake-token-1"},
        )
        assert res.status_code == 200

        # LLM must NOT be called because cross-encoder rerank score on cafeteria rules is very low
        mock_llm.assert_not_called()
        assert "This isn't covered in company documents" in parse_sse_text(res.text)

