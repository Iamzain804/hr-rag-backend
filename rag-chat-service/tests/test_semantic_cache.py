import json
import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch

from app.semantic_cache import (
    clear_cache,
    is_cache_eligible,
    lookup_semantic_cache,
    store_semantic_cache_entry,
)
from app.schemas import RetrievedChunk, UserContext


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



@pytest.fixture(autouse=True)
def reset_cache():
    """Ensure semantic cache collection is cleared before each test."""
    clear_cache()
    yield
    clear_cache()


def test_cache_eligibility_rules():
    """Verify that only grounded, non-attachment, non-fallback responses are eligible for caching."""
    # 1. Standard valid grounded response -> Eligible
    assert is_cache_eligible(
        has_attachment=False,
        top_score=0.85,
        is_grounded=True,
        answer="The company provides 20 annual leaves. [Source: leave_policy.pdf]",
    ) is True

    # 2. Has attachment -> Ineligible
    assert is_cache_eligible(
        has_attachment=True,
        top_score=0.85,
        is_grounded=True,
        answer="Based on what you shared...",
    ) is False

    # 3. Groundedness failed -> Ineligible
    assert is_cache_eligible(
        has_attachment=False,
        top_score=0.85,
        is_grounded=False,
        answer="Some hallucinated answer.",
    ) is False

    # 4. Low retrieval confidence score -> Ineligible
    assert is_cache_eligible(
        has_attachment=False,
        top_score=0.20,
        is_grounded=True,
        answer="Some answer",
    ) is False

    # 5. Fallback response -> Ineligible
    assert is_cache_eligible(
        has_attachment=False,
        top_score=0.10,
        is_grounded=True,
        answer="This isn't covered in company documents. Would you like this flagged to HR?",
    ) is False


def test_semantic_cache_hit_and_similarity_threshold():
    """Verify semantic cache hit for semantically identical questions and threshold behavior."""
    query = "What is the annual leave policy?"
    answer = "Employees receive 20 annual leaves per year. [Source: policy.pdf]"
    sources = [{"source_document": "policy.pdf", "chunk_index": 0, "score": 0.92}]
    version = "v_hash123"

    cache_id = store_semantic_cache_entry(
        query=query,
        answer=answer,
        branch_id=1,
        department_id=2,
        sources=sources,
        ingestion_version=version,
        ttl_seconds=3600,
    )
    assert cache_id is not None

    # Exact query match -> HIT (score ~1.0)
    hit = lookup_semantic_cache(
        query="What is the annual leave policy?",
        branch_id=1,
        department_id=2,
        current_ingestion_version=version,
        threshold=0.90,
    )
    assert hit is not None
    assert hit.answer == answer
    assert hit.score >= 0.90
    assert len(hit.sources) == 1

    # Semantically equivalent query -> HIT (e.g. "What is our annual leave policy?")
    hit_similar = lookup_semantic_cache(
        query="What is our annual leave policy?",
        branch_id=1,
        department_id=2,
        current_ingestion_version=version,
        threshold=0.85,
    )
    assert hit_similar is not None
    assert hit_similar.answer == answer

    # Completely different question -> MISS
    miss = lookup_semantic_cache(
        query="What is the reimbursement procedure for travel?",
        branch_id=1,
        department_id=2,
        current_ingestion_version=version,
        threshold=0.90,
    )
    assert miss is None


def test_strict_scope_isolation():
    """
    CRITICAL: Verify that Lahore/Engineering answers are NEVER served to
    an employee of a different branch or department.
    """
    version = "v_scope1"
    store_semantic_cache_entry(
        query="What are the working hours?",
        answer="Lahore branch office hours are 9:00 AM to 6:00 PM PKT.",
        branch_id=10,  # Lahore Branch
        department_id=5,  # Engineering
        sources=[{"source_document": "lahore_office.pdf", "chunk_index": 0, "score": 0.95}],
        ingestion_version=version,
    )

    # 1. Matching Lahore / Engineering caller -> HIT
    hit = lookup_semantic_cache(
        query="What are the working hours?",
        branch_id=10,
        department_id=5,
        current_ingestion_version=version,
    )
    assert hit is not None
    assert "Lahore branch" in hit.answer

    # 2. Karachi branch caller (branch_id=20, department_id=5) -> STRICT MISS
    miss_karachi = lookup_semantic_cache(
        query="What are the working hours?",
        branch_id=20,
        department_id=5,
        current_ingestion_version=version,
    )
    assert miss_karachi is None

    # 3. Lahore Marketing department caller (branch_id=10, department_id=8) -> STRICT MISS
    miss_marketing = lookup_semantic_cache(
        query="What are the working hours?",
        branch_id=10,
        department_id=8,
        current_ingestion_version=version,
    )
    assert miss_marketing is None

    # 4. Global scope caller (branch_id=None, department_id=None) -> STRICT MISS
    miss_global = lookup_semantic_cache(
        query="What are the working hours?",
        branch_id=None,
        department_id=None,
        current_ingestion_version=version,
    )
    assert miss_global is None


def test_scope_ingestion_version_invalidation():
    """Verify that stale cache entries are invalidated and purged when documents in the scope change."""
    old_version = "v_initial_docs_111"
    new_version = "v_updated_docs_222"

    store_semantic_cache_entry(
        query="What is the health insurance coverage limit?",
        answer="Coverage limit is PKR 500,000. [Source: insurance_2025.pdf]",
        branch_id=1,
        department_id=1,
        sources=[{"source_document": "insurance_2025.pdf"}],
        ingestion_version=old_version,
    )

    # Query with matching old version -> HIT
    hit = lookup_semantic_cache(
        query="What is the health insurance coverage limit?",
        branch_id=1,
        department_id=1,
        current_ingestion_version=old_version,
    )
    assert hit is not None

    # HR uploads a new document or re-ingests -> version updates to new_version -> Stale cache entry is INVALIDATED (MISS)
    invalidated_miss = lookup_semantic_cache(
        query="What is the health insurance coverage limit?",
        branch_id=1,
        department_id=1,
        current_ingestion_version=new_version,
    )
    assert invalidated_miss is None


def test_ttl_expiry_invalidation():
    """Verify that expired cache entries are not served."""
    version = "v_ttl_test"
    store_semantic_cache_entry(
        query="Where is the cafeteria located?",
        answer="Cafeteria is on the 2nd floor.",
        branch_id=1,
        department_id=1,
        sources=[],
        ingestion_version=version,
        ttl_seconds=1,  # 1 second TTL
    )

    # Fast lookup -> HIT
    hit = lookup_semantic_cache(
        query="Where is the cafeteria located?",
        branch_id=1,
        department_id=1,
        current_ingestion_version=version,
    )
    assert hit is not None

    # Wait for TTL to expire
    time.sleep(1.2)

    # Lookup after expiry -> MISS
    expired_miss = lookup_semantic_cache(
        query="Where is the cafeteria located?",
        branch_id=1,
        department_id=1,
        current_ingestion_version=version,
    )
    assert expired_miss is None


@pytest.mark.asyncio
async def test_chat_endpoint_semantic_cache_hit_and_streaming(client, sample_user_branch_1):
    """
    Test End-to-End Chat API:
    1st request: Cache Miss -> Retrieves from vector DB, invokes LLM, populates cache.
    2nd request: Cache Hit -> Serves directly from semantic cache (served_from_cache: True, no LLM call).
    """
    doc_chunk = RetrievedChunk(
        chunk_id="doc_leave_1",
        content="Standard annual leave entitlement is 20 days per year.",
        source_document="Leave_Policy.pdf",
        score=0.92,
        branch_id=1,
        department_id=1,
        is_company_wide=False,
    )

    async def mock_llm_stream(*args, **kwargs):
        yield "According to [Source: Leave_Policy.pdf], you receive 20 annual leaves."

    mock_llm = MagicMock(side_effect=mock_llm_stream)

    with patch("app.routers.chat.validate_user_context", AsyncMock(return_value=sample_user_branch_1)), \
         patch("app.routers.chat.get_scope_ingestion_version", AsyncMock(return_value="v_test_version_1")), \
         patch("app.routers.chat.search_knowledge_base", AsyncMock(return_value=[doc_chunk])) as mock_search, \
         patch("app.rag_engine.stream_llm_completion", mock_llm):

        # 1. First request -> Cache MISS
        res1 = client.post(
            "/api/v1/chat",
            data={"message": "How many annual leaves do I have?"},
            headers={"Authorization": "Bearer fake-token-1"},
        )
        assert res1.status_code == 200
        assert mock_search.call_count == 1
        assert mock_llm.call_count == 1

        # Check response metadata indicates served_from_cache: False
        assert '"served_from_cache": false' in res1.text.lower()
        assert "Leave_Policy.pdf" in res1.text

        # Reset mock call counts
        mock_search.reset_mock()
        mock_llm.reset_mock()

        # 2. Second request (identical or semantically close question) -> Cache HIT!
        res2 = client.post(
            "/api/v1/chat",
            data={"message": "How many annual leaves do I have?"},
            headers={"Authorization": "Bearer fake-token-1"},
        )
        assert res2.status_code == 200

        # Vector DB search and LLM should NOT be called on Cache Hit!
        mock_search.assert_not_called()
        mock_llm.assert_not_called()

        # Check response metadata indicates served_from_cache: True
        assert '"served_from_cache": true' in res2.text.lower()
        assert "20 annual leaves" in parse_sse_text(res2.text)


