import io
import json
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from PIL import Image, ImageDraw
from fastapi import HTTPException
from app.config import settings
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


@pytest.mark.asyncio
async def test_1_branch_and_department_filtered_retrieval(client, sample_user_branch_1, sample_user_branch_2):
    """
    Test 1: Confirm two employees in different branches/departments get answers
    grounded in their own branch/department documents, not the other's.
    """
    # 1. London Engineer Query
    london_chunk = RetrievedChunk(
        chunk_id="doc_london_c0",
        content="London Engineering team members receive £500 annual home office allowance.",
        source_document="London_Engineering_Handbook.pdf",
        score=0.88,
        branch_id=1,
        department_id=1,
        is_company_wide=False,
    )

    async def mock_stream_london(*args, **kwargs):
        yield "Based on [Source: London_Engineering_Handbook.pdf], London engineers get £500."

    with patch("app.routers.chat.validate_user_context", AsyncMock(return_value=sample_user_branch_1)), \
         patch("app.routers.chat.search_knowledge_base", AsyncMock(return_value=[london_chunk])) as mock_search_1, \
         patch("app.rag_engine.stream_llm_completion", side_effect=mock_stream_london):

        res1 = client.post(
            "/api/v1/chat",
            data={"message": "What is my home office allowance?"},
            headers={"Authorization": "Bearer fake-jwt-token-1"},
        )
        assert res1.status_code == 200
        text1 = parse_sse_text(res1.text)
        assert "London_Engineering_Handbook.pdf" in text1
        assert "500" in text1
        mock_search_1.assert_called_once_with(
            query="What is my home office allowance?",
            branch_id=1,
            department_id=1,
            top_k=4,
        )

    # 2. New York Marketer Query
    ny_chunk = RetrievedChunk(
        chunk_id="doc_ny_c0",
        content="New York marketing staff receive a $100 monthly subway transit credit.",
        source_document="NY_Marketing_Benefits.pdf",
        score=0.85,
        branch_id=2,
        department_id=2,
        is_company_wide=False,
    )

    async def mock_stream_ny(*args, **kwargs):
        yield "Based on [Source: NY_Marketing_Benefits.pdf], New York marketing employees receive $100 transit credit."

    with patch("app.routers.chat.validate_user_context", AsyncMock(return_value=sample_user_branch_2)), \
         patch("app.routers.chat.search_knowledge_base", AsyncMock(return_value=[ny_chunk])) as mock_search_2, \
         patch("app.rag_engine.stream_llm_completion", side_effect=mock_stream_ny):

        res2 = client.post(
            "/api/v1/chat",
            data={"message": "What transit benefits do I get?"},
            headers={"Authorization": "Bearer fake-jwt-token-2"},
        )
        assert res2.status_code == 200
        text2 = parse_sse_text(res2.text)
        assert "NY_Marketing_Benefits.pdf" in text2
        assert "$100" in text2
        mock_search_2.assert_called_once_with(
            query="What transit benefits do I get?",
            branch_id=2,
            department_id=2,
            top_k=4,
        )


@pytest.mark.asyncio
async def test_2_grounding_low_confidence_skips_llm_call(client, sample_user_branch_1):
    """
    Test 2: Ask a question with NO relevant document in the knowledge base (low similarity score).
    Confirm system returns fixed fallback message WITHOUT invoking the LLM (assert mock never called).
    """
    low_confidence_chunks = [
        RetrievedChunk(
            chunk_id="doc_unrelated_c0",
            content="Recycling bins are located on the second floor pantry.",
            source_document="Office_Map.pdf",
            score=0.22,  # Well below 0.50 threshold
            is_company_wide=True,
        )
    ]

    mock_llm = MagicMock()

    with patch("app.routers.chat.validate_user_context", AsyncMock(return_value=sample_user_branch_1)), \
         patch("app.routers.chat.search_knowledge_base", AsyncMock(return_value=low_confidence_chunks)), \
         patch("app.rag_engine.stream_llm_completion", mock_llm):

        res = client.post(
            "/api/v1/chat",
            data={"message": "Can I claim reimbursements for scuba diving certification?"},
            headers={"Authorization": "Bearer fake-jwt-token"},
        )
        assert res.status_code == 200
        streamed_text = parse_sse_text(res.text)

        # Verify fallback message was streamed
        assert "This isn't covered in company documents. Would you like this flagged to HR?" in streamed_text

        # Verify LLM was NEVER called!
        mock_llm.assert_not_called()


@pytest.mark.asyncio
async def test_3_ephemeral_attachment_prompt_separation_and_distinction(client, sample_user_branch_1):
    """
    Test 3: Paste an unverified notification as attachment with a question.
    Confirm prompt clearly labels [VERIFIED COMPANY DOCUMENT] and [EMPLOYEE-PROVIDED, UNVERIFIED]
    and LLM distinguishes between official vs user-provided content.
    """
    official_chunk = RetrievedChunk(
        chunk_id="doc_leave_c0",
        content="Official annual leave policy grants 25 paid vacation days per calendar year.",
        source_document="Company_Leave_Policy.pdf",
        score=0.82,
        is_company_wide=True,
    )

    unverified_attachment = "Slack message from Manager: You can take an extra 3 days of holiday bonus this December."

    async def mock_stream_attachment(*args, **kwargs):
        # Inspect messages passed to LLM
        messages = args[0] if args else kwargs.get("messages", [])
        user_prompt = messages[1]["content"]
        assert "[VERIFIED COMPANY DOCUMENT]" in user_prompt
        assert "[EMPLOYEE-PROVIDED, UNVERIFIED]" in user_prompt
        assert "Slack message from Manager" in user_prompt
        yield "Based on official documents [Source: Company_Leave_Policy.pdf], standard leave is 25 days. Based on what you shared in your attachment, your manager offered 3 extra bonus days."

    with patch("app.routers.chat.validate_user_context", AsyncMock(return_value=sample_user_branch_1)), \
         patch("app.routers.chat.search_knowledge_base", AsyncMock(return_value=[official_chunk])), \
         patch("app.rag_engine.stream_llm_completion", side_effect=mock_stream_attachment):

        res = client.post(
            "/api/v1/chat",
            data={
                "message": "How many days off can I take in December?",
                "attachment_text": unverified_attachment,
            },
            headers={"Authorization": "Bearer fake-jwt-token"},
        )
        assert res.status_code == 200
        content = res.text
        assert "Based on what you shared" in content
        assert "Company_Leave_Policy.pdf" in content


@pytest.mark.asyncio
async def test_4_attachment_ephemeral_isolation(client, sample_user_branch_1, sample_user_branch_2):
    """
    Test 4: Verify that user attachment is only used in memory for the single request prompt
    and is never ingested into vector DB.
    """
    with patch("app.routers.chat.validate_user_context", AsyncMock(return_value=sample_user_branch_1)), \
         patch("app.routers.chat.search_knowledge_base", AsyncMock(return_value=[])) as mock_search:

        async def mock_stream_ephemeral(*args, **kwargs):
            yield "Based on what you shared, your project budget is $5,000."

        with patch("app.rag_engine.stream_llm_completion", side_effect=mock_stream_ephemeral):
            res = client.post(
                "/api/v1/chat",
                data={
                    "message": "What is my budget?",
                    "attachment_text": "Confidential budget note: $5,000",
                },
                headers={"Authorization": "Bearer fake-jwt-token"},
            )
            assert res.status_code == 200

        # Vector DB search was only called with the query, attachment was never sent to vector DB ingestion
        mock_search.assert_called_once_with(
            query="What is my budget?",
            branch_id=1,
            department_id=1,
            top_k=4,
        )


@pytest.mark.asyncio
async def test_5_invalid_and_expired_jwt_handling(client):
    """
    Test 5: Confirm 401 Unauthorized is returned for invalid JWT, and LLM is never invoked.
    """
    mock_llm = MagicMock()

    with patch("app.auth_client.httpx.AsyncClient.get") as mock_get, \
         patch("app.rag_engine.stream_llm_completion", mock_llm):

        # Simulate identity service returning 401
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Token expired or invalid"
        mock_get.return_value = mock_response

        res = client.post(
            "/api/v1/chat",
            data={"message": "Hello HR"},
            headers={"Authorization": "Bearer invalid-expired-token"},
        )
        assert res.status_code == 401
        assert "invalid or expired" in res.json()["detail"].lower()
        mock_llm.assert_not_called()


@pytest.mark.asyncio
async def test_6_identity_service_unreachable_503(client):
    """
    Test 6: Confirm 503 Service Unavailable is returned when identity-org-service is down.
    """
    with patch("app.auth_client.httpx.AsyncClient.get", side_effect=HTTPException(status_code=503, detail="Identity service is currently unreachable.")):
        res = client.post(
            "/api/v1/chat",
            data={"message": "Hello HR"},
            headers={"Authorization": "Bearer any-token"},
        )
        assert res.status_code == 503
        assert "unreachable" in res.json()["detail"].lower()
