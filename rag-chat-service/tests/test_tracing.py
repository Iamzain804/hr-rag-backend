import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.schemas import RetrievedChunk
from app.tracing import (
    clear_traces,
    get_recent_traces,
    get_trace_by_id,
    mask_pii_and_secrets,
    sanitize_dict,
)


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
def reset_trace_registry():
    """Clear traces before and after each test."""
    clear_traces()
    yield
    clear_traces()


def test_pii_and_secrets_masking():
    """
    CRITICAL PRIVACY TEST:
    Verify that emails, phone numbers, national IDs, JWTs, and API keys are strictly masked.
    """
    sample_text = (
        "Employee John Doe (email: john.doe@company.com, phone: +92-300-1234567, CNIC: 35201-1234567-1). "
        "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMCJ9.signature. "
        "API Key: gsk_test_mock_secret_key_pattern_998877665544332211"
    )

    masked = mask_pii_and_secrets(sample_text)
    assert "john.doe@company.com" not in masked
    assert "[EMAIL_REDACTED]" in masked
    assert "+92-300-1234567" not in masked
    assert "[PHONE_REDACTED]" in masked
    assert "35201-1234567-1" not in masked
    assert "[ID_REDACTED]" in masked
    assert "eyJhbGciOi" not in masked
    assert "[JWT_REDACTED]" in masked
    assert "gsk_test_mock" not in masked
    assert "[API_KEY_REDACTED]" in masked


def test_sanitize_dict_omits_raw_attachments():
    """Verify raw attachment text is omitted from telemetry for privacy."""
    payload = {
        "message": "Please review this document.",
        "attachment_text": "CONFIDENTIAL FINANCIAL SALARY SLIP DETAILS: $120,000",
        "email": "employee@hr.com",
    }
    sanitized = sanitize_dict(payload)
    assert "CONFIDENTIAL" not in sanitized["attachment_text"]
    assert "[ATTACHMENT_OMITTED_FOR_PRIVACY" in sanitized["attachment_text"]
    assert sanitized["email"] == "[EMAIL_REDACTED]"


@pytest.mark.asyncio
async def test_chat_pipeline_end_to_end_tracing(client, sample_user_branch_1):
    """
    Verify that every step of the RAG pipeline produces child spans in the trace:
    1. identity_context_fetch
    2. input_guardrail
    3. cache_lookup
    4. retrieval
    5. rerank
    6. threshold_decision
    7. llm_call (with groq_key_index and latency)
    8. groundedness_check
    9. output_guardrail
    """
    doc_chunk = RetrievedChunk(
        chunk_id="doc_hr_policy_01",
        content="Standard probation period is 3 months from joining date.",
        source_document="HR_Handbook.pdf",
        score=0.88,
        branch_id=1,
        department_id=1,
        is_company_wide=False,
    )

    async def mock_llm_stream(*args, **kwargs):
        yield "Based on [Source: HR_Handbook.pdf], the probation period is 3 months."

    with patch("app.routers.chat.validate_user_context", AsyncMock(return_value=sample_user_branch_1)), \
         patch("app.routers.chat.get_scope_ingestion_version", AsyncMock(return_value="v_test_version")), \
         patch("app.routers.chat.search_knowledge_base", AsyncMock(return_value=[doc_chunk])), \
         patch("app.rag_engine.stream_llm_completion", side_effect=mock_llm_stream):

        res = client.post(
            "/api/v1/chat",
            data={"message": "What is the probation period?"},
            headers={"Authorization": "Bearer fake-jwt-token"},
        )
        assert res.status_code == 200

        # Extract trace_id from meta event
        trace_id = None
        for line in res.text.splitlines():
            if line.startswith("data:") and "trace_id" in line:
                payload = json.loads(line[5:].strip())
                trace_id = payload.get("trace_id")
                break

        assert trace_id is not None

        # Fetch detailed trace via inspection endpoint
        trace_res = client.get(f"/api/v1/chat/traces/{trace_id}")
        assert trace_res.status_code == 200
        trace_data = trace_res.json()

        assert trace_data["trace_id"] == trace_id
        assert trace_data["status"] == "success"
        assert trace_data["total_duration_ms"] >= 0.0

        # Verify child spans
        span_names = [s["name"] for s in trace_data["spans"]]
        assert "identity_context_fetch" in span_names
        assert "input_guardrail" in span_names
        assert "cache_lookup" in span_names
        assert "retrieval" in span_names
        assert "rerank" in span_names
        assert "threshold_decision" in span_names
        assert "llm_call" in span_names
        assert "groundedness_check" in span_names
        assert "output_guardrail" in span_names

        # Verify llm_call span logs groq_key_index and never API key
        llm_span = next(s for s in trace_data["spans"] if s["name"] == "llm_call")
        assert "groq_key_index" in llm_span["outputs"]
        assert isinstance(llm_span["outputs"]["groq_key_index"], int)
        assert "gsk_" not in json.dumps(trace_data)


@pytest.mark.asyncio
async def test_tracing_fail_open_policy(client, sample_user_branch_1):
    """
    FAIL-OPEN TEST:
    Verify that if LangSmith client throws an exception (e.g., cloud timeout or network error),
    the chat request continues normally without raising any error to the client.
    """
    doc_chunk = RetrievedChunk(
        chunk_id="doc_1",
        content="Employees get 10 sick leaves.",
        source_document="Sick_Leave.pdf",
        score=0.90,
    )

    async def mock_llm_stream(*args, **kwargs):
        yield "You get 10 sick leaves. [Source: Sick_Leave.pdf]"

    with patch("app.routers.chat.validate_user_context", AsyncMock(return_value=sample_user_branch_1)), \
         patch("app.routers.chat.get_scope_ingestion_version", AsyncMock(return_value="v1")), \
         patch("app.routers.chat.search_knowledge_base", AsyncMock(return_value=[doc_chunk])), \
         patch("app.rag_engine.stream_llm_completion", side_effect=mock_llm_stream), \
         patch("app.tracing.TraceContext._export_to_langsmith", side_effect=RuntimeError("LangSmith network timeout!")):

        # Chat request must succeed despite LangSmith failure!
        res = client.post(
            "/api/v1/chat",
            data={"message": "How many sick leaves do I get?"},
            headers={"Authorization": "Bearer fake-token"},
        )
        assert res.status_code == 200
        assert "10 sick leaves" in parse_sse_text(res.text)


def test_get_recent_traces_endpoint(client):
    """Verify list traces endpoint returns formatted recent traces."""
    res = client.get("/api/v1/chat/traces")
    assert res.status_code == 200
    assert "traces" in res.json()
    assert isinstance(res.json()["traces"], list)
