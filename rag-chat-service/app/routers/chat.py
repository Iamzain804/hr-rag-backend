import asyncio
import json
import logging
import time
from typing import List, Optional
from fastapi import APIRouter, File, Form, Header, HTTPException, Request, UploadFile, status

from sse_starlette.sse import EventSourceResponse

from app.auth_client import validate_user_context
from app.config import settings
from app.guardrails import check_groundedness_heuristic, validate_input_message, validate_output_message
from app.history_db import (
    add_message,
    create_conversation,
    delete_conversation,
    get_conversation,
    get_conversation_messages,
    get_user_conversations,
    update_conversation_title,
)
from app.ingestion_client import get_scope_ingestion_version, search_knowledge_base
from app.ocr_service import process_attachment_file
from app.rag_engine import execute_rag_pipeline, format_context_prompt
from app.reranker import rerank_chunks
from app.schemas import UserContext
from app.semantic_cache import (
    is_cache_eligible,
    lookup_semantic_cache,
    store_semantic_cache_entry,
)
from app.tracing import TraceContext, get_recent_traces, get_trace_by_id
from app.llm_router import get_active_groq_key_index

logger = logging.getLogger("rag_chat_service.router")

router = APIRouter(prefix="/api/v1/chat", tags=["RAG Chat"])


# --- Tracing & Observability Endpoints (Part C) ---


@router.get("/traces", summary="List recent request traces (LangSmith & Local)")
async def list_recent_traces(limit: int = 50):
    """Retrieve recent request traces with step-by-step latency and span breakdowns."""
    return {"traces": get_recent_traces(limit=limit)}


@router.get("/traces/{trace_id}", summary="Get detailed trace by ID")
async def get_trace_details(trace_id: str):
    """Retrieve full trace breakdown including all child spans for a given request."""
    trace = get_trace_by_id(trace_id)
    if not trace:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trace with ID '{trace_id}' not found.",
        )
    return trace


# --- Conversation Management Endpoints ---


@router.get("/conversations", summary="List user's conversation sessions")
async def list_conversations(authorization: Optional[str] = Header(None)):
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user: UserContext = await validate_user_context(authorization)
    conversations = get_user_conversations(user.user_id)
    return {"conversations": conversations}


@router.post("/conversations", summary="Create a new conversation session")
async def create_new_conversation(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user: UserContext = await validate_user_context(authorization)
    title = "New Chat"
    try:
        body = await request.json()
        if body and "title" in body:
            title = body.get("title") or "New Chat"
    except Exception:
        pass

    conv = create_conversation(user_id=user.user_id, title=title)
    return conv


@router.get("/conversations/{conversation_id}", summary="Get conversation messages")
async def get_conversation_details(
    conversation_id: str,
    authorization: Optional[str] = Header(None),
):
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user: UserContext = await validate_user_context(authorization)
    conv = get_conversation(conversation_id, user.user_id)
    if not conv:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found or access denied.",
        )
    messages = get_conversation_messages(conversation_id)
    return {
        "conversation": conv,
        "messages": messages,
    }


@router.delete("/conversations/{conversation_id}", summary="Delete conversation session")
async def remove_conversation(
    conversation_id: str,
    authorization: Optional[str] = Header(None),
):
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user: UserContext = await validate_user_context(authorization)
    deleted = delete_conversation(conversation_id, user.user_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found or access denied.",
        )
    return {"message": "Conversation deleted successfully"}


# --- Main Streaming Chat Endpoint with Tracing (Part C) ---


@router.post("", summary="Stream contextual HR Chat responses via SSE")
async def chat_endpoint(
    request: Request,
    authorization: Optional[str] = Header(None),
    message: Optional[str] = Form(None),
    attachment_text: Optional[str] = Form(None),
    attachment_file: Optional[UploadFile] = File(None),
    conversation_id: Optional[str] = Form(None),
):
    """
    RAG-powered HR Chat endpoint with Semantic Cache (Part A),
    Cross-Encoder Reranker (Part B), and LangSmith Tracing & Observability (Part C).
    """
    # 1. Parse payload (Supports Form-data or JSON body)
    actual_message = message
    actual_attachment_text = attachment_text
    actual_conversation_id = conversation_id

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            body = await request.json()
            actual_message = body.get("message")
            actual_attachment_text = body.get("attachment_text")
            actual_conversation_id = body.get("conversation_id")
        except Exception:
            pass

    # Initialize Request Trace Context (Part C)
    trace = TraceContext(
        conversation_id=actual_conversation_id,
        initial_inputs={"message": actual_message, "has_attachment": bool(actual_attachment_text or attachment_file)},
    )

    # 2. Validate Authorization (JWT) & Identity Context Fetch Span
    if not authorization:
        trace.complete_trace(outputs={"error": "Missing authorization header"}, status="error")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    trace.start_span("identity_context_fetch")
    try:
        user: UserContext = await validate_user_context(authorization)
        trace.end_span(
            "identity_context_fetch",
            outputs={"user_id": user.user_id, "branch_id": user.branch_id, "department_id": user.department_id, "role": user.role},
        )
        trace.user_id = user.user_id
        trace.branch_id = user.branch_id
        trace.department_id = user.department_id
    except Exception as exc:
        trace.end_span("identity_context_fetch", status="error", error=str(exc))
        trace.complete_trace(outputs={"error": "JWT validation failed"}, status="error")
        raise

    # 3. Input Guardrail Span
    trace.start_span("input_guardrail", span_type="guardrail", inputs={"message": actual_message})
    try:
        validate_input_message(actual_message or "")
        trace.end_span("input_guardrail", outputs={"passed": True})
    except Exception as exc:
        trace.end_span("input_guardrail", status="error", error=str(exc))
        trace.complete_trace(outputs={"error": "Input guardrail rejected"}, status="error")
        raise

    # 4. Ensure Conversation Exists
    conv = None
    if actual_conversation_id:
        conv = get_conversation(actual_conversation_id, user.user_id)

    if not conv:
        title_snippet = (actual_message or "New Chat").strip()[:35]
        if len((actual_message or "").strip()) > 35:
            title_snippet += "..."
        conv = create_conversation(user_id=user.user_id, title=title_snippet, conversation_id=actual_conversation_id)
        actual_conversation_id = conv["id"]
        trace.conversation_id = actual_conversation_id

    # 5. Handle Ephemeral Attachments (OCR or Text)
    extracted_attachment = actual_attachment_text or ""
    if attachment_file is not None and attachment_file.filename:
        file_text = await process_attachment_file(attachment_file)
        if file_text:
            extracted_attachment = (
                f"{extracted_attachment}\n{file_text}" if extracted_attachment else file_text
            )

    # 6. Save User Message to History DB
    add_message(
        conversation_id=actual_conversation_id,
        role="user",
        content=(actual_message or "").strip(),
        attachment_text=extracted_attachment if extracted_attachment else None,
        has_attachment=bool(extracted_attachment),
    )

    # 7. Check Scope Ingestion Version
    scope_ingestion_version = await get_scope_ingestion_version(
        branch_id=user.branch_id, department_id=user.department_id
    )

    # 8. Semantic Cache Lookup Span (Part A)
    has_attachment = bool(extracted_attachment and extracted_attachment.strip())
    trace.start_span(
        "cache_lookup",
        inputs={
            "query": actual_message,
            "branch_id": user.branch_id,
            "department_id": user.department_id,
            "has_attachment": has_attachment,
        },
    )

    if not has_attachment and settings.SEMANTIC_CACHE_ENABLED:
        cached_hit = lookup_semantic_cache(
            query=(actual_message or "").strip(),
            branch_id=user.branch_id,
            department_id=user.department_id,
            current_ingestion_version=scope_ingestion_version,
            threshold=settings.SEMANTIC_CACHE_THRESHOLD,
        )

        trace.end_span(
            "cache_lookup",
            outputs={
                "hit": bool(cached_hit),
                "similarity_score": cached_hit.score if cached_hit else 0.0,
                "cached_id": cached_hit.cache_id if cached_hit else None,
            },
        )

        if cached_hit:
            # Output Guardrail validation on cached response
            trace.start_span("output_guardrail", span_type="guardrail", inputs={"answer": cached_hit.answer})
            validate_output_message(cached_hit.answer)
            trace.end_span("output_guardrail", outputs={"passed": True})

            trace.complete_trace(
                outputs={"served_from_cache": True, "answer": cached_hit.answer, "sources": cached_hit.sources}
            )

            # Stream cached hit to client
            async def cache_hit_event_generator():
                yield {
                    "event": "meta",
                    "data": json.dumps(
                        {
                            "conversation_id": actual_conversation_id,
                            "user_id": user.user_id,
                            "branch_id": user.branch_id,
                            "department_id": user.department_id,
                            "chunks_found": len(cached_hit.sources),
                            "top_score": cached_hit.score,
                            "has_attachment": False,
                            "served_from_cache": True,
                            "cache_similarity": cached_hit.score,
                            "trace_id": trace.trace_id,
                            "sources": cached_hit.sources,
                        }
                    ),
                }

                # Stream cached answer tokens incrementally
                words = cached_hit.answer.split(" ")
                for i, word in enumerate(words):
                    token = word + (" " if i < len(words) - 1 else "")
                    yield {
                        "event": "token",
                        "data": json.dumps({"token": token, "conversation_id": actual_conversation_id}),
                    }
                    await asyncio.sleep(0.015)

                # Save assistant message to History DB
                add_message(
                    conversation_id=actual_conversation_id,
                    role="assistant",
                    content=cached_hit.answer,
                    sources=cached_hit.sources,
                )

                # Emit completion signal
                yield {
                    "event": "done",
                    "data": json.dumps({
                        "status": "complete",
                        "conversation_id": actual_conversation_id,
                        "served_from_cache": True,
                        "trace_id": trace.trace_id,
                    }),
                }

            return EventSourceResponse(cache_hit_event_generator())
    else:
        trace.end_span("cache_lookup", outputs={"hit": False, "reason": "Attachment present or cache disabled"})

    # 9. Cache Miss: Initial Vector Retrieval Span (Top-N Candidates)
    trace.start_span("retrieval", inputs={"query": actual_message, "top_k": settings.INITIAL_RETRIEVAL_TOP_K})
    raw_chunks = await search_knowledge_base(
        query=(actual_message or "").strip(),
        branch_id=user.branch_id,
        department_id=user.department_id,
        top_k=settings.INITIAL_RETRIEVAL_TOP_K,
    )
    trace.end_span(
        "retrieval",
        outputs={
            "candidates_count": len(raw_chunks),
            "chunk_ids": [c.chunk_id for c in raw_chunks],
            "vector_scores": [c.score for c in raw_chunks],
        },
    )

    # 10. Cross-Encoder Reranking Span (Part B)
    trace.start_span("rerank", inputs={"candidates_count": len(raw_chunks), "target_top_k": settings.RERANK_TOP_K})
    retrieved_chunks = rerank_chunks(
        query=(actual_message or "").strip(),
        chunks=raw_chunks,
        top_k=settings.RERANK_TOP_K,
    )
    trace.end_span(
        "rerank",
        outputs={
            "reranked_count": len(retrieved_chunks),
            "scores_before": [c.score for c in raw_chunks],
            "scores_after": [c.rerank_score for c in retrieved_chunks],
            "selected_chunk_ids": [c.chunk_id for c in retrieved_chunks],
        },
    )

    sources_metadata = [
        {
            "source_document": c.source_document,
            "chunk_index": c.chunk_index,
            "score": c.score,
            "rerank_score": c.rerank_score,
            "branch_id": c.branch_id,
            "department_id": c.department_id,
            "is_company_wide": c.is_company_wide,
        }
        for c in retrieved_chunks
    ]

    # 11. Threshold Decision Span
    top_score = max(
        [(c.rerank_score if c.rerank_score is not None else c.score) for c in retrieved_chunks],
        default=0.0,
    )
    effective_threshold = (
        settings.RERANK_CONFIDENCE_THRESHOLD if any(c.rerank_score is not None for c in retrieved_chunks)
        else settings.CONFIDENCE_THRESHOLD
    )
    proceed_to_llm = bool(top_score >= effective_threshold or has_attachment)

    trace.start_span(
        "threshold_decision",
        inputs={"top_score": top_score, "threshold": effective_threshold, "has_attachment": has_attachment},
    )
    trace.end_span(
        "threshold_decision",
        outputs={"proceed_to_llm": proceed_to_llm, "top_score": top_score},
    )

    # 12. Stream SSE Response & Populate Semantic Cache if Eligible
    async def event_generator():
        # Emit initial metadata
        yield {
            "event": "meta",
            "data": json.dumps(
                {
                    "conversation_id": actual_conversation_id,
                    "user_id": user.user_id,
                    "branch_id": user.branch_id,
                    "department_id": user.department_id,
                    "chunks_found": len(retrieved_chunks),
                    "top_score": top_score,
                    "has_attachment": bool(extracted_attachment),
                    "served_from_cache": False,
                    "trace_id": trace.trace_id,
                    "sources": sources_metadata,
                }
            ),
        }

        # LLM Call Span
        trace.start_span("llm_call", span_type="llm", inputs={"model": settings.DEFAULT_MODEL})
        llm_start = time.time()
        ttft_recorded = False
        time_to_first_token_ms = 0.0

        # Stream token chunks and accumulate full response
        response_tokens = []
        async for token in execute_rag_pipeline(
            user_message=(actual_message or "").strip(),
            user=user,
            retrieved_chunks=retrieved_chunks,
            attachment_text=extracted_attachment,
        ):
            if not ttft_recorded:
                time_to_first_token_ms = round((time.time() - llm_start) * 1000, 2)
                ttft_recorded = True

            response_tokens.append(token)
            yield {
                "event": "token",
                "data": json.dumps({"token": token, "conversation_id": actual_conversation_id}),
            }

        full_assistant_reply = "".join(response_tokens)
        key_idx = get_active_groq_key_index()

        trace.end_span(
            "llm_call",
            outputs={
                "groq_key_index": key_idx,
                "model": settings.DEFAULT_MODEL,
                "tokens_count": len(response_tokens),
                "ttft_ms": time_to_first_token_ms,
            },
        )

        # Groundedness Check Span
        context_text = format_context_prompt(retrieved_chunks, extracted_attachment)
        is_grounded = check_groundedness_heuristic(full_assistant_reply, context_text)
        trace.start_span("groundedness_check")
        trace.end_span("groundedness_check", outputs={"is_grounded": is_grounded})

        # Output Guardrail validation Span
        if full_assistant_reply:
            trace.start_span("output_guardrail", span_type="guardrail", inputs={"answer": full_assistant_reply})
            validate_output_message(full_assistant_reply)
            trace.end_span("output_guardrail", outputs={"passed": True})

        # Evaluate Cache Eligibility & Store in Semantic Cache
        if is_cache_eligible(
            has_attachment=bool(extracted_attachment),
            top_score=top_score,
            is_grounded=is_grounded,
            answer=full_assistant_reply,
        ):
            store_semantic_cache_entry(
                query=(actual_message or "").strip(),
                answer=full_assistant_reply,
                branch_id=user.branch_id,
                department_id=user.department_id,
                sources=sources_metadata,
                ingestion_version=scope_ingestion_version,
            )

        # Save Assistant Message to History DB
        if full_assistant_reply:
            add_message(
                conversation_id=actual_conversation_id,
                role="assistant",
                content=full_assistant_reply,
                sources=sources_metadata,
            )

        # Finalize and export Trace
        trace.complete_trace(
            outputs={
                "served_from_cache": False,
                "answer": full_assistant_reply,
                "sources": sources_metadata,
                "grounded": is_grounded,
            }
        )

        # Emit completion signal
        yield {
            "event": "done",
            "data": json.dumps({
                "status": "complete",
                "conversation_id": actual_conversation_id,
                "served_from_cache": False,
                "trace_id": trace.trace_id,
            }),
        }

    return EventSourceResponse(event_generator())


