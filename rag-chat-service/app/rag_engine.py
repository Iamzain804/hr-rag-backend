import asyncio
import logging
from typing import AsyncGenerator, List, Optional
from app.config import settings
from app.guardrails import check_groundedness_heuristic, validate_output_message
from app.llm_router import stream_llm_completion
from app.schemas import RetrievedChunk, UserContext

logger = logging.getLogger("rag_chat_service.engine")


def format_context_prompt(
    retrieved_chunks: List[RetrievedChunk],
    attachment_text: Optional[str] = None,
) -> str:
    """
    Format prompt context with strictly separated and labeled sections:
    1. [VERIFIED COMPANY DOCUMENT]
    2. [EMPLOYEE-PROVIDED, UNVERIFIED]
    """
    sections = []

    if retrieved_chunks:
        doc_lines = ["[VERIFIED COMPANY DOCUMENT]"]
        for idx, chunk in enumerate(retrieved_chunks, 1):
            source = chunk.source_document
            b_info = f"Branch #{chunk.branch_id}" if chunk.branch_id else "Global"
            d_info = f"Dept #{chunk.department_id}" if chunk.department_id else "All Depts"
            doc_lines.append(
                f"- Document: '{source}' ({b_info} / {d_info}) [Chunk {chunk.chunk_index}]:\n  \"{chunk.content}\""
            )
        sections.append("\n".join(doc_lines))

    if attachment_text and attachment_text.strip():
        attach_lines = [
            "[EMPLOYEE-PROVIDED, UNVERIFIED]",
            f"\"{attachment_text.strip()}\"",
        ]
        sections.append("\n".join(attach_lines))

    return "\n\n".join(sections)


def build_system_prompt(user: UserContext) -> str:
    """
    Build the grounded system prompt with caller role/branch context.
    """
    branch_desc = user.branch_name or f"Branch #{user.branch_id}" if user.branch_id else "Global / All Branches"
    dept_desc = user.department_name or f"Department #{user.department_id}" if user.department_id else "All Departments"

    return (
        f"You are the official HR AI Assistant for our company. You are assisting {user.full_name} "
        f"(Role: {user.role}, Branch: {branch_desc}, Department: {dept_desc}).\n\n"
        "STRICT GROUNDING & ANSWERING RULES:\n"
        "1. You must answer questions STRICTLY and ONLY from the two provided context sections: [VERIFIED COMPANY DOCUMENT] and [EMPLOYEE-PROVIDED, UNVERIFIED].\n"
        "2. Do NOT mention internal file names, document paths, or source citations (e.g. do NOT write [Source: ...] or 【Source: ...】). Keep your response clean, direct, and conversational.\n"
        "3. If you are relying on the [EMPLOYEE-PROVIDED, UNVERIFIED] section (such as an uploaded screenshot or pasted text), you MUST explicitly state: 'Based on what you shared...' or 'Based on the attached document...'.\n"
        "4. If neither section contains the answer, you must clearly state: 'I don't have this information in company documents.' Do NOT assume, invent, or guess details.\n"
        "5. Be concise, polite, professional, and clear.\n"
        "6. Language Matching: If the employee asks in Roman Urdu (e.g. 'leaves kitni hain', 'policy kya hai') or Urdu or English, respond naturally and fluently in the SAME language/style (Roman Urdu if asked in Roman Urdu), while strictly preserving all facts, numbers, and policies from the context."
    )


async def execute_rag_pipeline(
    user_message: str,
    user: UserContext,
    retrieved_chunks: List[RetrievedChunk],
    attachment_text: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Execute the RAG generation pipeline with confidence threshold check,
    prompt assembly, LLM streaming, and groundedness heuristic validation.
    """
    # 1. Evaluate Retrieval / Rerank Confidence Threshold
    has_valid_attachment = bool(attachment_text and attachment_text.strip())

    has_reranked = any(c.rerank_score is not None for c in retrieved_chunks)
    top_score = max(
        [(c.rerank_score if c.rerank_score is not None else c.score) for c in retrieved_chunks],
        default=0.0,
    )
    effective_threshold = (
        settings.RERANK_CONFIDENCE_THRESHOLD if has_reranked else settings.CONFIDENCE_THRESHOLD
    )

    # If top score is below threshold AND there is no valid attachment -> DO NOT CALL LLM
    if top_score < effective_threshold and not has_valid_attachment:
        logger.info(
            f"Top relevance score {top_score:.4f} (reranked={has_reranked}) is below threshold {effective_threshold:.4f}. LLM call skipped."
        )
        # Stream fixed fallback message incrementally
        fallback = settings.FALLBACK_MESSAGE
        words = fallback.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0.02)
        return


    # 2. Construct Augmented Prompt
    context_text = format_context_prompt(retrieved_chunks, attachment_text)
    system_prompt = build_system_prompt(user)

    user_prompt = (
        f"Context Information:\n{context_text}\n\n"
        f"Employee Question: {user_message}\n\n"
        "Please provide an accurate, grounded answer based strictly on the above context."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # 3. Stream from LiteLLM Router
    collected_tokens: List[str] = []
    async for token in stream_llm_completion(messages):
        collected_tokens.append(token)
        yield token

    # 4. Post-Generation Groundedness Heuristic Check
    full_response = "".join(collected_tokens)
    is_grounded = check_groundedness_heuristic(full_response, context_text)
    if not is_grounded:
        logger.warning("Post-generation groundedness heuristic failed for response.")
