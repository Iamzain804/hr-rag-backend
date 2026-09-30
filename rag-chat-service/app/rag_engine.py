import asyncio
import logging
from typing import AsyncGenerator, List, Optional
from app.config import settings
from app.guardrails import check_groundedness_heuristic, validate_output_message
from app.language_detector import detect_language, get_localized_fallback_message
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


def build_system_prompt(user: UserContext, detected_language: Optional[str] = None) -> str:
    """
    Build the grounded system prompt with caller role/branch context,
    strict language matching, and structured output formatting instructions.
    """
    branch_desc = user.branch_name or f"Branch #{user.branch_id}" if user.branch_id else "Global / All Branches"
    dept_desc = user.department_name or f"Department #{user.department_id}" if user.department_id else "All Departments"

    target_lang_str = f"Target Response Language for this query: {detected_language.upper()}." if detected_language else "Match incoming query language."
    lang_instruction = (
        f"LANGUAGE & SCRIPT MATCHING (CRITICAL - {target_lang_str}):\n"
        "- If the user asks in Urdu native script (e.g. 'چھٹیاں کتنی ہیں'), you MUST respond ENTIRELY in native Urdu script (نستعلیق / اردو رسم الخط).\n"
        "- If the user asks in Roman Urdu (e.g. 'leaves kitni hain', 'policy kya hai', 'duty timings batayein'), you MUST respond ENTIRELY in natural, fluent Roman Urdu.\n"
        "- If the user asks in English, you MUST respond in English.\n"
        "- NEVER switch languages silently. Always match the exact language and script used by the user.\n\n"
    )

    formatting_instruction = (
        "STRUCTURED OUTPUT FORMATTING (PROFESSIONAL ASSISTANT STYLE):\n"
        "- Bold Lead-in / Short Heading: Begin with a concise bold lead-in or short header indicating the policy or topic (e.g. **Annual Leave Policy:** or **سالانہ چھٹیوں کی تفصیلات:**).\n"
        "- Bullet Points: Use clean markdown bullet points (`- `) for listing rules, conditions, or multi-step procedures.\n"
        "- Markdown Tables: Whenever the answer contains multiple numbers, allocations, or comparative items (e.g., leave types and allocated days, salary bands, timings across shifts), present them in a clean Markdown table (`| Category | Details |`).\n"
        "- Concise & Direct: For simple single-fact questions, provide a direct answer without unnecessary fluff or excessive headings.\n\n"
    )

    grounding_rules = (
        "STRICT GROUNDING RULES:\n"
        "1. You must answer questions STRICTLY and ONLY from the two provided context sections: [VERIFIED COMPANY DOCUMENT] and [EMPLOYEE-PROVIDED, UNVERIFIED].\n"
        "2. Do NOT mention internal file names, document paths, or source citations (e.g. do NOT write [Source: ...] or 【Source: ...】). Keep your response clean, direct, and conversational.\n"
        "3. If relying on [EMPLOYEE-PROVIDED, UNVERIFIED] context (uploaded screenshot / pasted text), explicitly state: 'Based on what you shared...' (or in Roman Urdu 'Aapke share kiye gaye document ke mutabiq...' / Urdu 'آپ کی شیئر کردہ تفصیلات کے مطابق...').\n"
        "4. If neither section contains the answer, you must clearly state in the matching language that this is not covered in company documents. Do NOT invent or guess details.\n"
    )

    return (
        f"You are the official HR AI Assistant for our company. You are assisting {user.full_name} "
        f"(Role: {user.role}, Branch: {branch_desc}, Department: {dept_desc}).\n\n"
        f"{lang_instruction}"
        f"{formatting_instruction}"
        f"{grounding_rules}"
    )


async def execute_rag_pipeline(
    user_message: str,
    user: UserContext,
    retrieved_chunks: List[RetrievedChunk],
    attachment_text: Optional[str] = None,
    detected_language: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Execute the RAG generation pipeline with confidence threshold check,
    prompt assembly, LLM streaming, localized fallback, and groundedness validation.
    """
    if not detected_language:
        detected_language = detect_language(user_message)

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
        # Stream localized fallback message incrementally in the detected language
        fallback = get_localized_fallback_message(detected_language)
        words = fallback.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0.02)
        return

    # 2. Construct Augmented Prompt
    context_text = format_context_prompt(retrieved_chunks, attachment_text)
    system_prompt = build_system_prompt(user, detected_language=detected_language)

    user_prompt = (
        f"Context Information:\n{context_text}\n\n"
        f"Employee Question: {user_message}\n\n"
        f"Please provide an accurate, structured answer based strictly on the above context in the same language and script as the question."
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
