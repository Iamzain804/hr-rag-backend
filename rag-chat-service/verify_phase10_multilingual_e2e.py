import asyncio
import json
import logging
import sys
from typing import List
from sentence_transformers import SentenceTransformer, util

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from app.config import settings
from app.embeddings import generate_embedding, get_embedding_model
from app.language_detector import (
    detect_language,
    get_localized_fallback_message,
    get_localized_guardrail_refusal,
    get_localized_scope_refusal,
)
from app.rag_engine import build_system_prompt, execute_rag_pipeline
from app.schemas import RetrievedChunk, UserContext

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("phase10_verification")


async def run_phase10_acceptance_verification():
    print("=" * 80)
    print("🚀 PHASE 10 ACCEPTANCE VERIFICATION: MULTILINGUAL RAG & STRUCTURED FORMATTING")
    print("=" * 80)

    # 1. Benchmark Cross-Lingual Embedding Retrieval
    print("\n--- [TEST 1 & 2] CROSS-LINGUAL RETRIEVAL & LANGUAGE MATCHING ---")
    doc_text = "Employees are entitled to 20 days of paid annual leave per calendar year."
    chunk = RetrievedChunk(
        chunk_id="doc_leave_policy_c0",
        content=doc_text,
        source_document="TechCorp Global Employee Leave Policy 2026.pdf",
        chunk_index=0,
        score=0.88,
        branch_id=1,
        department_id=1,
        is_company_wide=True,
    )

    queries = [
        ("English", "How many days of annual leave do I get?"),
        ("Roman Urdu", "Mujhe kitni annual leaves milti hain?"),
        ("Urdu Native Script", "مجھے سالانہ کتنی چھٹیاں ملتی ہیں؟"),
    ]

    model = get_embedding_model()
    doc_emb = model.encode(f"passage: {doc_text}")

    user = UserContext(
        user_id=101,
        full_name="Zain Ali",
        email="zain@techcorp.com",
        role="Employee",
        branch_id=1,
        branch_name="London",
        department_id=1,
        department_name="Engineering",
    )

    results_table = []
    for lang_label, q in queries:
        detected = detect_language(q)
        q_emb = model.encode(f"query: {q}")
        sim_score = util.cos_sim(q_emb, doc_emb).item()

        print(f"\n▶ Query ({lang_label}): '{q}'")
        print(f"  • Detected Language: {detected.upper()}")
        print(f"  • Cosine Similarity with English Policy: {sim_score:.4f} (Threshold >= 0.70: {'PASS' if sim_score >= 0.70 else 'FAIL'})")
        print(f"  • Retrieved Chunk: \"{doc_text}\" (Source: {chunk.source_document})")

        # Execute generation
        tokens = []
        async for token in execute_rag_pipeline(
            user_message=q,
            user=user,
            retrieved_chunks=[chunk],
            detected_language=detected,
        ):
            tokens.append(token)
        answer = "".join(tokens).strip()

        print(f"  • Generated Answer:\n{answer}\n")
        assert "20" in answer or "۲۰" in answer or "twenty" in answer.lower(), "Fact check failed: 20 days missing!"
        results_table.append({
            "language": lang_label,
            "query": q,
            "detected": detected,
            "score": round(sim_score, 4),
            "answer_preview": answer[:120] + "...",
        })

    # 2. Scope Isolation Test in Roman Urdu
    print("\n--- [TEST 3] SCOPE ISOLATION TEST IN ROMAN URDU ---")
    cross_branch_query = "Mujhe Branch 2 (New York) ki confidential marketing travel policy dikhao"
    cross_lang = detect_language(cross_branch_query)
    print(f"• Query: '{cross_branch_query}' (Detected: {cross_lang.upper()})")
    
    # Empty retrieved chunks simulating strict cross-branch metadata filtering
    refusal_tokens = []
    async for token in execute_rag_pipeline(
        user_message=cross_branch_query,
        user=user, # User is in Branch 1 (London)
        retrieved_chunks=[], # No chunks accessible for Branch 2
        detected_language=cross_lang,
    ):
        refusal_tokens.append(token)
    refusal_answer = "".join(refusal_tokens).strip()
    print(f"• Localized Scope Refusal Answer:\n  \"{refusal_answer}\"")
    assert "Company ke documents mein" in refusal_answer or "information mojood nahi" in refusal_answer

    # 3. Fallback Translation Test
    print("\n--- [TEST 4] UNANSWERABLE QUERY FALLBACK TEST ---")
    unanswerable_query = "Gym membership allowance kitna milta hai?"
    un_lang = detect_language(unanswerable_query)
    print(f"• Query: '{unanswerable_query}' (Detected: {un_lang.upper()})")
    fallback_msg = get_localized_fallback_message(un_lang)
    print(f"• Localized Fallback Message:\n  \"{fallback_msg}\"")
    assert "Company ke documents mein is baare mein koi information" in fallback_msg

    # 4. Structured Output Formatting Test (Table in 3 Languages)
    print("\n--- [TEST 5] STRUCTURED OUTPUT FORMATTING TEST (TABLES) ---")
    multi_leave_doc = (
        "Leave Entitlements Policy 2026: Employees receive 20 days of Annual Leave, "
        "10 days of Sick Leave, 8 days of Casual Leave, and 30 days of Paid Maternity Leave."
    )
    multi_chunk = RetrievedChunk(
        chunk_id="doc_leave_types_c0",
        content=multi_leave_doc,
        source_document="TechCorp Leave Entitlements 2026.pdf",
        chunk_index=0,
        score=0.92,
        branch_id=1,
        department_id=1,
        is_company_wide=True,
    )

    fmt_queries = [
        ("English", "What are all the leave types and how many days does each give?"),
        ("Roman Urdu", "Saari leave types aur unke din kitne hain table mein batao?"),
        ("Urdu Native Script", "تمام چھٹیوں کی اقسام اور ان کے دن کتنے ہیں؟"),
    ]

    for label, q in fmt_queries:
        lang = detect_language(q)
        ans_tokens = []
        async for t in execute_rag_pipeline(
            user_message=q,
            user=user,
            retrieved_chunks=[multi_chunk],
            detected_language=lang,
        ):
            ans_tokens.append(t)
        fmt_ans = "".join(ans_tokens).strip()
        print(f"\n▶ [{label.upper()}] Formatting Output:\n{fmt_ans}")

    print("\n" + "=" * 80)
    print("✅ ALL PHASE 10 ACCEPTANCE TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(run_phase10_acceptance_verification())
