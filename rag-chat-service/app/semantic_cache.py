import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from pydantic import BaseModel

from app.config import settings
from app.embeddings import generate_embedding

logger = logging.getLogger("rag_chat_service.semantic_cache")

_cache_client: Optional[chromadb.ClientAPI] = None
_cache_collection: Optional[chromadb.Collection] = None


class CachedResponse(BaseModel):
    cache_id: str
    query: str
    answer: str
    sources: List[Dict[str, Any]]
    score: float
    branch_id: Optional[int] = None
    department_id: Optional[int] = None
    ingestion_version: str
    created_at: float
    ttl_seconds: int


def get_cache_collection() -> chromadb.Collection:
    """Initialize and retrieve the persistent ChromaDB collection for semantic cache."""
    global _cache_client, _cache_collection
    if _cache_collection is None:
        os.makedirs(settings.SEMANTIC_CACHE_DIR, exist_ok=True)
        if _cache_client is None:
            _cache_client = chromadb.PersistentClient(
                path=settings.SEMANTIC_CACHE_DIR,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        _cache_collection = _cache_client.get_or_create_collection(
            name=settings.SEMANTIC_CACHE_COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Initialized Semantic Cache ChromaDB collection: {settings.SEMANTIC_CACHE_COLLECTION}")
    return _cache_collection


def clear_cache() -> None:
    """Purge all entries from the semantic cache (useful for clean testing)."""
    global _cache_collection
    try:
        col = get_cache_collection()
        if col.count() > 0:
            col.delete(where={"created_at": {"$gte": 0.0}})
        logger.info("Semantic cache cleared.")
    except Exception as exc:
        logger.warning(f"Error clearing semantic cache: {exc}")


def is_cache_eligible(
    has_attachment: bool,
    top_score: float,
    is_grounded: bool,
    answer: str,
) -> bool:
    """
    Check if a generated response meets all eligibility rules for semantic caching:
    1. Must NOT have user attachments (ephemeral, unverified).
    2. Must NOT be a fallback 'not covered in company documents' response.
    3. Retrieval confidence must be >= threshold.
    4. Groundedness check must have PASSED.
    5. Answer must not be empty.
    """
    if has_attachment:
        return False

    if not is_grounded:
        return False

    if top_score < settings.CONFIDENCE_THRESHOLD:
        return False

    clean_answer = answer.strip()
    if not clean_answer:
        return False

    # Check fallback phrasing
    if settings.FALLBACK_MESSAGE.lower() in clean_answer.lower():
        return False

    if "isn't covered in company documents" in clean_answer.lower():
        return False

    return True


def lookup_semantic_cache(
    query: str,
    branch_id: Optional[int],
    department_id: Optional[int],
    current_ingestion_version: str,
    threshold: Optional[float] = None,
) -> Optional[CachedResponse]:
    """
    Lookup a query in the semantic cache with strict scope isolation,
    cosine similarity thresholding, TTL validation, and ingestion version check.
    """
    if not settings.SEMANTIC_CACHE_ENABLED:
        return None

    clean_query = query.strip()
    if not clean_query:
        return None

    sim_threshold = threshold if threshold is not None else settings.SEMANTIC_CACHE_THRESHOLD

    # Normalization of scope tags for strict equality filter
    b_tag = branch_id if branch_id is not None else -1
    d_tag = department_id if department_id is not None else -1

    collection = get_cache_collection()
    if collection.count() == 0:
        return None

    query_embedding = generate_embedding(clean_query)

    # Strict Scope Isolation: Filter by exact scope tags BEFORE similarity match
    where_scope = {
        "$and": [
            {"branch_id": b_tag},
            {"department_id": d_tag},
        ]
    }

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=1,
            where=where_scope,
        )
    except Exception as exc:
        logger.warning(f"Semantic cache lookup query failed: {exc}")
        return None

    if not results or "documents" not in results or not results["documents"] or not results["documents"][0]:
        return None

    cached_doc = results["documents"][0][0]
    cached_id = results["ids"][0][0]
    cached_meta = results["metadatas"][0][0]
    distance = results["distances"][0][0] if "distances" in results and results["distances"] else 1.0

    # Calculate cosine similarity score (distance is in [0, 2])
    score = round(max(0.0, 1.0 - distance), 4)

    # 1. Similarity Threshold Check (conservative ~0.90)
    if score < sim_threshold:
        logger.debug(
            f"Semantic cache near-match for '{clean_query}' score {score:.4f} < threshold {sim_threshold:.4f} (Miss)."
        )
        return None

    # 2. TTL Expiry Check
    created_at = float(cached_meta.get("created_at", 0.0))
    ttl = int(cached_meta.get("ttl_seconds", settings.SEMANTIC_CACHE_TTL_SECONDS))
    now = time.time()
    if now - created_at > ttl:
        logger.info(f"Semantic cache entry '{cached_id}' expired by TTL ({now - created_at:.1f}s > {ttl}s). Purging.")
        try:
            collection.delete(ids=[cached_id])
        except Exception:
            pass
        return None

    # 3. Scope Ingestion Version Invalidation Check
    stored_ingestion_version = cached_meta.get("ingestion_version", "")
    if current_ingestion_version and stored_ingestion_version != current_ingestion_version:
        logger.info(
            f"Semantic cache entry '{cached_id}' invalidated due to scope ingestion version mismatch "
            f"(stored: {stored_ingestion_version} vs current: {current_ingestion_version}). Purging."
        )
        try:
            collection.delete(ids=[cached_id])
        except Exception:
            pass
        return None

    # Parse stored sources
    raw_sources = cached_meta.get("sources_json", "[]")
    try:
        sources = json.loads(raw_sources)
    except Exception:
        sources = []

    answer = cached_meta.get("answer", cached_doc)

    logger.info(
        f"Semantic Cache HIT [score={score:.4f}, id={cached_id}, scope=(b:{b_tag}, d:{d_tag})] for query: '{clean_query}'"
    )

    return CachedResponse(
        cache_id=cached_id,
        query=cached_doc,
        answer=answer,
        sources=sources,
        score=score,
        branch_id=branch_id,
        department_id=department_id,
        ingestion_version=stored_ingestion_version,
        created_at=created_at,
        ttl_seconds=ttl,
    )


def store_semantic_cache_entry(
    query: str,
    answer: str,
    branch_id: Optional[int],
    department_id: Optional[int],
    sources: List[Dict[str, Any]],
    ingestion_version: str,
    ttl_seconds: Optional[int] = None,
) -> Optional[str]:
    """
    Store an eligible question/answer pair into the semantic cache collection.
    """
    if not settings.SEMANTIC_CACHE_ENABLED:
        return None

    clean_query = query.strip()
    clean_answer = answer.strip()
    if not clean_query or not clean_answer:
        return None

    ttl = ttl_seconds if ttl_seconds is not None else settings.SEMANTIC_CACHE_TTL_SECONDS
    b_tag = branch_id if branch_id is not None else -1
    d_tag = department_id if department_id is not None else -1

    # Deterministic ID based on query text and scope
    query_hash = hashlib.sha256(f"{clean_query.lower()}:{b_tag}:{d_tag}".encode("utf-8")).hexdigest()[:16]
    cache_id = f"cache_{b_tag}_{d_tag}_{query_hash}"

    query_embedding = generate_embedding(clean_query)
    now = time.time()

    metadata = {
        "branch_id": b_tag,
        "department_id": d_tag,
        "created_at": now,
        "ttl_seconds": ttl,
        "ingestion_version": ingestion_version,
        "sources_json": json.dumps(sources, default=str),
        "answer": clean_answer,
    }

    collection = get_cache_collection()
    try:
        collection.upsert(
            ids=[cache_id],
            documents=[clean_query],
            embeddings=[query_embedding],
            metadatas=[metadata],
        )
        logger.info(f"Stored Semantic Cache entry '{cache_id}' for scope (b:{b_tag}, d:{d_tag})")
        return cache_id
    except Exception as exc:
        logger.warning(f"Failed to store semantic cache entry: {exc}")
        return None
