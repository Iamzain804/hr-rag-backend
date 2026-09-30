import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.chunker import chunk_text
from app.config import settings
from app.embeddings import generate_embedding, generate_embeddings_batch
from app.schemas import DocumentItem, IngestionResponse, SearchResultItem

logger = logging.getLogger("ingestion_service")

_client: Optional[chromadb.ClientAPI] = None
_collection: Optional[chromadb.Collection] = None


def get_vector_collection() -> chromadb.Collection:
    """Initialize and retrieve the persistent ChromaDB collection."""
    global _client, _collection
    if _collection is None:
        os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
        if _client is None:
            _client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIR,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        _collection = _client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"Initialized ChromaDB collection: {settings.COLLECTION_NAME}")
    return _collection


def clear_all_data() -> None:
    """Reset all collection data and document registry (for clean test isolation)."""
    global _client, _collection
    try:
        col = get_vector_collection()
        if col.count() > 0:
            # Delete all documents from collection
            col.delete(where={"chunk_index": {"$gte": 0}})
    except Exception as e:
        logger.warning(f"Error resetting collection: {e}")

    # Remove document registry file
    registry_path = _get_registry_path()
    if os.path.exists(registry_path):
        try:
            os.remove(registry_path)
        except Exception:
            pass


def _get_registry_path() -> str:
    return os.path.join(settings.CHROMA_PERSIST_DIR, "document_registry.json")


def _load_registry() -> Dict[str, Dict[str, Any]]:
    path = _get_registry_path()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_registry(registry: Dict[str, Dict[str, Any]]) -> None:
    path = _get_registry_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, default=str)


def ingest_document(
    title: str,
    source_document: str,
    raw_text: str,
    branch_id: Optional[int] = None,
    department_id: Optional[int] = None,
) -> IngestionResponse:
    """
    Ingest a document into the vector store with idempotency dedup.
    
    Idempotency Approach:
    - SHA-256 hash of raw content generates unique doc signature.
    - Key composed of hash + branch_id + department_id ensures scoped uniqueness.
    - Re-ingestion removes any prior chunks for that document before inserting new ones,
      guaranteeing no duplicate chunks accumulate in the vector database.
    """
    clean_text = raw_text.strip()
    if not clean_text:
        raise ValueError("Cannot ingest empty document content.")

    doc_hash = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
    is_company_wide = branch_id is None and department_id is None

    # Deterministic Document ID based on content hash and target org tags
    b_tag = f"b{branch_id}" if branch_id is not None else "ball"
    d_tag = f"d{department_id}" if department_id is not None else "dall"
    doc_id = f"doc_{doc_hash[:16]}_{b_tag}_{d_tag}"

    collection = get_vector_collection()
    registry = _load_registry()

    # If document was previously ingested, purge old chunk IDs to enforce idempotency
    if doc_id in registry:
        old_chunk_ids = registry[doc_id].get("chunk_ids", [])
        if old_chunk_ids:
            try:
                collection.delete(ids=old_chunk_ids)
                logger.info(f"Purged {len(old_chunk_ids)} existing chunks for idempotent re-ingestion of doc {doc_id}")
            except Exception as e:
                logger.warning(f"Error purging old chunks: {e}")

    # 1. Chunk document
    chunks = chunk_text(clean_text)
    if not chunks:
        raise ValueError("Chunker produced 0 valid text chunks.")

    # 2. Generate local embeddings
    embeddings = generate_embeddings_batch(chunks)

    # 3. Construct chunk IDs and metadata
    chunk_ids = [f"{doc_id}_c{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "doc_id": doc_id,
            "title": title,
            "source_document": source_document,
            "branch_id": branch_id if branch_id is not None else -1,
            "department_id": department_id if department_id is not None else -1,
            "is_company_wide": is_company_wide,
            "chunk_index": i,
            "doc_hash": doc_hash,
            "char_count": len(chunk),
        }
        for i, chunk in enumerate(chunks)
    ]

    # 4. Insert into ChromaDB collection
    collection.upsert(
        ids=chunk_ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    # 5. Update Registry
    registry[doc_id] = {
        "doc_id": doc_id,
        "title": title,
        "source_document": source_document,
        "branch_id": branch_id,
        "department_id": department_id,
        "is_company_wide": is_company_wide,
        "chunk_count": len(chunks),
        "chunk_ids": chunk_ids,
        "doc_hash": doc_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_registry(registry)

    return IngestionResponse(
        status="success",
        message=f"Document '{title}' successfully ingested into {len(chunks)} chunks.",
        doc_id=doc_id,
        title=title,
        source_document=source_document,
        branch_id=branch_id,
        department_id=department_id,
        is_company_wide=is_company_wide,
        chunk_count=len(chunks),
        doc_hash=doc_hash,
    )


def list_ingested_documents() -> List[DocumentItem]:
    """Retrieve all ingested documents from the registry."""
    registry = _load_registry()
    docs = []
    for doc in registry.values():
        docs.append(
            DocumentItem(
                doc_id=doc["doc_id"],
                title=doc["title"],
                source_document=doc["source_document"],
                branch_id=doc.get("branch_id"),
                department_id=doc.get("department_id"),
                is_company_wide=doc.get("is_company_wide", True),
                chunk_count=doc.get("chunk_count", 0),
                doc_hash=doc.get("doc_hash", ""),
                created_at=doc.get("created_at", ""),
            )
        )
    return docs


def get_scope_ingestion_version(
    branch_id: Optional[int] = None, department_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Compute a deterministic ingestion version signature for a given scope.
    Includes all documents matching:
    1. Global / Company-wide, OR
    2. branch_id, OR
    3. department_id
    """
    registry = _load_registry()
    relevant_doc_signatures: List[str] = []
    last_updated: Optional[str] = None

    for doc in registry.values():
        doc_b = doc.get("branch_id")
        doc_d = doc.get("department_id")
        is_global = doc.get("is_company_wide", True) or (doc_b is None and doc_d is None)

        matches_branch = branch_id is not None and doc_b == branch_id
        matches_dept = department_id is not None and doc_d == department_id

        if is_global or matches_branch or matches_dept:
            doc_created = doc.get("created_at", "")
            doc_hash = doc.get("doc_hash", "")
            relevant_doc_signatures.append(f"{doc['doc_id']}:{doc_created}:{doc_hash}")
            if not last_updated or doc_created > last_updated:
                last_updated = doc_created

    if not relevant_doc_signatures:
        return {
            "scope_version": "v0_empty",
            "document_count": 0,
            "last_updated": None,
            "branch_id": branch_id,
            "department_id": department_id,
        }

    relevant_doc_signatures.sort()
    combined = "|".join(relevant_doc_signatures)
    scope_hash = hashlib.sha256(combined.encode("utf-8")).hexdigest()[:16]

    return {
        "scope_version": f"v_{scope_hash}",
        "document_count": len(relevant_doc_signatures),
        "last_updated": last_updated,
        "branch_id": branch_id,
        "department_id": department_id,
    }



def _build_where_filter(
    branch_id: Optional[int], department_id: Optional[int]
) -> Optional[Dict[str, Any]]:
    """Build metadata filter for branch / department context."""
    if branch_id is None and department_id is None:
        return None
    conditions = [{"is_company_wide": True}]
    if branch_id is not None:
        conditions.append({"branch_id": branch_id})
    if department_id is not None:
        conditions.append({"department_id": department_id})
    return {"$or": conditions}


def _parse_search_results(results: Dict[str, Any]) -> List[SearchResultItem]:
    """Extract SearchResultItem instances from ChromaDB query response."""
    if not results or "documents" not in results or not results["documents"]:
        return []

    docs_list = results["documents"][0]
    ids_list = results["ids"][0]
    metas_list = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    search_items: List[SearchResultItem] = []
    for i, doc_text in enumerate(docs_list):
        meta = metas_list[i] if i < len(metas_list) else {}
        dist = distances[i] if i < len(distances) else 0.0
        score = round(max(0.0, 1.0 - dist), 4)

        b_id = meta.get("branch_id")
        d_id = meta.get("department_id")

        search_items.append(
            SearchResultItem(
                chunk_id=ids_list[i],
                content=doc_text,
                source_document=meta.get("source_document", "Unknown"),
                chunk_index=meta.get("chunk_index", 0),
                score=score,
                branch_id=b_id if b_id != -1 else None,
                department_id=d_id if d_id != -1 else None,
                is_company_wide=meta.get("is_company_wide", True),
            )
        )
    return search_items


def search_vector_store(
    query: str,
    top_k: int = 4,
    branch_id: Optional[int] = None,
    department_id: Optional[int] = None,
) -> List[SearchResultItem]:
    """
    Search vector collection for chunks matching the user's branch/department context.
    Matches documents that are:
    1. Company-wide (is_company_wide=True), OR
    2. Tagged to caller's branch_id, OR
    3. Tagged to caller's department_id
    """
    collection = get_vector_collection()
    query_embedding = generate_embedding(query, is_query=True)
    where_filter = _build_where_filter(branch_id, department_id)

    query_params: Dict[str, Any] = {
        "query_embeddings": [query_embedding],
        "n_results": min(top_k, max(1, collection.count())),
    }
    if where_filter and collection.count() > 0:
        query_params["where"] = where_filter

    try:
        results = collection.query(**query_params)
    except Exception as exc:
        logger.warning(f"Filter query error: {exc}. Retrying without metadata filter...")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, max(1, collection.count())),
        )

    return _parse_search_results(results)
