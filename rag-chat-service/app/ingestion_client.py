import logging
from typing import List, Optional
import httpx

from app.config import settings
from app.schemas import RetrievedChunk

logger = logging.getLogger("rag_chat_service.ingestion")


async def search_knowledge_base(
    query: str,
    branch_id: Optional[int] = None,
    department_id: Optional[int] = None,
    top_k: int = 4,
) -> List[RetrievedChunk]:
    """
    Query the ingestion service vector store, filtering by branch_id and department_id
    along with company-wide chunks.
    """
    url = f"{settings.INGESTION_SERVICE_URL}/api/v1/ingestion/search"
    payload = {
        "query": query,
        "top_k": top_k,
        "branch_id": branch_id,
        "department_id": department_id,
    }

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.post(url, json=payload)

        if response.status_code == 200:
            data = response.json()
            raw_results = data.get("results", [])
            chunks: List[RetrievedChunk] = []
            for item in raw_results:
                chunks.append(
                    RetrievedChunk(
                        chunk_id=item.get("chunk_id", ""),
                        content=item.get("content", ""),
                        source_document=item.get("source_document", "Unknown"),
                        chunk_index=item.get("chunk_index", 0),
                        score=float(item.get("score", 0.0)),
                        branch_id=item.get("branch_id"),
                        department_id=item.get("department_id"),
                        is_company_wide=item.get("is_company_wide", True),
                    )
                )
            return chunks
        else:
            logger.warning(
                f"Ingestion service search returned status {response.status_code}: {response.text}"
            )
            return []

    except httpx.RequestError as exc:
        logger.error(f"Failed to connect to ingestion service at {url}: {exc}")
        return []


async def get_scope_ingestion_version(
    branch_id: Optional[int] = None,
    department_id: Optional[int] = None,
) -> str:
    """
    Fetch the current scope ingestion version signature from ingestion service.
    Returns version string (e.g. 'v_abc123' or 'v0_empty').
    """
    url = f"{settings.INGESTION_SERVICE_URL}/api/v1/ingestion/version"
    params = {}
    if branch_id is not None:
        params["branch_id"] = branch_id
    if department_id is not None:
        params["department_id"] = department_id

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            return data.get("scope_version", "v0_default")
        else:
            logger.warning(
                f"Ingestion version check returned status {response.status_code}: {response.text}"
            )
            return "v0_default"
    except Exception as exc:
        logger.warning(f"Failed to fetch ingestion version from {url}: {exc}")
        return "v0_default"

