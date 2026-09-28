import logging
from typing import Optional
import httpx
from fastapi import HTTPException, status

from app.config import settings
from app.schemas import UserContext

logger = logging.getLogger("rag_chat_service.auth")


async def validate_user_context(token: str) -> UserContext:
    """
    Validate JWT token by delegating to identity-org-service.
    Never re-implements JWT validation locally to preserve loose coupling and
    maintain identity-org-service as the single source of truth for identity.
    """
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    clean_token = token.replace("Bearer ", "").strip()
    url = f"{settings.IDENTITY_SERVICE_URL}/api/v1/identity/me/context"

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                url,
                headers={"Authorization": f"Bearer {clean_token}"},
            )

        if response.status_code == 200:
            data = response.json()
            first_name = data.get("first_name", "")
            last_name = data.get("last_name", "")
            full_name = f"{first_name} {last_name}".strip() or data.get("email", "")

            branch_data = data.get("branch") or {}
            dept_data = data.get("department") or {}

            return UserContext(
                user_id=data.get("user_id"),
                email=data.get("email"),
                full_name=full_name,
                role=data.get("role", "employee"),
                branch_id=branch_data.get("id"),
                branch_name=branch_data.get("name"),
                department_id=dept_data.get("id"),
                department_name=dept_data.get("name"),
                permissions=data.get("permissions", []),
            )
        elif response.status_code == 401:
            logger.warning(f"Identity service rejected token: {response.text}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired authentication token.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        else:
            logger.error(f"Unexpected response from identity service: {response.status_code} - {response.text}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Identity service returned error: {response.status_code}",
            )

    except httpx.RequestError as exc:
        logger.error(f"Failed to connect to identity service at {url}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Identity service is currently unreachable.",
        )
