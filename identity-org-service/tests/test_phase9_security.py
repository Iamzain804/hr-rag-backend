from datetime import datetime, timedelta, timezone
from jose import jwt
import pytest


def create_expired_token(user_id: int = 1, email: str = "admin@example.com") -> str:
    """Generate an expired token for security verification."""
    expire = datetime.now(timezone.utc) - timedelta(hours=2)
    payload = {
        "sub": str(user_id),
        "email": email,
        "exp": expire,
        "type": "access",
    }
    return jwt.encode(payload, "test_super_secret_jwt_key_identity_service_2026", algorithm="HS256")


def test_non_admin_direct_api_role_creation_forbidden_403(client, employee_token):
    """
    Phase 9 Test A:
    A logged-in employee (non-admin) attempts to call POST /api/v1/roles directly via API
    (bypassing the frontend entirely) — MUST receive 403 Forbidden.
    """
    response = client.post(
        "/api/v1/roles",
        headers={"Authorization": f"Bearer {employee_token}"},
        json={
            "name": "SuperPrivilegedRole",
            "description": "Attempting privilege escalation",
            "permission_ids": [1, 2, 3, 4, 5, 6, 7, 8],
        },
    )
    assert response.status_code == 403
    assert "Permission denied" in response.json()["detail"]


def test_expired_jwt_rejected_across_all_identity_endpoints(client):
    """
    Phase 9 Test C:
    An expired JWT is used against protected routes — confirm consistent 401 handling everywhere.
    """
    expired_token = create_expired_token()
    headers = {"Authorization": f"Bearer {expired_token}"}

    # 1. Identity Context route
    resp_context = client.get("/api/v1/identity/me/context", headers=headers)
    assert resp_context.status_code == 401

    # 2. Roles Management route
    resp_roles = client.get("/api/v1/roles", headers=headers)
    assert resp_roles.status_code == 401

    # 3. Branches Management route
    resp_branches = client.get("/api/v1/branches", headers=headers)
    assert resp_branches.status_code == 401

    # 4. Departments Management route
    resp_departments = client.get("/api/v1/departments", headers=headers)
    assert resp_departments.status_code == 401

    # 5. Users Management route
    resp_users = client.get("/api/v1/users", headers=headers)
    assert resp_users.status_code == 401
