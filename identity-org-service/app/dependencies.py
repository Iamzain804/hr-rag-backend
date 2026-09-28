from typing import Callable, List, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User
from app.security import decode_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Validate access token and return current User database model."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_token(token)
        token_type = payload.get("type")
        if token_type != "access":
            raise credentials_exception
        user_id_str = payload.get("sub")
        if user_id_str is None:
            raise credentials_exception
        user_id = int(user_id_str)
    except Exception:
        raise credentials_exception

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user account",
        )
    return user


def require_permission(
    permission_name: str,
    allow_unreset_password: bool = False,
) -> Callable:
    """
    Reusable dependency factory to enforce server-side RBAC permissions.
    - Ensures user has changed their temporary password before accessing protected endpoints.
    - Ensures user has the specific permission (or is Super Admin).
    """
    def permission_checker(
        current_user: User = Depends(get_current_user),
    ) -> User:
        # Enforce password reset requirement on protected business endpoints
        if current_user.must_reset_password and not allow_unreset_password:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Password reset required before accessing services. Please call /api/v1/auth/change-password.",
            )

        # Check role and permissions
        if not current_user.role:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="No role assigned to current user",
            )

        # Super Admin role bypasses granular permission checks
        if current_user.role.name == "Super Admin":
            return current_user

        user_permissions = [p.name for p in current_user.role.permissions]
        if permission_name not in user_permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission denied: requires '{permission_name}' permission",
            )
        return current_user

    return permission_checker
