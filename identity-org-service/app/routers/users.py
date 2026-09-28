from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import require_permission
from app.models import Branch, Department, Role, User
from app.schemas import (
    MessageResponse,
    UserCreatedResponse,
    UserCreate,
    UserResponse,
    UserUpdate,
)
from app.security import generate_temporary_password, hash_password

router = APIRouter(prefix="/api/v1/users", tags=["Users"])


def serialize_user(user: User) -> UserResponse:
    """Helper to serialize user model with role_name."""
    return UserResponse(
        id=user.id,
        first_name=user.first_name,
        last_name=user.last_name,
        email=user.email,
        role_id=user.role_id,
        role_name=user.role.name if user.role else None,
        branch_id=user.branch_id,
        department_id=user.department_id,
        must_reset_password=user.must_reset_password,
        is_active=user.is_active,
        created_at=user.created_at,
        updated_at=user.updated_at,
    )


@router.post("", response_model=UserCreatedResponse, status_code=status.HTTP_201_CREATED)
def create_user(
    payload: UserCreate,
    db: Session = Depends(get_db),
    _: User = Depends(require_permission("manage_users")),
):
    """
    Create a new user.
    - Generates a cryptographically secure temporary password.
    - Sets must_reset_password=True.
    - Validates email uniqueness, role existence, and branch/department references.
    """
    email = payload.email.lower()
    existing_user = db.query(User).filter(User.email == email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"User with email '{email}' already exists",
        )

    # Validate role
    role = db.query(Role).filter(Role.id == payload.role_id).first()
    if not role:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Role with ID {payload.role_id} does not exist",
        )

    # Validate branch if provided
    if payload.branch_id is not None:
        branch = db.query(Branch).filter(Branch.id == payload.branch_id).first()
        if not branch:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Branch with ID {payload.branch_id} does not exist",
            )

    # Validate department if provided
    if payload.department_id is not None:
        department = db.query(Department).filter(Department.id == payload.department_id).first()
        if not department:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Department with ID {payload.department_id} does not exist",
            )

    temp_password = generate_temporary_password(14)
    hashed_pw = hash_password(temp_password)

    new_user = User(
        first_name=payload.first_name,
        last_name=payload.last_name,
        email=email,
        hashed_password=hashed_pw,
        role_id=payload.role_id,
        branch_id=payload.branch_id,
        department_id=payload.department_id,
        must_reset_password=True,
        is_active=True,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return UserCreatedResponse(
        id=new_user.id,
        first_name=new_user.first_name,
        last_name=new_user.last_name,
        email=new_user.email,
        role_id=new_user.role_id,
        role_name=role.name,
        branch_id=new_user.branch_id,
        department_id=new_user.department_id,
        must_reset_password=new_user.must_reset_password,
        is_active=new_user.is_active,
        created_at=new_user.created_at,
        updated_at=new_user.updated_at,
        temporary_password=temp_password,
    )


@router.get("", response_model=List[UserResponse])
def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
    _: User = Depends(require_permission("view_users")),
):
    """List users with pagination."""
    users = db.query(User).offset(skip).limit(limit).all()
    return [serialize_user(u) for u in users]


@router.get("/{user_id}", response_model=UserResponse)
def get_user(
    user_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_permission("view_users")),
):
    """Get single user by ID."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found",
        )
    return serialize_user(user)


@router.put("/{user_id}", response_model=UserResponse)
def update_user(
    user_id: int,
    payload: UserUpdate,
    db: Session = Depends(get_db),
    _: User = Depends(require_permission("manage_users")),
):
    """Update user information (role, branch, department, active status)."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found",
        )

    if payload.role_id is not None:
        role = db.query(Role).filter(Role.id == payload.role_id).first()
        if not role:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Role with ID {payload.role_id} does not exist",
            )
        user.role_id = payload.role_id

    if payload.branch_id is not None:
        branch = db.query(Branch).filter(Branch.id == payload.branch_id).first()
        if not branch:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Branch with ID {payload.branch_id} does not exist",
            )
        user.branch_id = payload.branch_id

    if payload.department_id is not None:
        department = db.query(Department).filter(Department.id == payload.department_id).first()
        if not department:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Department with ID {payload.department_id} does not exist",
            )
        user.department_id = payload.department_id

    if payload.first_name is not None:
        user.first_name = payload.first_name
    if payload.last_name is not None:
        user.last_name = payload.last_name
    if payload.is_active is not None:
        user.is_active = payload.is_active

    db.commit()
    db.refresh(user)
    return serialize_user(user)


@router.delete("/{user_id}", response_model=MessageResponse)
def deactivate_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_admin: User = Depends(require_permission("manage_users")),
):
    """Deactivate a user account."""
    if current_admin.id == user_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot deactivate your own user account",
        )

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with ID {user_id} not found",
        )

    user.is_active = False
    db.commit()
    return MessageResponse(message=f"User '{user.email}' successfully deactivated")
