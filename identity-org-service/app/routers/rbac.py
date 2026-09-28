from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import require_permission
from app.models import Permission, Role, User
from app.schemas import (
    MessageResponse,
    PermissionCreate,
    PermissionResponse,
    RoleAssignPermissions,
    RoleCreate,
    RoleResponse,
    RoleUpdate,
)

router = APIRouter(prefix="/api/v1", tags=["RBAC"])


# ---------------------------------------------------------------------------
# Permissions Management
# ---------------------------------------------------------------------------

@router.get("/permissions", response_model=List[PermissionResponse])
def list_permissions(
    db: Session = Depends(get_db),
    _: User = Depends(require_permission("manage_permissions")),
):
    """List all available system permissions."""
    return db.query(Permission).all()


@router.post("/permissions", response_model=PermissionResponse, status_code=status.HTTP_201_CREATED)
def create_permission(
    payload: PermissionCreate,
    db: Session = Depends(get_db),
    _: User = Depends(require_permission("manage_permissions")),
):
    """Create a new system permission."""
    existing = db.query(Permission).filter(Permission.name == payload.name).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Permission with name '{payload.name}' already exists",
        )

    perm = Permission(name=payload.name, description=payload.description)
    db.add(perm)
    db.commit()
    db.refresh(perm)
    return perm


@router.get("/permissions/{permission_id}", response_model=PermissionResponse)
def get_permission(
    permission_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_permission("manage_permissions")),
):
    """Get permission details by ID."""
    perm = db.query(Permission).filter(Permission.id == permission_id).first()
    if not perm:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Permission with ID {permission_id} not found",
        )
    return perm


@router.delete("/permissions/{permission_id}", response_model=MessageResponse)
def delete_permission(
    permission_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_permission("manage_permissions")),
):
    """Delete a permission by ID."""
    perm = db.query(Permission).filter(Permission.id == permission_id).first()
    if not perm:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Permission with ID {permission_id} not found",
        )
    db.delete(perm)
    db.commit()
    return MessageResponse(message=f"Permission '{perm.name}' successfully deleted")


# ---------------------------------------------------------------------------
# Roles Management
# ---------------------------------------------------------------------------

@router.get("/roles", response_model=List[RoleResponse])
def list_roles(
    db: Session = Depends(get_db),
    _: User = Depends(require_permission("manage_roles")),
):
    """List all user roles with attached permissions."""
    return db.query(Role).all()


@router.post("/roles", response_model=RoleResponse, status_code=status.HTTP_201_CREATED)
def create_role(
    payload: RoleCreate,
    db: Session = Depends(get_db),
    _: User = Depends(require_permission("manage_roles")),
):
    """Create a new role with optional initial permissions."""
    existing = db.query(Role).filter(Role.name == payload.name).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Role with name '{payload.name}' already exists",
        )

    role = Role(name=payload.name, description=payload.description, is_active=payload.is_active)
    if payload.permission_ids:
        perms = db.query(Permission).filter(Permission.id.in_(payload.permission_ids)).all()
        if len(perms) != len(payload.permission_ids):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="One or more specified permission IDs do not exist",
            )
        role.permissions = perms

    db.add(role)
    db.commit()
    db.refresh(role)
    return role


@router.get("/roles/{role_id}", response_model=RoleResponse)
def get_role(
    role_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_permission("manage_roles")),
):
    """Get role details by ID."""
    role = db.query(Role).filter(Role.id == role_id).first()
    if not role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Role with ID {role_id} not found",
        )
    return role


@router.put("/roles/{role_id}", response_model=RoleResponse)
def update_role(
    role_id: int,
    payload: RoleUpdate,
    db: Session = Depends(get_db),
    _: User = Depends(require_permission("manage_roles")),
):
    """Update role name, description, or active status."""
    role = db.query(Role).filter(Role.id == role_id).first()
    if not role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Role with ID {role_id} not found",
        )

    if payload.name and payload.name != role.name:
        existing = db.query(Role).filter(Role.name == payload.name).first()
        if existing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Role with name '{payload.name}' already exists",
            )
        role.name = payload.name

    if payload.description is not None:
        role.description = payload.description
    if payload.is_active is not None:
        role.is_active = payload.is_active

    db.commit()
    db.refresh(role)
    return role


@router.delete("/roles/{role_id}", response_model=MessageResponse)
def delete_role(
    role_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_permission("manage_roles")),
):
    """Delete a role by ID."""
    role = db.query(Role).filter(Role.id == role_id).first()
    if not role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Role with ID {role_id} not found",
        )
    if role.name == "Super Admin":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete default Super Admin role",
        )

    if role.users:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete role that is currently assigned to users. Reassign users first.",
        )

    db.delete(role)
    db.commit()
    return MessageResponse(message=f"Role '{role.name}' successfully deleted")


# ---------------------------------------------------------------------------
# Assign / Remove Permissions from Role
# ---------------------------------------------------------------------------

@router.post("/roles/{role_id}/permissions", response_model=RoleResponse)
def assign_permissions_to_role(
    role_id: int,
    payload: RoleAssignPermissions,
    db: Session = Depends(get_db),
    _: User = Depends(require_permission("manage_roles")),
):
    """Assign or replace the list of permissions for a role."""
    role = db.query(Role).filter(Role.id == role_id).first()
    if not role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Role with ID {role_id} not found",
        )

    perms = db.query(Permission).filter(Permission.id.in_(payload.permission_ids)).all()
    if len(perms) != len(set(payload.permission_ids)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="One or more specified permission IDs do not exist",
        )

    role.permissions = perms
    db.commit()
    db.refresh(role)
    return role


@router.delete("/roles/{role_id}/permissions/{permission_id}", response_model=RoleResponse)
def remove_permission_from_role(
    role_id: int,
    permission_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_permission("manage_roles")),
):
    """Remove a single permission from a role."""
    role = db.query(Role).filter(Role.id == role_id).first()
    if not role:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Role with ID {role_id} not found",
        )

    perm = db.query(Permission).filter(Permission.id == permission_id).first()
    if not perm:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Permission with ID {permission_id} not found",
        )

    if perm in role.permissions:
        role.permissions.remove(perm)
        db.commit()
        db.refresh(role)

    return role
