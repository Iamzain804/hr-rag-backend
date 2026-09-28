from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, EmailStr, Field, ConfigDict


# ---------------------------------------------------------------------------
# Auth Schemas
# ---------------------------------------------------------------------------

class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    must_reset_password: bool = False


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class AccessTokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str = Field(..., min_length=8, description="New password must be at least 8 characters long")


class MessageResponse(BaseModel):
    message: str


# ---------------------------------------------------------------------------
# RBAC Schemas (Permissions & Roles)
# ---------------------------------------------------------------------------

class PermissionBase(BaseModel):
    name: str = Field(..., max_length=100)
    description: Optional[str] = Field(None, max_length=255)


class PermissionCreate(PermissionBase):
    pass


class PermissionResponse(PermissionBase):
    id: int

    model_config = ConfigDict(from_attributes=True)


class RoleBase(BaseModel):
    name: str = Field(..., max_length=100)
    description: Optional[str] = Field(None, max_length=255)
    is_active: bool = True


class RoleCreate(RoleBase):
    permission_ids: Optional[List[int]] = Field(default_factory=list)


class RoleUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=100)
    description: Optional[str] = Field(None, max_length=255)
    is_active: Optional[bool] = None


class RoleAssignPermissions(BaseModel):
    permission_ids: List[int]


class RoleResponse(RoleBase):
    id: int
    created_at: datetime
    updated_at: datetime
    permissions: List[PermissionResponse] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# Organization Schemas (Company, Branch, Department)
# ---------------------------------------------------------------------------

class CompanyBase(BaseModel):
    name: str = Field(..., max_length=255)
    code: Optional[str] = Field(None, max_length=50)


class CompanyCreate(CompanyBase):
    pass


class CompanyResponse(CompanyBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class BranchBase(BaseModel):
    name: str = Field(..., max_length=255)
    company_id: int
    location: Optional[str] = Field(None, max_length=255)
    address: Optional[str] = None
    contact_info: Optional[str] = None
    working_hours: Optional[str] = Field(None, max_length=255)
    policies: Optional[str] = None
    is_active: bool = True


class BranchCreate(BranchBase):
    pass


class BranchUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=255)
    company_id: Optional[int] = None
    location: Optional[str] = None
    address: Optional[str] = None
    contact_info: Optional[str] = None
    working_hours: Optional[str] = None
    policies: Optional[str] = None
    is_active: Optional[bool] = None


class BranchResponse(BranchBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class DepartmentBase(BaseModel):
    name: str = Field(..., max_length=255)
    branch_id: int
    description: Optional[str] = None
    responsibilities: Optional[str] = None
    duty_timings: Optional[str] = Field(None, max_length=255)
    policies: Optional[str] = None
    escalation_procedures: Optional[str] = None
    is_active: bool = True


class DepartmentCreate(DepartmentBase):
    pass


class DepartmentUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=255)
    branch_id: Optional[int] = None
    description: Optional[str] = None
    responsibilities: Optional[str] = None
    duty_timings: Optional[str] = None
    policies: Optional[str] = None
    escalation_procedures: Optional[str] = None
    is_active: Optional[bool] = None


class DepartmentResponse(DepartmentBase):
    id: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ---------------------------------------------------------------------------
# User Schemas
# ---------------------------------------------------------------------------

class UserBase(BaseModel):
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    email: EmailStr
    role_id: int
    branch_id: Optional[int] = None
    department_id: Optional[int] = None


class UserCreate(UserBase):
    pass


class UserUpdate(BaseModel):
    first_name: Optional[str] = Field(None, min_length=1, max_length=100)
    last_name: Optional[str] = Field(None, max_length=100)
    role_id: Optional[int] = None
    branch_id: Optional[int] = None
    department_id: Optional[int] = None
    is_active: Optional[bool] = None


class UserResponse(BaseModel):
    id: int
    first_name: str
    last_name: Optional[str] = None
    email: EmailStr
    role_id: int
    role_name: Optional[str] = None
    branch_id: Optional[int] = None
    department_id: Optional[int] = None
    must_reset_password: bool
    is_active: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class UserCreatedResponse(UserResponse):
    temporary_password: str


# ---------------------------------------------------------------------------
# Identity Context Schema
# ---------------------------------------------------------------------------

class BranchContext(BaseModel):
    id: int
    name: str
    location: Optional[str] = None
    address: Optional[str] = None
    contact_info: Optional[str] = None
    working_hours: Optional[str] = None
    policies: Optional[str] = None


class DepartmentContext(BaseModel):
    id: int
    name: str
    description: Optional[str] = None
    responsibilities: Optional[str] = None
    duty_timings: Optional[str] = None
    policies: Optional[str] = None
    escalation_procedures: Optional[str] = None


class UserContextResponse(BaseModel):
    user_id: int
    email: str
    first_name: str
    last_name: Optional[str] = None
    role: str
    must_reset_password: bool
    branch: Optional[BranchContext] = None
    department: Optional[DepartmentContext] = None
    permissions: List[str] = Field(default_factory=list)
