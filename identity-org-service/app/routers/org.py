from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import require_permission
from app.models import Branch, Company, Department, User
from app.schemas import (
    BranchCreate,
    BranchResponse,
    BranchUpdate,
    CompanyCreate,
    CompanyResponse,
    DepartmentCreate,
    DepartmentResponse,
    DepartmentUpdate,
    MessageResponse,
)

router = APIRouter(prefix="/api/v1", tags=["Organization"])


# ---------------------------------------------------------------------------
# Companies
# ---------------------------------------------------------------------------

@router.get("/companies", response_model=List[CompanyResponse])
def list_companies(
    db: Session = Depends(get_db),
    _: User = Depends(require_permission("view_org")),
):
    """List all companies."""
    return db.query(Company).all()


@router.post("/companies", response_model=CompanyResponse, status_code=status.HTTP_201_CREATED)
def create_company(
    payload: CompanyCreate,
    db: Session = Depends(get_db),
    _: User = Depends(require_permission("manage_org")),
):
    """Create a new company."""
    existing = db.query(Company).filter(Company.name == payload.name).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Company with name '{payload.name}' already exists",
        )
    company = Company(name=payload.name, code=payload.code)
    db.add(company)
    db.commit()
    db.refresh(company)
    return company


@router.get("/companies/{company_id}", response_model=CompanyResponse)
def get_company(
    company_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_permission("view_org")),
):
    """Get company details by ID."""
    company = db.query(Company).filter(Company.id == company_id).first()
    if not company:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Company with ID {company_id} not found",
        )
    return company


# ---------------------------------------------------------------------------
# Branches
# ---------------------------------------------------------------------------

@router.get("/branches", response_model=List[BranchResponse])
def list_branches(
    db: Session = Depends(get_db),
    _: User = Depends(require_permission("view_org")),
):
    """List all organization branches."""
    return db.query(Branch).all()


@router.post("/branches", response_model=BranchResponse, status_code=status.HTTP_201_CREATED)
def create_branch(
    payload: BranchCreate,
    db: Session = Depends(get_db),
    _: User = Depends(require_permission("manage_org")),
):
    """Create a new branch under a valid company."""
    company = db.query(Company).filter(Company.id == payload.company_id).first()
    if not company:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Company with ID {payload.company_id} does not exist",
        )

    branch = Branch(
        company_id=payload.company_id,
        name=payload.name,
        location=payload.location,
        address=payload.address,
        contact_info=payload.contact_info,
        working_hours=payload.working_hours,
        policies=payload.policies,
        is_active=payload.is_active,
    )
    db.add(branch)
    db.commit()
    db.refresh(branch)
    return branch


@router.get("/branches/{branch_id}", response_model=BranchResponse)
def get_branch(
    branch_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_permission("view_org")),
):
    """Get branch details by ID."""
    branch = db.query(Branch).filter(Branch.id == branch_id).first()
    if not branch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Branch with ID {branch_id} not found",
        )
    return branch


@router.put("/branches/{branch_id}", response_model=BranchResponse)
def update_branch(
    branch_id: int,
    payload: BranchUpdate,
    db: Session = Depends(get_db),
    _: User = Depends(require_permission("manage_org")),
):
    """Update branch details."""
    branch = db.query(Branch).filter(Branch.id == branch_id).first()
    if not branch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Branch with ID {branch_id} not found",
        )

    if payload.company_id is not None:
        company = db.query(Company).filter(Company.id == payload.company_id).first()
        if not company:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Company with ID {payload.company_id} does not exist",
            )
        branch.company_id = payload.company_id

    if payload.name is not None:
        branch.name = payload.name
    if payload.location is not None:
        branch.location = payload.location
    if payload.address is not None:
        branch.address = payload.address
    if payload.contact_info is not None:
        branch.contact_info = payload.contact_info
    if payload.working_hours is not None:
        branch.working_hours = payload.working_hours
    if payload.policies is not None:
        branch.policies = payload.policies
    if payload.is_active is not None:
        branch.is_active = payload.is_active

    db.commit()
    db.refresh(branch)
    return branch


@router.delete("/branches/{branch_id}", response_model=MessageResponse)
def delete_branch(
    branch_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_permission("manage_org")),
):
    """Delete a branch."""
    branch = db.query(Branch).filter(Branch.id == branch_id).first()
    if not branch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Branch with ID {branch_id} not found",
        )
    db.delete(branch)
    db.commit()
    return MessageResponse(message=f"Branch '{branch.name}' successfully deleted")


# ---------------------------------------------------------------------------
# Departments
# ---------------------------------------------------------------------------

@router.get("/departments", response_model=List[DepartmentResponse])
def list_departments(
    db: Session = Depends(get_db),
    _: User = Depends(require_permission("view_org")),
):
    """List all departments across branches."""
    return db.query(Department).all()


@router.post("/departments", response_model=DepartmentResponse, status_code=status.HTTP_201_CREATED)
def create_department(
    payload: DepartmentCreate,
    db: Session = Depends(get_db),
    _: User = Depends(require_permission("manage_org")),
):
    """
    Create a new department.
    Validates that the referenced branch_id exists in the database.
    """
    branch = db.query(Branch).filter(Branch.id == payload.branch_id).first()
    if not branch:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Branch with ID {payload.branch_id} does not exist",
        )

    department = Department(
        branch_id=payload.branch_id,
        name=payload.name,
        description=payload.description,
        responsibilities=payload.responsibilities,
        duty_timings=payload.duty_timings,
        policies=payload.policies,
        escalation_procedures=payload.escalation_procedures,
        is_active=payload.is_active,
    )
    db.add(department)
    db.commit()
    db.refresh(department)
    return department


@router.get("/departments/{department_id}", response_model=DepartmentResponse)
def get_department(
    department_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_permission("view_org")),
):
    """Get department details by ID."""
    department = db.query(Department).filter(Department.id == department_id).first()
    if not department:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Department with ID {department_id} not found",
        )
    return department


@router.put("/departments/{department_id}", response_model=DepartmentResponse)
def update_department(
    department_id: int,
    payload: DepartmentUpdate,
    db: Session = Depends(get_db),
    _: User = Depends(require_permission("manage_org")),
):
    """Update department details with branch validation."""
    department = db.query(Department).filter(Department.id == department_id).first()
    if not department:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Department with ID {department_id} not found",
        )

    if payload.branch_id is not None:
        branch = db.query(Branch).filter(Branch.id == payload.branch_id).first()
        if not branch:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Branch with ID {payload.branch_id} does not exist",
            )
        department.branch_id = payload.branch_id

    if payload.name is not None:
        department.name = payload.name
    if payload.description is not None:
        department.description = payload.description
    if payload.responsibilities is not None:
        department.responsibilities = payload.responsibilities
    if payload.duty_timings is not None:
        department.duty_timings = payload.duty_timings
    if payload.policies is not None:
        department.policies = payload.policies
    if payload.escalation_procedures is not None:
        department.escalation_procedures = payload.escalation_procedures
    if payload.is_active is not None:
        department.is_active = payload.is_active

    db.commit()
    db.refresh(department)
    return department


@router.delete("/departments/{department_id}", response_model=MessageResponse)
def delete_department(
    department_id: int,
    db: Session = Depends(get_db),
    _: User = Depends(require_permission("manage_org")),
):
    """Delete a department."""
    department = db.query(Department).filter(Department.id == department_id).first()
    if not department:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Department with ID {department_id} not found",
        )
    db.delete(department)
    db.commit()
    return MessageResponse(message=f"Department '{department.name}' successfully deleted")
