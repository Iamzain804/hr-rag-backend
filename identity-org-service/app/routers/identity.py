from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.dependencies import get_current_user
from app.models import User
from app.schemas import BranchContext, DepartmentContext, UserContextResponse

router = APIRouter(prefix="/api/v1/identity", tags=["Identity Context"])


@router.get("/me/context", response_model=UserContextResponse)
def get_current_user_context(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Returns the single source of truth for the authenticated caller:
    { user_id, email, first_name, last_name, role, branch, department, permissions[], must_reset_password }
    Usable by frontend and other microservices.
    """
    branch_ctx = None
    if current_user.branch:
        branch_ctx = BranchContext(
            id=current_user.branch.id,
            name=current_user.branch.name,
            location=current_user.branch.location,
            address=current_user.branch.address,
            contact_info=current_user.branch.contact_info,
            working_hours=current_user.branch.working_hours,
            policies=current_user.branch.policies,
        )

    dept_ctx = None
    if current_user.department:
        dept_ctx = DepartmentContext(
            id=current_user.department.id,
            name=current_user.department.name,
            description=current_user.department.description,
            responsibilities=current_user.department.responsibilities,
            duty_timings=current_user.department.duty_timings,
            policies=current_user.department.policies,
            escalation_procedures=current_user.department.escalation_procedures,
        )

    permissions_list = []
    if current_user.role:
        permissions_list = [p.name for p in current_user.role.permissions]

    return UserContextResponse(
        user_id=current_user.id,
        email=current_user.email,
        first_name=current_user.first_name,
        last_name=current_user.last_name,
        role=current_user.role.name if current_user.role else "None",
        must_reset_password=current_user.must_reset_password,
        branch=branch_ctx,
        department=dept_ctx,
        permissions=permissions_list,
    )
