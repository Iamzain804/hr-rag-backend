from sqlalchemy.orm import Session
from app.config import settings
from app.models import Company, Permission, Role, User
from app.security import hash_password

INITIAL_PERMISSIONS = [
    # RBAC
    ("manage_roles", "Ability to create, update, and delete roles and manage permissions"),
    ("manage_permissions", "Ability to create and manage system permissions"),
    # Users
    ("manage_users", "Ability to create, update, and deactivate users"),
    ("view_users", "Ability to view user lists and details"),
    # Organization
    ("manage_org", "Ability to create, update, and delete companies, branches, and departments"),
    ("view_org", "Ability to view organization structures, branches, and departments"),
    # Services
    ("chat_rag", "Ability to perform RAG search and chat with documents"),
    ("upload_documents", "Ability to upload and ingest organizational documents"),
    ("manage_notifications", "Ability to send and broadcast system notifications"),
]

INITIAL_ROLES = [
    ("Super Admin", "Full system administrator with unrestricted privileges", [p[0] for p in INITIAL_PERMISSIONS]),
    ("HR Manager", "Human resources manager with user and organization management privileges", ["manage_users", "view_users", "view_org", "chat_rag", "upload_documents", "manage_notifications"]),
    ("Employee", "Standard employee role for searching policies and RAG chat", ["view_org", "chat_rag"]),
]


def seed_database(db: Session):
    """Seed initial company, permissions, roles, and default Super Admin user."""
    # 1. Seed Permissions
    perm_map = {}
    for name, desc in INITIAL_PERMISSIONS:
        perm = db.query(Permission).filter(Permission.name == name).first()
        if not perm:
            perm = Permission(name=name, description=desc)
            db.add(perm)
            db.flush()
        perm_map[name] = perm

    # 2. Seed Roles & Assign Permissions
    role_map = {}
    for role_name, role_desc, role_perms in INITIAL_ROLES:
        role = db.query(Role).filter(Role.name == role_name).first()
        if not role:
            role = Role(name=role_name, description=role_desc)
            db.add(role)
            db.flush()
        # Ensure permissions are attached
        role.permissions = [perm_map[p_name] for p_name in role_perms if p_name in perm_map]
        db.flush()
        role_map[role_name] = role

    # 3. Seed Default Company
    company = db.query(Company).filter(Company.name == settings.INITIAL_COMPANY_NAME).first()
    if not company:
        company = Company(name=settings.INITIAL_COMPANY_NAME, code="DEFAULT_ORG")
        db.add(company)
        db.flush()

    # 4. Seed Super Admin User
    admin_email = settings.INITIAL_ADMIN_EMAIL.lower()
    admin_user = db.query(User).filter(User.email == admin_email).first()
    if not admin_user:
        admin_user = User(
            first_name=settings.INITIAL_ADMIN_NAME,
            last_name="System",
            email=admin_email,
            hashed_password=hash_password(settings.INITIAL_ADMIN_PASSWORD),
            role_id=role_map["Super Admin"].id,
            must_reset_password=False,
            is_active=True,
        )
        db.add(admin_user)
        db.flush()

    db.commit()
