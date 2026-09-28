import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app
from app.models import Branch, Company, Department, Role, User
from app.seed import seed_database
from app.security import create_access_token, hash_password

# Use an in-memory SQLite database for test isolation
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="function")
def db_session():
    """Create a fresh database schema and session for each test."""
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    seed_database(session)
    yield session
    session.close()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="function")
def client(db_session):
    """FastAPI TestClient with overridden get_db dependency."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def admin_token(db_session):
    """Generate a valid JWT token for the seeded Super Admin user."""
    admin = db_session.query(User).filter(User.email == "admin@example.com").first()
    token = create_access_token({
        "sub": str(admin.id),
        "email": admin.email,
        "role": admin.role.name,
    })
    return token


@pytest.fixture
def employee_user(db_session):
    """Create and return a standard Employee user with must_reset_password=False."""
    company = db_session.query(Company).first()
    branch = Branch(
        company_id=company.id,
        name="London Office",
        location="London, UK",
        address="10 Downing Street",
    )
    db_session.add(branch)
    db_session.flush()

    department = Department(
        branch_id=branch.id,
        name="Engineering",
        description="Software Engineering Department",
    )
    db_session.add(department)
    db_session.flush()

    emp_role = db_session.query(Role).filter(Role.name == "Employee").first()

    user = User(
        first_name="Alice",
        last_name="Smith",
        email="alice@example.com",
        hashed_password=hash_password("Alice@123456"),
        role_id=emp_role.id,
        branch_id=branch.id,
        department_id=department.id,
        must_reset_password=False,
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def employee_token(employee_user):
    """Generate a valid JWT token for standard employee user."""
    return create_access_token({
        "sub": str(employee_user.id),
        "email": employee_user.email,
        "role": employee_user.role.name,
    })
