import os
import sys
from pathlib import Path

os.environ["LITELLM_TELEMETRY"] = "False"

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas import UserContext


@pytest.fixture
def client():
    """FastAPI TestClient for rag-chat-service."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_user_branch_1():
    """Sample user belonging to Branch 1 (London) and Dept 1 (Engineering)."""
    return UserContext(
        user_id=10,
        email="london.engineer@company.com",
        full_name="Alice Engineer",
        role="employee",
        branch_id=1,
        branch_name="London HQ",
        department_id=1,
        department_name="Engineering",
        permissions=["chat:read", "chat:write"],
    )


@pytest.fixture
def sample_user_branch_2():
    """Sample user belonging to Branch 2 (New York) and Dept 2 (Marketing)."""
    return UserContext(
        user_id=20,
        email="ny.marketer@company.com",
        full_name="Bob Marketer",
        role="employee",
        branch_id=2,
        branch_name="New York Office",
        department_id=2,
        department_name="Marketing",
        permissions=["chat:read", "chat:write"],
    )
