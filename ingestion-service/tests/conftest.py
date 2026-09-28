import os
import shutil
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

# Override chroma directory to temporary test directory
TEST_CHROMA_DIR = "./test_chroma_db"
os.environ["CHROMA_PERSIST_DIR"] = TEST_CHROMA_DIR

from app.config import settings
settings.CHROMA_PERSIST_DIR = TEST_CHROMA_DIR

from app.main import app
from app.vector_store import clear_all_data


@pytest.fixture(autouse=True)
def clean_test_db():
    """Ensure clean test vector database before and after each test."""
    clear_all_data()
    yield
    clear_all_data()


@pytest.fixture
def client():
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client
