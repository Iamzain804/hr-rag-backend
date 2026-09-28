from unittest.mock import AsyncMock, patch
import pytest
from app.schemas import UserContext


@pytest.fixture
def mock_user_context():
    return UserContext(
        user_id=101,
        email="history_user@example.com",
        full_name="History Tester",
        role="employee",
        branch_id=1,
        branch_name="Headquarters",
        department_id=1,
        department_name="Engineering",
        permissions=["chat_rag"],
    )


def test_conversation_crud_endpoints(client, mock_user_context):
    """Test conversation creation, listing, message retrieval, and deletion."""
    with patch("app.routers.chat.validate_user_context", AsyncMock(return_value=mock_user_context)):
        # 1. Create a conversation
        res = client.post(
            "/api/v1/chat/conversations",
            json={"title": "Test Onboarding Query"},
            headers={"Authorization": "Bearer test-jwt"},
        )
        assert res.status_code == 200
        data = res.json()
        conv_id = data["id"]
        assert data["title"] == "Test Onboarding Query"
        assert data["user_id"] == 101

        # 2. List conversations
        res = client.get(
            "/api/v1/chat/conversations",
            headers={"Authorization": "Bearer test-jwt"},
        )
        assert res.status_code == 200
        conv_list = res.json()["conversations"]
        assert len(conv_list) >= 1
        assert any(c["id"] == conv_id for c in conv_list)

        # 3. Get single conversation
        res = client.get(
            f"/api/v1/chat/conversations/{conv_id}",
            headers={"Authorization": "Bearer test-jwt"},
        )
        assert res.status_code == 200
        detail = res.json()
        assert detail["conversation"]["id"] == conv_id
        assert isinstance(detail["messages"], list)

        # 4. Delete conversation
        res = client.delete(
            f"/api/v1/chat/conversations/{conv_id}",
            headers={"Authorization": "Bearer test-jwt"},
        )
        assert res.status_code == 200
        assert res.json()["message"] == "Conversation deleted successfully"

        # 5. Verify deleted (404)
        res = client.get(
            f"/api/v1/chat/conversations/{conv_id}",
            headers={"Authorization": "Bearer test-jwt"},
        )
        assert res.status_code == 404
