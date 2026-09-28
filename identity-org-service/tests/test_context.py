def test_me_context_returns_correct_user_data(client, employee_user, employee_token):
    """Test 3: GET /me/context returns correct role/branch/department/permissions for a seeded test user."""
    response = client.get(
        "/api/v1/identity/me/context",
        headers={"Authorization": f"Bearer {employee_token}"},
    )
    assert response.status_code == 200
    data = response.json()

    # Verify user fields
    assert data["user_id"] == employee_user.id
    assert data["email"] == "alice@example.com"
    assert data["first_name"] == "Alice"
    assert data["last_name"] == "Smith"
    assert data["role"] == "Employee"
    assert data["must_reset_password"] is False

    # Verify branch context
    assert data["branch"] is not None
    assert data["branch"]["name"] == "London Office"
    assert data["branch"]["location"] == "London, UK"

    # Verify department context
    assert data["department"] is not None
    assert data["department"]["name"] == "Engineering"

    # Verify permissions array
    assert isinstance(data["permissions"], list)
    assert "chat_rag" in data["permissions"]
    assert "view_org" in data["permissions"]
    assert "manage_roles" not in data["permissions"]


def test_me_context_without_token_returns_401(client):
    """Calling /me/context without token returns 401."""
    response = client.get("/api/v1/identity/me/context")
    assert response.status_code == 401
