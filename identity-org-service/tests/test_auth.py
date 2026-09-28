def test_login_success_returns_jwt(client):
    """Test 1a: Successful login returns a valid JWT access token + refresh token."""
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@example.com", "password": "Admin@123456"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"
    assert data["must_reset_password"] is False


def test_login_wrong_password_returns_401(client):
    """Test 1b: Wrong password returns 401 Unauthorized."""
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@example.com", "password": "WrongPassword999"},
    )
    assert response.status_code == 401
    assert "Invalid email or password" in response.json()["detail"]


def test_login_nonexistent_email_returns_401(client):
    """Test 1c: Non-existent email returns 401."""
    response = client.post(
        "/api/v1/auth/login",
        json={"email": "nonexistent@example.com", "password": "SomePassword123"},
    )
    assert response.status_code == 401


def test_refresh_token_success(client):
    """Test refresh token returns new access token."""
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"email": "admin@example.com", "password": "Admin@123456"},
    )
    refresh_tok = login_resp.json()["refresh_token"]

    refresh_resp = client.post(
        "/api/v1/auth/refresh",
        json={"refresh_token": refresh_tok},
    )
    assert refresh_resp.status_code == 200
    assert "access_token" in refresh_resp.json()
    assert refresh_resp.json()["token_type"] == "bearer"


def test_change_password_flow(client, employee_user, employee_token):
    """Test change password updates password and clears must_reset_password flag."""
    # Ensure current user has must_reset_password
    employee_user.must_reset_password = True

    response = client.post(
        "/api/v1/auth/change-password",
        headers={"Authorization": f"Bearer {employee_token}"},
        json={"current_password": "Alice@123456", "new_password": "NewSecretPassword@2026"},
    )
    assert response.status_code == 200
    assert "successfully updated" in response.json()["message"]

    # Verify login with new password
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"email": "alice@example.com", "password": "NewSecretPassword@2026"},
    )
    assert login_resp.status_code == 200
    assert login_resp.json()["must_reset_password"] is False
