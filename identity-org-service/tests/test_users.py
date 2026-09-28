def test_create_user_with_temp_password_and_reset_flag(client, admin_token, db_session):
    """Test 2a: New user created with must_reset_password=True and secure temporary password."""
    response = client.post(
        "/api/v1/users",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={
            "first_name": "John",
            "last_name": "Doe",
            "email": "john.doe@example.com",
            "role_id": 2,  # HR Manager
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "john.doe@example.com"
    assert data["must_reset_password"] is True
    assert "temporary_password" in data
    assert len(data["temporary_password"]) >= 12


def test_user_with_must_reset_password_blocked_on_protected_routes(client, admin_token):
    """Test 2b: User with must_reset_password=True is blocked from protected mutating endpoints with 403."""
    # 1. Admin creates user
    create_resp = client.post(
        "/api/v1/users",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={
            "first_name": "Bob",
            "email": "bob@example.com",
            "role_id": 1,  # Super Admin role
        },
    )
    assert create_resp.status_code == 201
    temp_pw = create_resp.json()["temporary_password"]

    # 2. Bob logs in with temporary password
    login_resp = client.post(
        "/api/v1/auth/login",
        json={"email": "bob@example.com", "password": temp_pw},
    )
    assert login_resp.status_code == 200
    assert login_resp.json()["must_reset_password"] is True
    bob_token = login_resp.json()["access_token"]

    # 3. Bob attempts to perform a protected action before resetting password -> Should be blocked (403)
    blocked_resp = client.post(
        "/api/v1/branches",
        headers={"Authorization": f"Bearer {bob_token}"},
        json={"name": "Bob Branch", "company_id": 1},
    )
    assert blocked_resp.status_code == 403
    assert "Password reset required" in blocked_resp.json()["detail"]

    # 4. Bob resets password
    reset_resp = client.post(
        "/api/v1/auth/change-password",
        headers={"Authorization": f"Bearer {bob_token}"},
        json={"current_password": temp_pw, "new_password": "BobSecurePassword@2026"},
    )
    assert reset_resp.status_code == 200

    # 5. Bob logs in again with new password and now CAN access protected routes
    login2 = client.post(
        "/api/v1/auth/login",
        json={"email": "bob@example.com", "password": "BobSecurePassword@2026"},
    )
    bob_new_token = login2.json()["access_token"]

    allowed_resp = client.post(
        "/api/v1/branches",
        headers={"Authorization": f"Bearer {bob_new_token}"},
        json={"name": "Bob Branch", "company_id": 1},
    )
    assert allowed_resp.status_code == 201


def test_duplicate_email_user_creation_returns_400(client, admin_token):
    """Test 6: Duplicate email on user creation returns clear 400 error, not a crash."""
    payload = {
        "first_name": "Duplicate",
        "email": "admin@example.com",  # Already seeded
        "role_id": 2,
    }
    response = client.post(
        "/api/v1/users",
        headers={"Authorization": f"Bearer {admin_token}"},
        json=payload,
    )
    assert response.status_code == 400
    assert "already exists" in response.json()["detail"]
