def test_create_branch_and_department_success(client, admin_token):
    """Test creating branches and departments successfully."""
    # 1. Create branch
    branch_resp = client.post(
        "/api/v1/branches",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={
            "company_id": 1,
            "name": "Tokyo Office",
            "location": "Tokyo, Japan",
            "address": "Shibuya City",
            "working_hours": "09:00 - 18:00 JST",
        },
    )
    assert branch_resp.status_code == 201
    branch_data = branch_resp.json()
    assert branch_data["name"] == "Tokyo Office"
    branch_id = branch_data["id"]

    # 2. Create department under Tokyo Office
    dept_resp = client.post(
        "/api/v1/departments",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={
            "branch_id": branch_id,
            "name": "Design & UX",
            "description": "Product design and user experience",
            "duty_timings": "09:00 - 17:30",
        },
    )
    assert dept_resp.status_code == 201
    dept_data = dept_resp.json()
    assert dept_data["name"] == "Design & UX"
    assert dept_data["branch_id"] == branch_id


def test_create_department_with_nonexistent_branch_returns_400_validation_error(client, admin_token):
    """Test 5: Creating a Department with a non-existent branch_id returns a validation error (400), not a 500."""
    response = client.post(
        "/api/v1/departments",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={
            "branch_id": 99999,  # Non-existent branch
            "name": "Ghost Department",
            "description": "Should fail gracefully",
        },
    )
    assert response.status_code == 400
    assert "does not exist" in response.json()["detail"]
