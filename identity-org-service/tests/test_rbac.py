def test_unauthorized_user_without_manage_roles_permission_gets_403(client, employee_token):
    """Test 4: A user WITHOUT the 'manage_roles' permission gets 403 when calling role-CRUD routes."""
    # Attempt to list roles
    list_resp = client.get(
        "/api/v1/roles",
        headers={"Authorization": f"Bearer {employee_token}"},
    )
    assert list_resp.status_code == 403
    assert "Permission denied: requires 'manage_roles' permission" in list_resp.json()["detail"]

    # Attempt to create a role
    create_resp = client.post(
        "/api/v1/roles",
        headers={"Authorization": f"Bearer {employee_token}"},
        json={"name": "Attacker Role", "description": "Hacked"},
    )
    assert create_resp.status_code == 403
    assert "Permission denied: requires 'manage_roles' permission" in create_resp.json()["detail"]


def test_admin_can_manage_roles_and_permissions(client, admin_token):
    """Admin with manage_roles permission can perform full role CRUD."""
    # 1. Create role
    create_resp = client.post(
        "/api/v1/roles",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"name": "Auditor", "description": "Compliance auditor", "permission_ids": [1]},
    )
    assert create_resp.status_code == 201
    role_id = create_resp.json()["id"]

    # 2. Get role
    get_resp = client.get(
        f"/api/v1/roles/{role_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert get_resp.status_code == 200
    assert get_resp.json()["name"] == "Auditor"

    # 3. Update role
    update_resp = client.put(
        f"/api/v1/roles/{role_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"description": "Updated auditor description"},
    )
    assert update_resp.status_code == 200
    assert update_resp.json()["description"] == "Updated auditor description"

    # 4. Assign permissions
    assign_resp = client.post(
        f"/api/v1/roles/{role_id}/permissions",
        headers={"Authorization": f"Bearer {admin_token}"},
        json={"permission_ids": [1, 2]},
    )
    assert assign_resp.status_code == 200
    assert len(assign_resp.json()["permissions"]) == 2

    # 5. Delete role
    del_resp = client.delete(
        f"/api/v1/roles/{role_id}",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert del_resp.status_code == 200
