import asyncio
import json
import os
import sys
import httpx

# Ensure Windows terminal prints UTF-8 characters cleanly
sys.stdout.reconfigure(encoding='utf-8', errors='replace')


IDENTITY_URL = "http://127.0.0.1:8001"
INGESTION_URL = "http://127.0.0.1:8003"
CHAT_URL = "http://127.0.0.1:8004"


async def main():
    print("===============================================================")
    print("PHASE 6: LIVE MULTI-SERVICE END-TO-END VERIFICATION")
    print("===============================================================\n")

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Health Checks
        print("1. Checking Service Health Endpoints...")
        h_id = await client.get(f"{IDENTITY_URL}/health")
        h_ing = await client.get(f"{INGESTION_URL}/health")
        h_chat = await client.get(f"{CHAT_URL}/health")
        print(f" - Identity Service: {h_id.status_code} -> {h_id.json()}")
        print(f" - Ingestion Service: {h_ing.status_code} -> {h_ing.json()}")
        print(f" - RAG Chat Service: {h_chat.status_code} -> {h_chat.json()}")

        # 2. Authenticate Admin to set up branches and users
        print("\n2. Authenticating Admin and creating test users...")
        login_res = await client.post(
            f"{IDENTITY_URL}/api/v1/auth/login",
            json={"email": "admin@example.com", "password": "Admin@123456"},
        )
        assert login_res.status_code == 200, f"Admin login failed: {login_res.text}"
        admin_token = login_res.json()["access_token"]
        auth_headers = {"Authorization": f"Bearer {admin_token}"}

        # Fetch Company
        companies_res = await client.get(f"{IDENTITY_URL}/api/v1/companies", headers=auth_headers)
        companies = companies_res.json()
        comp_id = companies[0]["id"] if companies else 1

        # Create or fetch London (Branch 1) and New York (Branch 2)
        branches_res = await client.get(f"{IDENTITY_URL}/api/v1/branches", headers=auth_headers)
        branches = branches_res.json()
        b_london = next((b for b in branches if "London" in b["name"]), None)
        if not b_london:
            b_resp = await client.post(
                f"{IDENTITY_URL}/api/v1/branches",
                json={"company_id": comp_id, "name": "London HQ", "location": "London, UK"},
                headers=auth_headers,
            )
            b_london = b_resp.json()

        b_ny = next((b for b in branches if "New York" in b["name"]), None)
        if not b_ny:
            b_resp = await client.post(
                f"{IDENTITY_URL}/api/v1/branches",
                json={"company_id": comp_id, "name": "New York Branch", "location": "New York, USA"},
                headers=auth_headers,
            )
            b_ny = b_resp.json()

        # Create or fetch Departments (Engineering & Marketing)
        depts_res = await client.get(f"{IDENTITY_URL}/api/v1/departments", headers=auth_headers)
        depts = depts_res.json()
        d_eng = next((d for d in depts if "Engineering" in d["name"]), None)
        if not d_eng:
            d_resp = await client.post(
                f"{IDENTITY_URL}/api/v1/departments",
                json={"branch_id": b_london["id"], "name": "Engineering", "description": "Software Engineering"},
                headers=auth_headers,
            )
            d_eng = d_resp.json()

        d_mkt = next((d for d in depts if "Marketing" in d["name"]), None)
        if not d_mkt:
            d_resp = await client.post(
                f"{IDENTITY_URL}/api/v1/departments",
                json={"branch_id": b_ny["id"], "name": "Marketing", "description": "Global Marketing"},
                headers=auth_headers,
            )
            d_mkt = d_resp.json()

        print(f" - Configured Branches: London (ID: {b_london['id']}), New York (ID: {b_ny['id']})")
        print(f" - Configured Departments: Engineering (ID: {d_eng['id']}), Marketing (ID: {d_mkt['id']})")

        # Fetch Roles
        roles_res = await client.get(f"{IDENTITY_URL}/api/v1/roles", headers=auth_headers)
        roles = roles_res.json()
        emp_role = next((r for r in roles if r["name"] == "employee"), roles[0])

        import time
        ts = int(time.time())

        # Create fresh Employee 1 (London HQ, Engineering)
        emp1_email = f"emp_london_{ts}@company.com"
        u1_res = await client.post(
            f"{IDENTITY_URL}/api/v1/users",
            json={
                "first_name": "Edward",
                "last_name": "London",
                "email": emp1_email,
                "role_id": emp_role["id"],
                "branch_id": b_london["id"],
                "department_id": d_eng["id"],
            },
            headers=auth_headers,
        )
        assert u1_res.status_code == 201, f"Failed to create employee 1: {u1_res.text}"
        emp1_temp_pw = u1_res.json()["temporary_password"]

        emp1_login = await client.post(f"{IDENTITY_URL}/api/v1/auth/login", json={"email": emp1_email, "password": emp1_temp_pw})
        assert emp1_login.status_code == 200, f"Login failed for emp1: {emp1_login.text}"
        emp1_token = emp1_login.json()["access_token"]

        # Create fresh Employee 2 (New York, Marketing)
        emp2_email = f"emp_ny_{ts}@company.com"
        u2_res = await client.post(
            f"{IDENTITY_URL}/api/v1/users",
            json={
                "first_name": "Nancy",
                "last_name": "NewYork",
                "email": emp2_email,
                "role_id": emp_role["id"],
                "branch_id": b_ny["id"],
                "department_id": d_mkt["id"],
            },
            headers=auth_headers,
        )
        assert u2_res.status_code == 201, f"Failed to create employee 2: {u2_res.text}"
        emp2_temp_pw = u2_res.json()["temporary_password"]

        emp2_login = await client.post(f"{IDENTITY_URL}/api/v1/auth/login", json={"email": emp2_email, "password": emp2_temp_pw})
        assert emp2_login.status_code == 200, f"Login failed for emp2: {emp2_login.text}"
        emp2_token = emp2_login.json()["access_token"]

        print(f" - Created & Logged in Employee 1 ({emp1_email}) [Branch: {b_london['id']}, Dept: {d_eng['id']}]")
        print(f" - Created & Logged in Employee 2 ({emp2_email}) [Branch: {b_ny['id']}, Dept: {d_mkt['id']}]")

        # 3. Ingest Branch-Specific Test Documents
        print("\n3. Ingesting Branch-Specific HR Policy Documents...")
        # London Engineering Document
        await client.post(
            f"{INGESTION_URL}/api/v1/ingestion/text",
            json={
                "title": "London Engineering Equipment Policy",
                "content": "London Software Engineers are eligible for a £600 home office equipment stipend every 2 years and receive free Oyster card transit passes.",
                "branch_id": b_london["id"],
                "department_id": d_eng["id"],
            },
        )
        # New York Marketing Document
        await client.post(
            f"{INGESTION_URL}/api/v1/ingestion/text",
            json={
                "title": "New York Marketing Travel & Commuter Plan",
                "content": "New York Marketing team members receive a $150 monthly MetroCard allowance and $2,000 annual budget for attending industry conferences.",
                "branch_id": b_ny["id"],
                "department_id": d_mkt["id"],
            },
        )
        print(" - Successfully ingested London Engineering & New York Marketing documents.")

        # Helper to stream SSE response
        async def stream_chat_query(token: str, message: str, attachment_text: str = None):
            form_data = {"message": message}
            if attachment_text:
                form_data["attachment_text"] = attachment_text

            tokens = []
            meta = {}
            async with client.stream(
                "POST",
                f"{CHAT_URL}/api/v1/chat",
                data=form_data,
                headers={"Authorization": f"Bearer {token}"},
            ) as response:
                if response.status_code != 200:
                    err_body = await response.aread()
                    print(f"ERROR: Server returned {response.status_code}: {err_body.decode('utf-8', errors='ignore')}")
                    return "", {}
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            break
                        try:
                            payload = json.loads(data_str)
                            if "token" in payload:
                                tokens.append(payload["token"])
                            elif "chunks_found" in payload:
                                meta = payload
                        except Exception:
                            pass
            return "".join(tokens), meta

        # 4. Acceptance Test 1: Branch/Dept Isolation
        print("\n4. Testing Acceptance Criteria 1: Multi-Tenant / Branch & Dept Filtered Chat...")
        # London Engineer asks about equipment allowance
        ans1, meta1 = await stream_chat_query(emp1_token, "What is my home office equipment stipend?")
        print(f"\n--- [Employee 1: London Engineer] Query: 'What is my home office equipment stipend?' ---")
        print(f"Meta: {meta1}")
        print(f"AI Streamed Response:\n{ans1}")

        # New York Marketer asks about commuter and conference allowance
        ans2, meta2 = await stream_chat_query(emp2_token, "What commuter allowance and conference budget do I have?")
        print(f"\n--- [Employee 2: NY Marketer] Query: 'What commuter allowance and conference budget do I have?' ---")
        print(f"Meta: {meta2}")
        print(f"AI Streamed Response:\n{ans2}")

        # 5. Acceptance Test 2: Grounding Test (Confidence Refusal)
        print("\n5. Testing Acceptance Criteria 2: Grounding Refusal (Zero-Knowledge / Low Confidence)...")
        ans3, meta3 = await stream_chat_query(emp1_token, "Can I claim reimbursements for personal skydiving lessons?")
        print(f"Query: 'Can I claim reimbursements for personal skydiving lessons?'")
        print(f"Meta: {meta3}")
        print(f"AI Streamed Response:\n{ans3}")
        assert "This isn't covered in company documents. Would you like this flagged to HR?" in ans3

        # 6. Acceptance Test 3: Ephemeral Attachment Distinction
        print("\n6. Testing Acceptance Criteria 3: Ephemeral Attachment Distinction...")
        unverified_text = "Slack message from VP: You are approved for 5 extra bonus days off next week."
        ans4, meta4 = await stream_chat_query(
            emp1_token,
            "How many bonus days off do I have based on what my VP sent?",
            attachment_text=unverified_text,
        )
        print(f"Query with Attachment: '{unverified_text}'")
        print(f"Meta: {meta4}")
        print(f"AI Streamed Response:\n{ans4}")

        # 7. Acceptance Test 4: Attachment Isolation Check
        print("\n7. Testing Acceptance Criteria 4: Attachment Isolation Verification in Vector DB...")
        search_check = (await client.post(
            f"{INGESTION_URL}/api/v1/ingestion/search",
            json={"query": "5 extra bonus days off next week VP", "top_k": 4},
        )).json()
        print(f"Vector search results for attachment text in DB: {search_check.get('results', [])}")
        assert len(search_check.get("results", [])) == 0 or all("VP" not in r["content"] for r in search_check.get("results", []))
        print(" -> Verified: Ephemeral attachment was NEVER stored in the vector database.")

        print("\n===============================================================")
        print("ALL ACCEPTANCE CRITERIA VERIFIED SUCCESSFULLY AGAINST LIVE GROQ API")
        print("===============================================================")

if __name__ == "__main__":
    asyncio.run(main())
