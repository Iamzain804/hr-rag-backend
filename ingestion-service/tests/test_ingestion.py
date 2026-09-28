from app.embeddings import generate_embedding, generate_embeddings_batch
from app.vector_store import get_vector_collection


def test_1_ingest_two_sample_documents_and_list_metadata(client):
    """
    Test 1: Ingest 2 sample documents:
    - One tagged to specific branch_id=1 and department_id=1
    - One company-wide (branch_id=null, department_id=null)
    Confirm via GET /api/v1/ingestion/documents that both appear with correct metadata.
    """
    # 1. Ingest Branch/Dept Specific Document
    doc1_payload = {
        "title": "Engineering Duty & Overtime Policy",
        "content": "All London software engineers are eligible for flexible hours between 08:00 and 19:00. Overtime requires prior written approval from the department lead.",
        "branch_id": 1,
        "department_id": 1,
    }
    res1 = client.post("/api/v1/ingestion/text", json=doc1_payload)
    assert res1.status_code == 201
    data1 = res1.json()
    assert data1["branch_id"] == 1
    assert data1["department_id"] == 1
    assert data1["is_company_wide"] is False
    assert data1["chunk_count"] >= 1

    # 2. Ingest Company-Wide Document (branch_id=null, department_id=null)
    doc2_payload = {
        "title": "Global Annual Leave & Code of Conduct",
        "content": "All full-time employees across all branches receive 25 days of paid annual leave per calendar year. Standard bereavement leave is 5 consecutive working days.",
        "branch_id": None,
        "department_id": None,
    }
    res2 = client.post("/api/v1/ingestion/text", json=doc2_payload)
    assert res2.status_code == 201
    data2 = res2.json()
    assert data2["branch_id"] is None
    assert data2["department_id"] is None
    assert data2["is_company_wide"] is True

    # 3. Verify via GET /api/v1/ingestion/documents
    list_res = client.get("/api/v1/ingestion/documents")
    assert list_res.status_code == 200
    docs = list_res.json()
    assert len(docs) == 2

    # Verify metadata accuracy
    titles = [d["title"] for d in docs]
    assert "Engineering Duty & Overtime Policy" in titles
    assert "Global Annual Leave & Code of Conduct" in titles

    company_doc = next(d for d in docs if d["title"] == "Global Annual Leave & Code of Conduct")
    assert company_doc["is_company_wide"] is True
    assert company_doc["branch_id"] is None

    dept_doc = next(d for d in docs if d["title"] == "Engineering Duty & Overtime Policy")
    assert dept_doc["is_company_wide"] is False
    assert dept_doc["branch_id"] == 1
    assert dept_doc["department_id"] == 1


def test_2_reingestion_idempotency_no_duplicate_chunks(client):
    """
    Test 2: Re-ingest the exact same document and confirm no duplicate chunks were created.
    Asserts chunk count stays the same after re-ingestion.
    """
    payload = {
        "title": "Remote Work Guidelines",
        "content": "Employees may work remotely up to 2 days per week with manager coordination. High-speed internet is required.",
        "branch_id": 1,
        "department_id": None,
    }

    # Initial ingestion
    res1 = client.post("/api/v1/ingestion/text", json=payload)
    assert res1.status_code == 201
    initial_chunks = res1.json()["chunk_count"]

    collection = get_vector_collection()
    initial_collection_count = collection.count()
    assert initial_collection_count == initial_chunks

    # Re-ingest the EXACT SAME document
    res2 = client.post("/api/v1/ingestion/text", json=payload)
    assert res2.status_code == 201
    reingest_chunks = res2.json()["chunk_count"]
    assert reingest_chunks == initial_chunks

    # Verify collection count has not duplicated
    after_collection_count = collection.count()
    assert after_collection_count == initial_collection_count

    # Check documents registry count
    list_res = client.get("/api/v1/ingestion/documents")
    docs = list_res.json()
    assert len(docs) == 1
    assert docs[0]["chunk_count"] == initial_chunks


def test_3_embedding_vector_dimensions():
    """
    Test 3: Verify chunk embeddings have the expected vector dimension
    for the chosen model (all-MiniLM-L6-v2 -> 384 dimensions).
    """
    sample_text = "This is a sample human resources policy paragraph."
    embedding = generate_embedding(sample_text)

    assert isinstance(embedding, list)
    assert len(embedding) == 384
    assert all(isinstance(val, float) for val in embedding)

    batch_embeddings = generate_embeddings_batch([sample_text, "Second chunk"])
    assert len(batch_embeddings) == 2
    assert len(batch_embeddings[0]) == 384
    assert len(batch_embeddings[1]) == 384


def test_4_semantic_vector_search_returns_relevant_chunk(client):
    """
    Test 4: Ingest a document and search via vector similarity with a question,
    confirming the correct chunk comes back as top result.
    """
    client.post(
        "/api/v1/ingestion/text",
        json={
            "title": "Maternity and Paternity Benefits",
            "content": "Female employees are entitled to 16 weeks of fully paid maternity leave. Male employees receive 4 weeks of paid paternity leave upon birth or adoption.",
            "branch_id": None,
            "department_id": None,
        },
    )

    # Search with a semantic question
    search_res = client.post(
        "/api/v1/ingestion/search",
        json={
            "query": "How many weeks of paternity leave do fathers get?",
            "top_k": 2,
        },
    )
    assert search_res.status_code == 200
    results = search_res.json()["results"]
    assert len(results) > 0

    top_result = results[0]
    assert "paternity leave" in top_result["content"].lower()
    assert "4 weeks" in top_result["content"]
    assert top_result["score"] > 0.4  # High semantic similarity


def test_5_empty_and_corrupt_file_upload_error_handling(client):
    """
    Test 5: Test with empty file and unsupported/corrupt formats to confirm
    clear HTTP 400 error is returned, not a crash.
    """
    # 1. Empty file upload
    empty_file = ("empty.txt", b"", "text/plain")
    res_empty = client.post(
        "/api/v1/ingestion/upload",
        files={"file": empty_file},
    )
    assert res_empty.status_code == 400
    assert "empty" in res_empty.json()["detail"].lower()

    # 2. Corrupt PDF upload
    corrupt_pdf = ("corrupt.pdf", b"%PDF-1.4 corrupt random binary content that is invalid", "application/pdf")
    res_corrupt = client.post(
        "/api/v1/ingestion/upload",
        files={"file": corrupt_pdf},
    )
    assert res_corrupt.status_code == 400
    assert "corrupt or invalid pdf" in res_corrupt.json()["detail"].lower()

    # 3. Unsupported extension
    invalid_file = ("script.exe", b"binary content", "application/octet-stream")
    res_invalid = client.post(
        "/api/v1/ingestion/upload",
        files={"file": invalid_file},
    )
    assert res_invalid.status_code == 400
    assert "unsupported file format" in res_invalid.json()["detail"].lower()
