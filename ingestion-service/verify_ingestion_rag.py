import os
import sys

# Ensure app is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.vector_store import clear_all_data, ingest_document, search_vector_store, list_ingested_documents

def run_manual_verification():
    print("=== Clearing test storage ===")
    clear_all_data()

    sample_doc_title = "Global Healthcare & Dental Coverage Policy 2026"
    sample_doc_content = """
    POLICY SECTION 1: Standard Health Insurance
    All permanent employees at TechCorp are enrolled from Day 1 in the Premium Health Insurance Plan.
    The plan covers inpatient hospitalization, specialist consultations, prescription medications with a 90% reimbursement rate, and emergency evacuation up to $500,000.
    
    POLICY SECTION 2: Dental and Vision Benefits
    Employees are entitled to annual dental checkups, cleaning, and up to $1,500 per year in dental restorative procedures.
    Vision care covers one comprehensive eye examination every 12 months and provides a $300 biennial allowance for prescription eyewear or contact lenses.
    
    POLICY SECTION 3: Mental Health & Wellness Stipend
    TechCorp provides 10 free confidential therapy sessions per year through our Employee Assistance Program (EAP), alongside an annual $600 wellness stipend for gym memberships or meditation apps.
    """

    print("\n=== Ingesting Sample Policy Document ===")
    res = ingest_document(
        title=sample_doc_title,
        source_document="TechCorp_Health_and_Dental_Policy_2026.pdf",
        raw_text=sample_doc_content,
        branch_id=None,
        department_id=None
    )
    print(f"Ingestion Result: status={res.status}, doc_id={res.doc_id}, chunks={res.chunk_count}, is_company_wide={res.is_company_wide}")

    print("\n=== Listing Ingested Documents ===")
    docs = list_ingested_documents()
    for d in docs:
        print(f" - Doc ID: {d.doc_id} | Title: '{d.title}' | Chunks: {d.chunk_count} | Company-wide: {d.is_company_wide}")

    sample_queries = [
        "What is the annual allowance for dental restorative procedures?",
        "How many confidential therapy sessions can employees take?"
    ]

    for q in sample_queries:
        print(f"\n=== Executing Direct Vector Query: '{q}' ===")
        results = search_vector_store(query=q, top_k=2)
        print(f"Top {len(results)} search results retrieved:")
        for idx, item in enumerate(results, 1):
            print(f"\n--- Result #{idx} (Similarity Score: {item.score:.4f}, Source: {item.source_document}) ---")
            print(f"Chunk ID: {item.chunk_id}")
            print(f"Retrieved Chunk Content:\n{item.content}")

if __name__ == "__main__":
    run_manual_verification()
