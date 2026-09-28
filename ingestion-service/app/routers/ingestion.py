from typing import List, Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.chunker import extract_text_from_pdf
from app.schemas import (
    DocumentItem,
    IngestionResponse,
    IngestTextRequest,
    ScopeVersionResponse,
    SearchRequest,
    SearchResponse,
)
from app.vector_store import (
    get_scope_ingestion_version,
    ingest_document,
    list_ingested_documents,
    search_vector_store,
)


router = APIRouter(prefix="/api/v1/ingestion", tags=["Ingestion"])


@router.post(
    "/upload",
    response_model=IngestionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload & Ingest Document (PDF or Text)",
)
async def upload_document(
    file: UploadFile = File(..., description="Document file (.pdf or .txt)"),
    title: Optional[str] = Form(None, description="Document title (defaults to filename)"),
    branch_id: Optional[int] = Form(None, description="Branch ID tag (null for company-wide)"),
    department_id: Optional[int] = Form(None, description="Department ID tag (null for company-wide)"),
):
    """
    Ingest a document file (PDF or TXT) tagged with optional branch and department IDs.
    Extracts text, creates overlapping chunks, computes local embeddings, and stores in ChromaDB.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a valid filename",
        )

    content_bytes = await file.read()
    if not content_bytes or len(content_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    filename_lower = file.filename.lower()
    doc_title = title if title and title.strip() else file.filename

    raw_text = ""
    if filename_lower.endswith(".pdf"):
        try:
            raw_text = extract_text_from_pdf(content_bytes)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Corrupt or invalid PDF file: {str(exc)}",
            )
    elif filename_lower.endswith((".txt", ".md", ".csv", ".json")):
        try:
            raw_text = content_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                raw_text = content_bytes.decode("latin-1")
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unable to decode text file: {str(exc)}",
                )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{file.filename}'. Supported formats: .pdf, .txt, .md",
        )

    if not raw_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Extracted document text is empty.",
        )

    try:
        response = ingest_document(
            title=doc_title,
            source_document=file.filename,
            raw_text=raw_text,
            branch_id=branch_id,
            department_id=department_id,
        )
        return response
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document into vector database: {str(exc)}",
        )


@router.post(
    "/text",
    response_model=IngestionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest Raw Text Document",
)
def ingest_text_endpoint(payload: IngestTextRequest):
    """
    Ingest raw text content tagged with optional branch and department IDs.
    """
    if not payload.content.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document content cannot be empty.",
        )

    try:
        response = ingest_document(
            title=payload.title,
            source_document=f"{payload.title}.txt",
            raw_text=payload.content,
            branch_id=payload.branch_id,
            department_id=payload.department_id,
        )
        return response
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest text document: {str(exc)}",
        )


@router.get(
    "/documents",
    response_model=List[DocumentItem],
    summary="List Ingested Documents",
)
def get_documents():
    """
    List all ingested documents and their branch/department scope tags,
    chunk counts, and content hashes without directly querying the vector database.
    """
    return list_ingested_documents()


@router.get(
    "/version",
    response_model=ScopeVersionResponse,
    summary="Get Ingestion Version for Scope",
)
def get_version(
    branch_id: Optional[int] = None,
    department_id: Optional[int] = None,
):
    """
    Get the deterministic scope ingestion version. Used by rag-chat-service
    to invalidate cached responses when documents in a scope change.
    """
    return get_scope_ingestion_version(branch_id=branch_id, department_id=department_id)



@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Semantic Vector Search",
)
def search_documents(payload: SearchRequest):
    """
    Perform semantic vector similarity search filtered by caller's branch/department context.
    """
    results = search_vector_store(
        query=payload.query,
        top_k=payload.top_k,
        branch_id=payload.branch_id,
        department_id=payload.department_id,
    )
    return SearchResponse(
        query=payload.query,
        results=results,
        total_found=len(results),
    )
