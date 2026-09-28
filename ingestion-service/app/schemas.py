from typing import List, Optional
from pydantic import BaseModel, Field


class IngestTextRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=255, description="Document title")
    content: str = Field(..., min_length=1, description="Document raw text content")
    branch_id: Optional[int] = Field(None, description="Branch ID tag (null for company-wide)")
    department_id: Optional[int] = Field(None, description="Department ID tag (null for company-wide)")


class IngestionResponse(BaseModel):
    status: str = "success"
    message: str
    doc_id: str
    title: str
    source_document: str
    branch_id: Optional[int] = None
    department_id: Optional[int] = None
    is_company_wide: bool
    chunk_count: int
    doc_hash: str


class DocumentItem(BaseModel):
    doc_id: str
    title: str
    source_document: str
    branch_id: Optional[int] = None
    department_id: Optional[int] = None
    is_company_wide: bool
    chunk_count: int
    doc_hash: str
    created_at: str


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search question / query")
    top_k: int = Field(default=4, ge=1, le=20, description="Number of results to retrieve")
    branch_id: Optional[int] = Field(None, description="Caller's branch ID")
    department_id: Optional[int] = Field(None, description="Caller's department ID")


class SearchResultItem(BaseModel):
    chunk_id: str
    content: str
    source_document: str
    chunk_index: int
    score: float
    branch_id: Optional[int] = None
    department_id: Optional[int] = None
    is_company_wide: bool


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResultItem]
    total_found: int


class ScopeVersionResponse(BaseModel):
    scope_version: str
    document_count: int
    last_updated: Optional[str] = None
    branch_id: Optional[int] = None
    department_id: Optional[int] = None

