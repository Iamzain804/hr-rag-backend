from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class UserContext(BaseModel):
    user_id: int
    email: str
    full_name: str
    role: str
    branch_id: Optional[int] = None
    branch_name: Optional[str] = None
    department_id: Optional[int] = None
    department_name: Optional[str] = None
    permissions: List[str] = []


class ChatRequestJSON(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    attachment_text: Optional[str] = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    content: str
    source_document: str
    chunk_index: int = 0
    score: float
    rerank_score: Optional[float] = None
    branch_id: Optional[int] = None
    department_id: Optional[int] = None
    is_company_wide: bool = True



class ChatResponseMeta(BaseModel):
    status: str
    grounded: bool
    fallback_used: bool
    retrieved_chunks: List[RetrievedChunk] = []
    attachment_detected: bool = False
    ocr_extracted_length: int = 0
    model_used: Optional[str] = None
