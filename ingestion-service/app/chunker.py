import io
from typing import List
from pypdf import PdfReader
from app.config import settings


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract clean text from PDF binary data using pypdf."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        extracted_text = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                extracted_text.append(page_text.strip())
        
        full_text = "\n\n".join(extracted_text).strip()
        if not full_text:
            raise ValueError("PDF contains no extractable text (it may be an empty or scanned image file).")
        return full_text
    except Exception as exc:
        raise ValueError(f"Failed to read and parse PDF: {str(exc)}")


def chunk_text(
    text: str,
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
) -> List[str]:
    """
    Split text into overlapping character chunks with boundary preservation.
    
    Why this strategy?
    - Chunk Size: 500 characters (~100-120 words).
      HR policies, duty timings, and leave rules are typically concise bullet points or short clauses.
      500 chars captures the complete rule context without diluting vector similarity.
    - Chunk Overlap: 100 characters (~20 words).
      Ensures continuity across adjacent sentences and prevents clause cutoff at chunk boundaries.
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        return []

    if len(cleaned_text) <= chunk_size:
        return [cleaned_text]

    chunks = []
    start = 0
    text_length = len(cleaned_text)

    while start < text_length:
        end = start + chunk_size
        
        if end >= text_length:
            chunk = cleaned_text[start:].strip()
            if chunk:
                chunks.append(chunk)
            break

        # Try to break cleanly at paragraph or newline or space boundary
        break_point = -1
        # Look for double newline first (paragraph)
        p_idx = cleaned_text.rfind("\n\n", start + chunk_overlap, end)
        if p_idx != -1:
            break_point = p_idx + 2
        else:
            # Look for single newline
            n_idx = cleaned_text.rfind("\n", start + chunk_overlap, end)
            if n_idx != -1:
                break_point = n_idx + 1
            else:
                # Look for sentence boundary or space
                s_idx = cleaned_text.rfind(". ", start + chunk_overlap, end)
                if s_idx != -1:
                    break_point = s_idx + 2
                else:
                    sp_idx = cleaned_text.rfind(" ", start + chunk_overlap, end)
                    if sp_idx != -1:
                        break_point = sp_idx + 1

        if break_point != -1 and break_point > start:
            chunk = cleaned_text[start:break_point].strip()
            if chunk:
                chunks.append(chunk)
            start = max(start + 1, break_point - chunk_overlap)
        else:
            chunk = cleaned_text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - chunk_overlap

    return chunks
