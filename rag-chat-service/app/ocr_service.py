import io
import logging
from typing import Optional
from PIL import Image
import pytesseract
from fastapi import HTTPException, UploadFile, status

logger = logging.getLogger("rag_chat_service.ocr")


def extract_text_from_image_bytes(image_bytes: bytes) -> str:
    """
    Extract text from raw image bytes using Pillow and PyTesseract.
    Note: Requires the Tesseract-OCR binary to be installed on the host OS / PATH.
    """
    if not image_bytes:
        return ""

    try:
        image = Image.open(io.BytesIO(image_bytes))
        # Convert RGBA/P to RGB for clean OCR
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")

        extracted_text = pytesseract.image_to_string(image)
        return extracted_text.strip()

    except pytesseract.TesseractNotFoundError:
        logger.warning(
            "Tesseract binary not found on host system. OCR is unavailable until Tesseract is installed."
        )
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Tesseract OCR binary is not installed on the server. Please install Tesseract-OCR or upload text directly.",
        )
    except Exception as exc:
        logger.error(f"Error during image OCR processing: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to process image attachment for OCR: {str(exc)}",
        )


async def process_attachment_file(file: UploadFile) -> str:
    """
    Read uploaded attachment file (Image or Text) and extract raw text.
    """
    content = await file.read()
    if not content:
        return ""

    content_type = file.content_type or ""
    filename = (file.filename or "").lower()

    if (
        "image" in content_type
        or filename.endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"))
    ):
        return extract_text_from_image_bytes(content)
    elif "text" in content_type or filename.endswith(".txt"):
        try:
            return content.decode("utf-8").strip()
        except UnicodeDecodeError:
            return content.decode("latin-1", errors="ignore").strip()
    else:
        # Default try as image first, fallback to text
        try:
            return extract_text_from_image_bytes(content)
        except Exception:
            return content.decode("utf-8", errors="ignore").strip()
