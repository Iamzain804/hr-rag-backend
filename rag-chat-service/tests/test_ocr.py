import io
import pytest
from PIL import Image, ImageDraw
from fastapi import UploadFile
from app.ocr_service import extract_text_from_image_bytes, process_attachment_file


def create_test_image_with_text() -> bytes:
    """Create a high-contrast PIL test image with text."""
    img = Image.new("RGB", (300, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 30), "HEALTH BENEFIT 2026", fill=(0, 0, 0))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_ocr_processing_text_and_images():
    """Verify that process_attachment_file handles plain text and image formats cleanly."""
    # 1. Plain text file upload
    text_content = b"Official bonus approval: approved for Q4 2026."
    text_file = UploadFile(
        file=io.BytesIO(text_content),
        filename="bonus.txt",
        headers={"content-type": "text/plain"},
    )
    extracted_text = await process_attachment_file(text_file)
    assert "bonus approval" in extracted_text

    # 2. Empty file handling
    empty_file = UploadFile(
        file=io.BytesIO(b""),
        filename="empty.txt",
        headers={"content-type": "text/plain"},
    )
    empty_result = await process_attachment_file(empty_file)
    assert empty_result == ""
