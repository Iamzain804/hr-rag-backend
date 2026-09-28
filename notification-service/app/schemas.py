from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class SendTempPasswordRequest(BaseModel):
    email: EmailStr = Field(..., description="Recipient email address")
    first_name: str = Field(..., min_length=1, max_length=100, description="Recipient's first name")
    temp_password: str = Field(..., min_length=6, description="Temporary auto-generated password")
    app_url: Optional[str] = Field(default="http://localhost:5173", description="Frontend application login URL")


class SendEmailRequest(BaseModel):
    to_email: EmailStr = Field(..., description="Recipient email address")
    subject: str = Field(..., min_length=1, max_length=255, description="Email subject line")
    body_text: str = Field(..., min_length=1, description="Plain-text email body")
    body_html: Optional[str] = Field(None, description="HTML formatted email body")


class NotificationResponse(BaseModel):
    status: str = "success"
    message: str
    recipient: str
