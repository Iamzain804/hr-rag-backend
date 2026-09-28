import asyncio
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

import aiosmtplib
from app.config import settings

logger = logging.getLogger("notification_service")


class SMTPDeliveryError(Exception):
    """Raised when an email cannot be delivered via SMTP after retries."""
    def __init__(self, message: str, original_exception: Optional[Exception] = None):
        super().__init__(message)
        self.original_exception = original_exception


def build_temp_password_email(first_name: str, temp_password: str, app_url: str) -> tuple[str, str]:
    """Generate both plain-text and HTML versions of the temporary password email."""
    subject = "Welcome to HR Assistant — Your Temporary Login Credentials"

    plain_text = f"""Hello {first_name},

Welcome to the HR RAG Assistant platform! An account has been created for you.

Your temporary credentials:
Temporary Password: {temp_password}

Login URL: {app_url}

IMPORTANT SECURITY NOTICE:
This temporary password is valid for initial login only. You will be required to set a new password upon your first sign-in before accessing company resources.

If you did not request this account, please notify your system administrator immediately.

Best regards,
{settings.SMTP_FROM_NAME}
"""

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{subject}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: #f4f6f9;
            margin: 0;
            padding: 24px;
            color: #1e293b;
        }}
        .container {{
            max-width: 580px;
            margin: 0 auto;
            background-color: #ffffff;
            border-radius: 12px;
            padding: 32px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.06);
            border: 1px solid #e2e8f0;
        }}
        .header {{
            text-align: center;
            padding-bottom: 20px;
            border-bottom: 2px solid #f1f5f9;
        }}
        .logo-text {{
            font-size: 22px;
            font-weight: 700;
            color: #2563eb;
            letter-spacing: -0.5px;
        }}
        .content {{
            padding: 24px 0;
            line-height: 1.6;
        }}
        .greeting {{
            font-size: 18px;
            font-weight: 600;
            color: #0f172a;
        }}
        .cred-card {{
            background: linear-gradient(135deg, #f8fafc 0%, #eff6ff 100%);
            border: 1px solid #bfdbfe;
            border-radius: 8px;
            padding: 18px 24px;
            margin: 20px 0;
            text-align: center;
        }}
        .cred-label {{
            font-size: 12px;
            font-weight: 600;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }}
        .cred-val {{
            font-family: 'Courier New', Courier, monospace;
            font-size: 20px;
            font-weight: 700;
            color: #1e3a8a;
            letter-spacing: 1.5px;
            margin-top: 6px;
            user-select: all;
        }}
        .alert-box {{
            background-color: #fffbeb;
            border-left: 4px solid #f59e0b;
            padding: 12px 16px;
            border-radius: 4px;
            font-size: 13px;
            color: #92400e;
            margin: 16px 0;
        }}
        .btn {{
            display: inline-block;
            background-color: #2563eb;
            color: #ffffff !important;
            text-decoration: none;
            font-weight: 600;
            font-size: 15px;
            padding: 12px 28px;
            border-radius: 6px;
            margin: 16px 0;
            text-align: center;
        }}
        .footer {{
            margin-top: 24px;
            padding-top: 16px;
            border-top: 1px solid #f1f5f9;
            font-size: 12px;
            color: #94a3b8;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo-text">HR RAG Assistant</div>
        </div>
        <div class="content">
            <p class="greeting">Hello {first_name},</p>
            <p>Welcome! An account has been created for you on the <strong>HR RAG Assistant</strong> platform.</p>
            
            <div class="cred-card">
                <div class="cred-label">Temporary Password</div>
                <div class="cred-val">{temp_password}</div>
            </div>

            <div class="alert-box">
                <strong>Important Security Notice:</strong> This password is for single-use. You will be prompted to choose a permanent, secure password upon your first sign-in.
            </div>

            <div style="text-align: center;">
                <a href="{app_url}" class="btn" target="_blank">Sign In to Your Account</a>
            </div>
        </div>
        <div class="footer">
            <p>This is an automated notification from {settings.SMTP_FROM_NAME}. Please do not reply directly to this email.</p>
        </div>
    </div>
</body>
</html>
"""
    return subject, plain_text, html_content


async def send_email_async(
    to_email: str,
    subject: str,
    body_text: str,
    body_html: Optional[str] = None,
) -> None:
    """
    Send an email via SMTP with automatic retry logic and structured error handling.
    """
    message = MIMEMultipart("alternative")
    message["Subject"] = subject
    message["From"] = f"{settings.SMTP_FROM_NAME} <{settings.SMTP_FROM_EMAIL}>"
    message["To"] = to_email

    # Plain text alternative
    message.attach(MIMEText(body_text, "plain", "utf-8"))

    # HTML alternative if provided
    if body_html:
        message.attach(MIMEText(body_html, "html", "utf-8"))

    last_error: Optional[Exception] = None
    total_attempts = 1 + max(0, settings.SMTP_MAX_RETRIES)

    for attempt in range(1, total_attempts + 1):
        try:
            logger.info(f"Attempting SMTP delivery to {to_email} (attempt {attempt}/{total_attempts})...")
            await aiosmtplib.send(
                message,
                hostname=settings.SMTP_HOST,
                port=settings.SMTP_PORT,
                username=settings.SMTP_USERNAME or None,
                password=settings.SMTP_PASSWORD or None,
                start_tls=settings.SMTP_USE_TLS,
                use_tls=settings.SMTP_USE_SSL,
                timeout=10,
            )
            logger.info(f"Successfully sent email to {to_email}")
            return
        except (aiosmtplib.SMTPException, OSError, ConnectionError, asyncio.TimeoutError) as exc:
            last_error = exc
            logger.warning(
                f"SMTP delivery attempt {attempt} to {to_email} failed: {type(exc).__name__}: {str(exc)}"
            )
            if attempt < total_attempts:
                await asyncio.sleep(settings.SMTP_RETRY_BACKOFF_SECONDS * attempt)
        except Exception as exc:
            last_error = exc
            logger.error(f"Unexpected SMTP error sending to {to_email}: {str(exc)}")
            break

    raise SMTPDeliveryError(
        f"Failed to deliver email to '{to_email}' after {total_attempts} attempts: {str(last_error)}",
        original_exception=last_error,
    )
