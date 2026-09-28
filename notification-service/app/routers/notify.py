from fastapi import APIRouter, HTTPException, status

from app.email_service import (
    SMTPDeliveryError,
    build_temp_password_email,
    send_email_async,
)
from app.schemas import (
    NotificationResponse,
    SendEmailRequest,
    SendTempPasswordRequest,
)

router = APIRouter(prefix="/api/v1/notify", tags=["Notification"])


@router.post(
    "/send-temp-password",
    response_model=NotificationResponse,
    status_code=status.HTTP_200_OK,
    summary="Send Temporary Password Notification Email",
)
async def send_temp_password(payload: SendTempPasswordRequest):
    """
    Sends a formatted welcome email containing the user's temporary login credentials.
    Includes security guidelines instructing the user to reset their password upon initial login.
    """
    subject, plain_text, html_content = build_temp_password_email(
        first_name=payload.first_name,
        temp_password=payload.temp_password,
        app_url=payload.app_url or "http://localhost:5173",
    )

    try:
        await send_email_async(
            to_email=payload.email,
            subject=subject,
            body_text=plain_text,
            body_html=html_content,
        )
    except SMTPDeliveryError as err:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"SMTP gateway failed to deliver temporary password notification: {str(err)}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while dispatching notification: {str(exc)}",
        )

    return NotificationResponse(
        status="success",
        message="Temporary password notification sent successfully",
        recipient=payload.email,
    )


@router.post(
    "/send-email",
    response_model=NotificationResponse,
    status_code=status.HTTP_200_OK,
    summary="Send Custom Notification Email",
)
async def send_custom_email(payload: SendEmailRequest):
    """Sends a general custom email with plain-text and optional HTML body."""
    try:
        await send_email_async(
            to_email=payload.to_email,
            subject=payload.subject,
            body_text=payload.body_text,
            body_html=payload.body_html,
        )
    except SMTPDeliveryError as err:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"SMTP delivery failed: {str(err)}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected notification failure: {str(exc)}",
        )

    return NotificationResponse(
        status="success",
        message="Email sent successfully",
        recipient=payload.to_email,
    )
