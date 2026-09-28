from unittest.mock import AsyncMock, patch
import aiosmtplib
import pytest


def test_send_temp_password_success(client):
    """
    Test 1: Mocks the SMTP call and verifies the endpoint returns 200 on success
    with correct message payload passed to the SMTP client.
    """
    with patch("app.email_service.aiosmtplib.send", new_callable=AsyncMock) as mock_send:
        mock_send.return_value = ({"test.user@example.com": (250, "OK")}, "250 2.0.0 OK")

        payload = {
            "email": "test.user@example.com",
            "first_name": "Alexander",
            "temp_password": "TempSecret@9876",
            "app_url": "http://localhost:5173/login",
        }

        response = client.post("/api/v1/notify/send-temp-password", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["recipient"] == "test.user@example.com"
        assert "sent successfully" in data["message"]

        # Verify SMTP send was called with correct message headers
        assert mock_send.call_count == 1
        sent_message = mock_send.call_args[0][0]
        assert sent_message["To"] == "test.user@example.com"
        assert "Temporary Login Credentials" in sent_message["Subject"]

        # Check payload contents in both text/plain and text/html parts
        parts = sent_message.get_payload()
        plain_part = parts[0].get_payload(decode=True).decode("utf-8")
        html_part = parts[1].get_payload(decode=True).decode("utf-8")

        assert "Alexander" in plain_part
        assert "TempSecret@9876" in plain_part
        assert "http://localhost:5173/login" in plain_part

        assert "Alexander" in html_part
        assert "TempSecret@9876" in html_part
        assert "http://localhost:5173/login" in html_part


@pytest.mark.parametrize(
    "invalid_payload",
    [
        # Invalid email format
        {"email": "not-an-email", "first_name": "John", "temp_password": "Pass@123456"},
        # Missing email
        {"first_name": "John", "temp_password": "Pass@123456"},
        # Missing first_name
        {"email": "john@example.com", "temp_password": "Pass@123456"},
        # Missing temp_password
        {"email": "john@example.com", "first_name": "John"},
        # Empty first_name
        {"email": "john@example.com", "first_name": "", "temp_password": "Pass@123456"},
        # Too short temp_password
        {"email": "john@example.com", "first_name": "John", "temp_password": "123"},
    ],
)
def test_send_temp_password_validation_error_returns_422(client, invalid_payload):
    """Test 2: Missing or invalid email format/fields return HTTP 422 validation error."""
    response = client.post("/api/v1/notify/send-temp-password", json=invalid_payload)
    assert response.status_code == 422


def test_smtp_failure_returns_502_bad_gateway(client):
    """
    Test 3: Simulates SMTP failure and verifies endpoint returns 502 with structured error JSON,
    rather than leaking raw Python traceback or unhandled 500 crash.
    """
    with patch(
        "app.email_service.aiosmtplib.send",
        side_effect=aiosmtplib.SMTPConnectError("Connection refused by SMTP server"),
    ) as mock_send:
        payload = {
            "email": "fail.test@example.com",
            "first_name": "Failure",
            "temp_password": "TempSecret@1234",
        }

        response = client.post("/api/v1/notify/send-temp-password", json=payload)

        # Returns 502 Bad Gateway with clear message
        assert response.status_code == 502
        data = response.json()
        assert "detail" in data
        assert "SMTP gateway failed to deliver" in data["detail"]
        assert "Connection refused" in data["detail"]
        # Confirms retries were attempted before giving up
        assert mock_send.call_count == 3  # 1 initial + 2 retries


def test_smtp_transient_failure_retry_success(client):
    """Test retry mechanism: transient error on attempt 1, succeeds on attempt 2."""
    with patch(
        "app.email_service.aiosmtplib.send",
        side_effect=[
            aiosmtplib.SMTPServerDisconnected("Temporary network drop"),
            ({"user@example.com": (250, "OK")}, "250 OK"),
        ],
    ) as mock_send:
        payload = {
            "email": "user@example.com",
            "first_name": "RetryUser",
            "temp_password": "TempSecret@5678",
        }

        response = client.post("/api/v1/notify/send-temp-password", json=payload)

        assert response.status_code == 200
        assert mock_send.call_count == 2


def test_send_custom_email_success(client):
    """Test sending custom notification email."""
    with patch("app.email_service.aiosmtplib.send", new_callable=AsyncMock) as mock_send:
        response = client.post(
            "/api/v1/notify/send-email",
            json={
                "to_email": "broadcast@example.com",
                "subject": "System Maintenance Notice",
                "body_text": "System will be down for 5 minutes.",
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "success"
        assert mock_send.call_count == 1
