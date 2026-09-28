Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "Starting All Backend Microservices (PowerShell Launcher)" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan

$baseDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Start Identity Service
Write-Host " -> Starting Identity & Org Service on http://127.0.0.1:8001..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$baseDir\identity-org-service'; uv run uvicorn app.main:app --port 8001 --reload"

# Start Notification Service
Write-Host " -> Starting Notification Service on http://127.0.0.1:8002..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$baseDir\notification-service'; uv run uvicorn app.main:app --port 8002 --reload"

# Start Ingestion Service
Write-Host " -> Starting Ingestion Service on http://127.0.0.1:8003..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$baseDir\ingestion-service'; uv run uvicorn app.main:app --port 8003 --reload"

# Start RAG Chat Service
Write-Host " -> Starting RAG Chat Service on http://127.0.0.1:8004..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$baseDir\rag-chat-service'; uv run uvicorn app.main:app --port 8004 --reload"

Write-Host "`nAll 4 backend services launched in separate windows!" -ForegroundColor Yellow
Write-Host "Frontend start karne ke liye: cd ..\frontend && npm run dev" -ForegroundColor Yellow
