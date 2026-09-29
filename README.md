# 🏢 HR RAG Assistant — Backend Microservices

![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)
![Security Passed](https://img.shields.io/badge/security-SonarQube%20%7C%20Trivy%20%7C%20Snyk-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![Tests](https://img.shields.io/badge/tests-64%20passed-success.svg)
![Architecture](https://img.shields.io/badge/architecture-Microservices%20(FastAPI)-orange.svg)

Enterprise-grade Retrieval-Augmented Generation (RAG) backend platform for intelligent HR policy querying, document parsing, role-based access control (RBAC), multi-branch data isolation, and real-time Server-Sent Events (SSE) streaming.

---

## 🏗️ Architecture Overview

The backend is built as an asynchronous, decoupled microservices ecosystem:

```
                                  ┌────────────────────────┐
                                  │      React Client      │
                                  │   (Vite + Tailwind)    │
                                  └───────────┬────────────┘
                                              │
                                              │ HTTP / SSE / REST
                                              ▼
       ┌───────────────────┬───────────────────┬───────────────────┐
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ Identity &   │    │ Ingestion    │    │ RAG Chat     │    │ Notification │
│ Org Service  │    │ Service      │    │ Service      │    │ Service      │
│  (Port 8001) │    │  (Port 8003) │    │  (Port 8004) │    │  (Port 8002) │
└──────┬───────┘    └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  SQLite / PG │    │ PyPDF / OCR  │    │ Milvus /     │    │ SSE / Email  │
│  Auth DB     │    │ Docx Parser  │    │ FAISS Vector │    │ Queue Engine │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### Microservices Breakdown

| Service | Port | Description | Core Stack |
| :--- | :--- | :--- | :--- |
| **`identity-org-service`** | `8001` | JWT authentication, user & branch management, role-based access control (Admin, HR Manager, Employee). | FastAPI, SQLAlchemy, PyJWT, Passlib (bcrypt) |
| **`notification-service`** | `8002` | Real-time notification dispatch, system alerts, broadcast announcements, event logs. | FastAPI, Asyncio, WebSockets / SSE |
| **`ingestion-service`** | `8003` | Multi-format document parser (PDF, DOCX, TXT, OCR), chunking pipeline, branch metadata tagging. | FastAPI, pdfplumber, pytesseract, python-docx |
| **`rag-chat-service`** | `8004` | Semantic vector search, Groq LLM rotation pool, multi-tenant branch data isolation, semantic caching, token streaming. | FastAPI, LangChain, Milvus/FAISS, SentenceTransformers, Groq API |

---

## ✨ Key Features

- **⚡ Multi-Key Groq LLM Rotation**: Automatic fallback across API keys to bypass rate limits (`llama-3.3-70b-versatile`, `mixtral-8x7b-32768`).
- **🛡️ Multi-Branch Data Isolation**: Semantic vector queries are strictly partitioned by `branch_id`. Employees in Branch A cannot retrieve confidential documents belonging to Branch B.
- **⚡ Semantic Caching**: Sub-millisecond response caching for recurring queries while respecting tenant isolation.
- **📄 Robust Document Ingestion**: Handles raw text, scanned PDFs with OCR fallback, Word documents, and metadata-aware chunking.
- **🔒 Enterprise Security (Phase 9)**:
  - SonarQube static code security analysis.
  - Trivy & Snyk automated filesystem/dependency CVE scanning.
  - Zero-trust RBAC (Role-Based Access Control) & strict JWT signature/expiry validation.
  - GitHub Secret Push Protection (GH013) compliant configuration.

---

## 📁 Repository Structure

```
backend/
├── .github/workflows/
│   ├── ci.yml                 # Automated test matrix & linting
│   └── security.yml           # SonarQube, Trivy, and Snyk security scans
├── identity-org-service/      # Authentication & organization management
│   ├── app/                   # FastAPI routes, models, schemas, auth utils
│   └── tests/                 # Unit & security authorization tests
├── notification-service/      # Real-time event notifications
│   ├── app/
│   └── tests/
├── ingestion-service/         # Document ingestion & OCR pipeline
│   ├── app/
│   └── tests/
├── rag-chat-service/          # Vector retrieval, semantic cache, LLM streaming
│   ├── app/
│   └── tests/
├── run_all.py                 # Master cross-platform orchestration runner
├── run-all.ps1                # PowerShell launcher
├── run-all.sh                 # Unix/macOS launcher
└── README.md
```

---

## 🚀 Quick Start & Installation

### Prerequisites
- Python 3.10+
- Tesseract OCR (optional, for scanned PDF OCR extraction)

### 1. Clone the Repository
```bash
git clone https://github.com/Iamzain804/hr-rag-backend.git
cd hr-rag-backend
```

### 2. Configure Environment Variables
Create a `.env` file in `rag-chat-service/` or the root workspace:

```env
# Groq API Keys (comma-separated or JSON list for automatic failover)
GROQ_API_KEYS=gsk_your_key_1,gsk_your_key_2

# JWT Secret & Encryption
SECRET_KEY=your_super_secret_jwt_key_here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=1440

# Vector Database (Milvus / FAISS fallback)
VECTOR_DB_TYPE=faiss
```

### 3. Launch All Microservices
Use the master python runner to launch all 4 services with live logging and health-check monitoring:

```bash
python run_all.py
```

*Or individually:*
```bash
# Identity & Org
uvicorn identity-org-service.app.main:app --port 8001 --reload

# Notification Service
uvicorn notification-service.app.main:app --port 8002 --reload

# Ingestion Service
uvicorn ingestion-service.app.main:app --port 8003 --reload

# RAG Chat Service
uvicorn rag-chat-service.app.main:app --port 8004 --reload
```

---

## 🧪 Testing & Security Verification

Run the full test suite across all microservices:

```bash
# Run all 64 automated tests
pytest backend/identity-org-service/tests
pytest backend/notification-service/tests
pytest backend/ingestion-service/tests
pytest backend/rag-chat-service/tests
```

### Security & RBAC Verification Tests
- `identity-org-service/tests/test_phase9_security.py`: Verifies non-admin users receive `403 Forbidden` on admin routes and expired JWTs are rejected with `401 Unauthorized`.
- `rag-chat-service/tests/test_phase9_security.py`: Validates multi-branch data isolation, prompt injection resistance, and semantic cache tenant separation.

---

## 🛡️ Security Compliance

This repository conforms to **Phase 9 Enterprise Security Standards**:
- **SAST**: Scanned with SonarQube / SonarCloud.
- **Vulnerability Scanning**: Continuous scans via Trivy and Snyk.
- **Network Hardening**: Local services bound strictly to `127.0.0.1`.
- **Zero Secret Leaks**: Strict gitignore policy, externalized credentials.

---

## 📄 License

Licensed under the [MIT License](LICENSE).
