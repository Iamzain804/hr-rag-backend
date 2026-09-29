# 🔒 Security Policy — HR RAG Assistant Backend

## Supported Versions

We release security patches and vulnerability updates for the active versions of the **HR RAG Assistant** microservices platform.

| Version | Supported          | Status |
| ------- | ------------------ | ------ |
| 1.0.x   | :white_check_mark: | Active Support & Monitoring |
| < 1.0   | :x:                | Unsupported |

---

## 🛡️ Security Architecture & Defense-in-Depth

The backend platform is engineered with a multi-tier security framework:
- **Role-Based Access Control (RBAC)**: Zero-trust route authorization on all administrative endpoints.
- **Tenant & Branch Data Isolation**: Vector embeddings and semantic search are strictly partitioned by `branch_id`.
- **AI Safety & Guardrails**: NVIDIA NeMo Guardrails (`rails.co`) protect against prompt injections, off-topic misuse, and hallucinations.
- **Continuous Static & Dependency Scanning**: Automated CI pipelines running **SonarQube**, **Trivy**, and **Snyk**.
- **Secret Scanning & Push Protection**: Strict compliance with GitHub Push Protection (GH013).

---

## 📢 Reporting a Vulnerability

We take the security and integrity of our platform very seriously. If you discover a vulnerability, security flaw, or sensitive data exposure, please follow these responsible disclosure steps:

### 1. Private Vulnerability Reporting (Preferred)
- Use the **[Private Vulnerability Reporting](https://github.com/Iamzain804/hr-rag-backend/security/advisories/new)** feature directly on our GitHub repository.
- This creates an encrypted, private channel between you and the maintainers.

### 2. What to Include in Your Report
To help us resolve the issue efficiently, please provide:
- A clear description of the vulnerability.
- Affected service (`identity-org-service`, `notification-service`, `ingestion-service`, or `rag-chat-service`).
- Step-by-step reproduction steps or a minimal Proof of Concept (PoC).
- Potential impact and severity assessment.

### 3. Response & Resolution SLA
- **Initial Acknowledgment**: Within 24 hours.
- **Severity Assessment & Triage**: Within 48 hours.
- **Patch Deployment**: High and critical vulnerabilities will receive emergency patches within 3–5 business days.

---

## 🚫 Please Do Not:
- Publicly disclose or discuss the vulnerability in GitHub Issues, Discussions, or social media before a patch is published.
- Execute destructive testing, DDoS attacks, or access data belonging to other users.

Thank you for helping keep the **HR RAG Assistant Platform** safe and secure!
