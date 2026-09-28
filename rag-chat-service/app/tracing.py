from collections import deque
import logging
import re
import time
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field

from app.config import settings

logger = logging.getLogger("rag_chat_service.tracing")

# In-memory circular buffer of the last 100 traces
_trace_buffer: deque = deque(maxlen=100)
_trace_map: Dict[str, "TraceRun"] = {}

# PII Regex Patterns
EMAIL_REGEX = re.compile(r'[\w\.-]+@[\w\.-]+\.\w+', re.IGNORECASE)
PHONE_REGEX = re.compile(r'(?:\+?\d{1,4}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{3,6}\b')
CNIC_SSN_REGEX = re.compile(r'\b\d{5}-\d{7}-\d\b|\b\d{13}\b|\b\d{3}-\d{2}-\d{4}\b')
JWT_REGEX = re.compile(r'eyJ[a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+')
GROQ_KEY_REGEX = re.compile(r'gsk_[a-zA-Z0-9_]{10,}')



def mask_pii_and_secrets(text: Any) -> Any:
    """
    Sanitize text by redacting sensitive HR PII (emails, phone numbers, CNIC/national IDs)
    and credentials (JWTs, API keys, passwords) before storing in traces or sending to LangSmith.
    """
    if not isinstance(text, str):
        return text

    if not settings.TRACING_MASK_PII:
        return text

    sanitized = JWT_REGEX.sub("[JWT_REDACTED]", text)
    sanitized = GROQ_KEY_REGEX.sub("[API_KEY_REDACTED]", sanitized)
    sanitized = EMAIL_REGEX.sub("[EMAIL_REDACTED]", sanitized)
    sanitized = CNIC_SSN_REGEX.sub("[ID_REDACTED]", sanitized)
    sanitized = PHONE_REGEX.sub("[PHONE_REDACTED]", sanitized)
    return sanitized


def sanitize_dict(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Recursively mask PII in dictionary payloads."""
    if not data:
        return {}
    clean = {}
    for k, v in data.items():
        # Never trace raw attachments, api keys, or passwords
        if k in ("password", "raw_token", "attachment_file", "attachment_bytes", "api_key", "groq_key"):
            clean[k] = "[REDACTED_SECRET]"
        elif k in ("attachment_text", "raw_attachment") and not settings.TRACING_INCLUDE_RAW_ATTACHMENTS:
            clean[k] = f"[ATTACHMENT_OMITTED_FOR_PRIVACY (length: {len(str(v)) if v else 0})]"
        elif isinstance(v, str):
            clean[k] = mask_pii_and_secrets(v)
        elif isinstance(v, dict):
            clean[k] = sanitize_dict(v)
        elif isinstance(v, list):
            clean[k] = [
                sanitize_dict(item) if isinstance(item, dict)
                else (mask_pii_and_secrets(item) if isinstance(item, str) else item)
                for item in v
            ]
        else:
            clean[k] = v
    return clean


class TraceSpan(BaseModel):
    span_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    span_type: str = "tool"  # "tool", "chain", "llm", "guardrail"
    start_time: float
    end_time: Optional[float] = None
    duration_ms: float = 0.0
    status: str = "running"
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


class TraceRun(BaseModel):
    trace_id: str
    conversation_id: Optional[str] = None
    user_id: Optional[int] = None
    branch_id: Optional[int] = None
    department_id: Optional[int] = None
    start_time: float
    end_time: Optional[float] = None
    total_duration_ms: float = 0.0
    status: str = "running"
    spans: List[TraceSpan] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)


class TraceContext:
    """
    Manages end-to-end tracing for a single RAG chat request with hierarchical spans.
    Supports LangSmith hosted free-tier tracing with fail-open fallback to local store.
    """

    def __init__(
        self,
        conversation_id: Optional[str] = None,
        user_id: Optional[int] = None,
        branch_id: Optional[int] = None,
        department_id: Optional[int] = None,
        initial_inputs: Optional[Dict[str, Any]] = None,
    ):
        self.trace_id = str(uuid.uuid4())
        self.conversation_id = conversation_id
        self.user_id = user_id
        self.branch_id = branch_id
        self.department_id = department_id
        self.start_time = time.time()
        self.spans: List[TraceSpan] = []
        self.active_spans: Dict[str, TraceSpan] = {}
        self.inputs = sanitize_dict(initial_inputs or {})
        self.metadata = {
            "service": settings.SERVICE_NAME,
            "environment": "production" if not settings.DEBUG else "development",
            "langsmith_enabled": settings.LANGSMITH_TRACING,
        }

        # Initialize and register trace in local ring buffer
        self.run = TraceRun(
            trace_id=self.trace_id,
            conversation_id=conversation_id,
            user_id=user_id,
            branch_id=branch_id,
            department_id=department_id,
            start_time=self.start_time,
            inputs=self.inputs,
            metadata=self.metadata,
        )
        _trace_buffer.append(self.run)
        _trace_map[self.trace_id] = self.run

    def start_span(
        self,
        name: str,
        span_type: str = "tool",
        inputs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TraceSpan:
        """Create and start a new child span in the trace."""
        span = TraceSpan(
            name=name,
            span_type=span_type,
            start_time=time.time(),
            inputs=sanitize_dict(inputs or {}),
            metadata=sanitize_dict(metadata or {}),
        )
        self.spans.append(span)
        self.active_spans[name] = span
        return span

    def end_span(
        self,
        name: str,
        outputs: Optional[Dict[str, Any]] = None,
        status: str = "success",
        error: Optional[str] = None,
    ) -> Optional[TraceSpan]:
        """Complete a child span and record its duration."""
        span = self.active_spans.pop(name, None)
        if not span:
            # If not in active map, find in list
            for s in reversed(self.spans):
                if s.name == name and s.end_time is None:
                    span = s
                    break

        if span:
            span.end_time = time.time()
            span.duration_ms = round((span.end_time - span.start_time) * 1000, 2)
            span.status = status
            span.outputs = sanitize_dict(outputs or {})
            if error:
                span.error = mask_pii_and_secrets(str(error))
        return span

    def complete_trace(
        self,
        outputs: Optional[Dict[str, Any]] = None,
        status: str = "success",
    ) -> None:
        """Finalize trace run and optionally send to LangSmith (Fail-Open)."""
        self.run.end_time = time.time()
        self.run.total_duration_ms = round((self.run.end_time - self.run.start_time) * 1000, 2)
        self.run.status = status
        self.run.spans = self.spans
        self.run.outputs = sanitize_dict(outputs or {})

        # LangSmith Cloud Export (Fail-Open)
        if settings.LANGSMITH_TRACING and settings.LANGSMITH_API_KEY:
            try:
                self._export_to_langsmith()
            except Exception as exc:
                logger.warning(f"LangSmith export fail-open (non-blocking): {exc}")

    def _export_to_langsmith(self) -> None:
        """Export trace and child runs to LangSmith API."""
        try:
            from langsmith import Client

            client = Client(
                api_key=settings.LANGSMITH_API_KEY,
                api_url=settings.LANGSMITH_ENDPOINT,
            )

            # Create Root Run
            root_run_id = uuid.UUID(self.trace_id)
            client.create_run(
                name="rag_chat_request",
                run_type="chain",
                inputs=self.inputs,
                outputs=self.run.outputs,
                id=root_run_id,
                project_name=settings.LANGSMITH_PROJECT,
                start_time=self.start_time,
                end_time=self.run.end_time,
                extra={"metadata": self.metadata},
            )

            # Create Child Spans
            for span in self.spans:
                span_id = uuid.UUID(span.span_id)
                client.create_run(
                    name=span.name,
                    run_type=span.span_type,
                    inputs=span.inputs,
                    outputs=span.outputs,
                    id=span_id,
                    parent_run_id=root_run_id,
                    project_name=settings.LANGSMITH_PROJECT,
                    start_time=span.start_time,
                    end_time=span.end_time or span.start_time,
                    extra={"metadata": span.metadata},
                    error=span.error,
                )
            logger.info(f"Exported trace '{self.trace_id}' with {len(self.spans)} spans to LangSmith.")
        except Exception as exc:
            logger.warning(f"Failed to submit trace to LangSmith (continuing normally): {exc}")


def get_recent_traces(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieve recent traces from the local in-memory registry."""
    traces = list(_trace_buffer)
    traces.reverse()
    return [t.model_dump() for t in traces[:limit]]


def get_trace_by_id(trace_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve full trace details for a specific trace ID."""
    trace = _trace_map.get(trace_id)
    return trace.model_dump() if trace else None


def clear_traces() -> None:
    """Clear in-memory traces (used for clean testing)."""
    _trace_buffer.clear()
    _trace_map.clear()
