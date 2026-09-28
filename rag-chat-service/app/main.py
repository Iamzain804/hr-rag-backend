from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.llm_router import get_llm_router
from app.routers import chat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("rag_chat_service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize LiteLLM Router with Groq key pool
    logger.info("Initializing LiteLLM router on startup...")
    get_llm_router()
    yield
    logger.info("Shutting down rag-chat-service...")


app = FastAPI(
    title="HR RAG Assistant - RAG Chat Service",
    description="Microservice responsible for Branch/Department Contextual Retrieval, Guardrails, OCR Attachments, and LLM Generation via LiteLLM/Groq",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)


@app.get("/health", tags=["Health"])
def health_check():
    """Service health check endpoint."""
    return {
        "status": "healthy",
        "service": settings.SERVICE_NAME,
        "port": settings.PORT,
        "version": "1.0.0",
        "groq_keys_count": len(settings.GROQ_API_KEYS),
    }
