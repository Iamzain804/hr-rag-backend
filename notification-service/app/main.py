from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import notify

app = FastAPI(
    title="HR RAG Assistant - Notification Service",
    description="Microservice responsible for dispatching email and transactional notifications",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(notify.router)


@app.get("/health", tags=["Health"])
def health_check():
    """Service health check endpoint."""
    return {"status": "healthy", "service": "notification-service", "version": "1.0.0"}
