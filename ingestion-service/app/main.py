from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import ingestion
from app.vector_store import get_vector_collection


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Chroma collection on boot
    get_vector_collection()
    yield


app = FastAPI(
    title="HR RAG Assistant - Ingestion Service",
    description="Microservice responsible for Document Ingestion, Text Chunking, Local Vector Embeddings, and ChromaDB Storage",
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

app.include_router(ingestion.router)


@app.get("/health", tags=["Health"])
def health_check():
    """Service health check endpoint."""
    return {"status": "healthy", "service": "ingestion-service", "version": "1.0.0"}
