from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.database import Base, SessionLocal, engine
from app.routers import auth, identity, org, rbac, users
from app.seed import seed_database


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize DB schema
    Base.metadata.create_all(bind=engine)
    # Seed initial roles, permissions and admin user
    db = SessionLocal()
    try:
        seed_database(db)
    finally:
        db.close()
    yield


app = FastAPI(
    title="HR RAG Assistant - Identity & Organization Service",
    description="Microservice providing Authentication, RBAC, User Management, and Organization Structure",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS for frontend and inter-service communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers with /api/v1 prefix
app.include_router(auth.router)
app.include_router(identity.router)
app.include_router(rbac.router)
app.include_router(users.router)
app.include_router(org.router)


@app.get("/health", tags=["Health"])
def health_check():
    """Service health check endpoint."""
    return {"status": "healthy", "service": "identity-org-service", "version": "1.0.0"}
