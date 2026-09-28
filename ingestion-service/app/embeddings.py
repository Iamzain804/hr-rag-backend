import logging
from typing import List
from sentence_transformers import SentenceTransformer
from app.config import settings

logger = logging.getLogger("ingestion_service")

_model: SentenceTransformer = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy-load the local SentenceTransformer model singleton."""
    global _model
    if _model is None:
        logger.info(f"Loading local embedding model: {settings.EMBEDDING_MODEL_NAME}...")
        _model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        logger.info("Local embedding model loaded successfully.")
    return _model


def generate_embedding(text: str) -> List[float]:
    """Generate local 384-dimensional vector embedding for a single text chunk."""
    model = get_embedding_model()
    embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return embedding.tolist()


def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Generate local vector embeddings for a batch of text chunks."""
    if not texts:
        return []
    model = get_embedding_model()
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.tolist()
