import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer
from app.config import settings

logger = logging.getLogger("rag_chat_service.embeddings")

_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Lazy-load the local SentenceTransformer model singleton."""
    global _model
    if _model is None:
        logger.info(f"Loading local embedding model: {settings.EMBEDDING_MODEL_NAME}...")
        _model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        logger.info("Local embedding model loaded successfully.")
    return _model


def generate_embedding(text: str) -> List[float]:
    """Generate normalized 384-dimensional vector embedding for query/text."""
    model = get_embedding_model()
    embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return embedding.tolist()
