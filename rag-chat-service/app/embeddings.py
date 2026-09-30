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


def _format_input(text: str, is_query: bool = True) -> str:
    """Format input with appropriate prefix for E5 / asymmetric embedding models."""
    model_name = settings.EMBEDDING_MODEL_NAME.lower()
    if "e5" in model_name:
        prefix = "query: " if is_query else "passage: "
        if not text.startswith("query: ") and not text.startswith("passage: "):
            return f"{prefix}{text}"
    return text


def generate_embedding(text: str, is_query: bool = True) -> List[float]:
    """Generate normalized 768-dimensional vector embedding for query/text."""
    model = get_embedding_model()
    formatted = _format_input(text, is_query=is_query)
    embedding = model.encode(formatted, convert_to_numpy=True, normalize_embeddings=True)
    return embedding.tolist()
