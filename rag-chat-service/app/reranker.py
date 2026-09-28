import logging
import math
from typing import List, Optional
import numpy as np
from sentence_transformers.cross_encoder import CrossEncoder

from app.config import settings
from app.schemas import RetrievedChunk

logger = logging.getLogger("rag_chat_service.reranker")

_reranker_model: Optional[CrossEncoder] = None


def get_reranker_model() -> CrossEncoder:
    """Lazy-load the local CrossEncoder model singleton."""
    global _reranker_model
    if _reranker_model is None:
        logger.info(f"Loading local CrossEncoder model: {settings.RERANKER_MODEL_NAME}...")
        _reranker_model = CrossEncoder(settings.RERANKER_MODEL_NAME)
        logger.info("Local CrossEncoder reranker loaded successfully.")
    return _reranker_model


def normalize_rerank_score(raw_logit: float) -> float:
    """
    Calibrate CrossEncoder raw logits into a [0.0, 1.0] confidence score.
    MS MARCO models produce raw logits centered around -3.0 for matches and -11.0 for non-matches.
    Shift by +3.0 and scale by 2.0 to map:
    - strong match (>= 0.0) -> >= 0.82
    - good match (~ -2.5) -> ~ 0.56
    - weak/irrelevant (<= -6.0) -> <= 0.18
    - completely irrelevant (<= -10.0) -> <= 0.03
    """
    shifted = (float(raw_logit) + 3.0) / 2.0
    return round(float(1.0 / (1.0 + math.exp(-shifted))), 4)


def rerank_chunks(
    query: str,
    chunks: List[RetrievedChunk],
    top_k: Optional[int] = None,
) -> List[RetrievedChunk]:
    """
    Re-rank candidate chunks using deep cross-attention between the query and chunk texts.
    
    Returns chunks sorted in descending order of rerank_score, truncated to top_k.
    """
    if not chunks:
        return []

    clean_query = query.strip()
    if not clean_query:
        return chunks[:top_k] if top_k else chunks

    target_top_k = top_k if top_k is not None else settings.RERANK_TOP_K

    # If reranker is disabled in config, fallback to initial bi-encoder vector score
    if not settings.RERANKER_ENABLED:
        sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
        return sorted_chunks[:target_top_k]

    try:
        model = get_reranker_model()
        pairs = [[clean_query, chunk.content] for chunk in chunks]

        # Predict relevance scores (logits)
        raw_scores = model.predict(pairs, show_progress_bar=False)

        reranked_chunks: List[RetrievedChunk] = []
        for i, chunk in enumerate(chunks):
            raw_val = raw_scores[i]
            # If the model outputs multi-dimensional array or float
            if isinstance(raw_val, (list, np.ndarray)):
                raw_val = raw_val[0]

            # Calibrate logit to [0, 1] probability score
            norm_score = normalize_rerank_score(float(raw_val))

            # Create updated chunk with rerank_score
            chunk_copy = chunk.model_copy()
            chunk_copy.rerank_score = norm_score
            reranked_chunks.append(chunk_copy)


        # Sort descending by rerank_score
        reranked_chunks.sort(key=lambda c: c.rerank_score or 0.0, reverse=True)

        logger.info(
            f"Reranked {len(chunks)} chunks down to top-{target_top_k}. "
            f"Top rerank score: {reranked_chunks[0].rerank_score if reranked_chunks else 0.0}"
        )

        return reranked_chunks[:target_top_k]

    except Exception as exc:
        logger.warning(f"Reranker execution failed: {exc}. Falling back to initial vector ranking.")
        sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)
        return sorted_chunks[:target_top_k]
