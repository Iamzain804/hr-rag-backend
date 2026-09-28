import os
os.environ["LITELLM_TELEMETRY"] = "False"
import logging
from typing import AsyncGenerator, Dict, List, Optional
import litellm
litellm.telemetry = False
litellm.suppress_debug_info = True
litellm.drop_params = True
from litellm import Router

from app.config import settings

logger = logging.getLogger("rag_chat_service.llm")

_router: Optional[Router] = None
_active_key_index: int = 0


def get_active_groq_key_index() -> int:
    """Return the integer index of the active Groq key (0-10) for privacy-safe tracing."""
    return _active_key_index


def get_llm_router() -> Router:
    """
    Initialize and return LiteLLM Router configured with multiple Groq API keys
    and automatic failover on rate-limiting (429) or transient errors.
    """
    global _router
    if _router is None:
        model_list = []
        for idx, key in enumerate(settings.GROQ_API_KEYS):
            # Register primary model for each API key
            model_list.append(
                {
                    "model_name": "hr-assistant-primary",
                    "litellm_params": {
                        "model": settings.DEFAULT_MODEL,
                        "api_key": key,
                    },
                    "model_info": {"key_index": idx},
                }
            )
            # Register fallback fast model for each API key
            model_list.append(
                {
                    "model_name": "hr-assistant-fallback",
                    "litellm_params": {
                        "model": settings.FALLBACK_MODEL,
                        "api_key": key,
                    },
                    "model_info": {"key_index": idx},
                }
            )

        _router = Router(
            model_list=model_list,
            routing_strategy="simple-shuffle",
            num_retries=len(settings.GROQ_API_KEYS) + 2,
            retry_after=1,
            timeout=15,
            allowed_fails=3,
        )
        logger.info(f"Initialized LiteLLM Router with {len(settings.GROQ_API_KEYS)} Groq keys.")
    return _router



async def _stream_from_model(
    router: Router,
    model_alias: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> AsyncGenerator[str, None]:
    """Helper to stream chunks from a specific router model."""
    response = await router.acompletion(
        model=model_alias,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    async for chunk in response:
        if hasattr(chunk, "choices") and chunk.choices:
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                yield content


async def stream_llm_completion(
    messages: List[Dict[str, str]],
    model_alias: str = "hr-assistant-primary",
    temperature: float = 0.2,
    max_tokens: int = 800,
) -> AsyncGenerator[str, None]:
    """
    Stream completion tokens using LiteLLM Router with automatic key rotation.
    """
    router = get_llm_router()
    try:
        async for token in _stream_from_model(
            router, model_alias, messages, temperature, max_tokens
        ):
            yield token
    except Exception as exc:
        logger.warning(f"Primary model stream error: {exc}. Attempting fallback model...")
        if model_alias != "hr-assistant-fallback":
            try:
                async for token in _stream_from_model(
                    router, "hr-assistant-fallback", messages, temperature, max_tokens
                ):
                    yield token
            except Exception as inner_exc:
                logger.error(f"Fallback model also failed: {inner_exc}")
                yield f"\n\n[Error: Unable to connect to LLM provider: {str(inner_exc)}]"
        else:
            yield f"\n\n[Error: Unable to connect to LLM provider: {str(exc)}]"
