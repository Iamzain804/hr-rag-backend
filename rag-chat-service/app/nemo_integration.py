import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional
from nemoguardrails import RailsConfig, LLMRails

logger = logging.getLogger("rag_chat_service.nemo")

_rails_instance: Optional[LLMRails] = None

def get_nemo_rails() -> LLMRails:
    """
    Initialize or return the singleton NeMo Guardrails LLMRails instance.
    """
    global _rails_instance
    if _rails_instance is None:
        config_path = Path(__file__).parent / "guardrails_config"
        config = RailsConfig.from_path(str(config_path))
        _rails_instance = LLMRails(config)
        logger.info("NeMo Guardrails LLMRails initialized successfully.")
    return _rails_instance

async def evaluate_nemo_guardrails(user_prompt: str) -> Dict[str, Any]:
    """
    Evaluate user prompt against NeMo Guardrails rules.
    Returns dictionary with:
      - is_blocked: bool
      - response: str (if blocked or processed)
    """
    try:
        rails = get_nemo_rails()
        res = await rails.generate_async(prompt=user_prompt)
        # Check if response matches known canned refusal
        if "I am the HR AI Assistant. I can only assist with company HR policies" in res.response[0]["content"] or \
           "I cannot process instructions" in res.response[0]["content"]:
            return {
                "is_blocked": True,
                "response": res.response[0]["content"],
            }
        return {
            "is_blocked": False,
            "response": res.response[0]["content"] if res.response else "",
        }
    except Exception as e:
        logger.warning(f"NeMo Guardrails evaluation error: {e}. Falling back to default pipeline.")
        return {
            "is_blocked": False,
            "response": "",
        }
