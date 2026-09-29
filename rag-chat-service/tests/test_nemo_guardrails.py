import pytest
import os
from pathlib import Path
from nemoguardrails import RailsConfig, LLMRails
from app.config import settings
from app.nemo_integration import get_nemo_rails

@pytest.mark.asyncio
async def test_nemo_guardrails_initialization():
    config_path = Path(__file__).parent.parent / "app" / "guardrails_config"
    config = RailsConfig.from_path(str(config_path))
    assert config is not None
    rails = LLMRails(config)
    assert rails is not None

@pytest.mark.asyncio
async def test_nemo_guardrails_singleton_helper():
    rails = get_nemo_rails()
    assert rails is not None

@pytest.mark.asyncio
async def test_nemo_guardrails_config_loaded():
    config_path = Path(__file__).parent.parent / "app" / "guardrails_config"
    config = RailsConfig.from_path(str(config_path))
    assert len(config.models) > 0
    assert config.models[0].model == "llama-3.3-70b-versatile"
