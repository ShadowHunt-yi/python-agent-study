"""测试配置和共享 fixture"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture(autouse=True)
def mock_settings():
    """自动 Mock 设置，避免需要真实的 API Key"""
    with patch("mini_agent.config.settings") as mock:
        mock.openai_api_key = "test-key"
        mock.openai_model = "gpt-4o"
        mock.openai_base_url = "http://localhost:8001"
        mock.max_agent_steps = 5
        mock.log_level = "DEBUG"
        yield mock


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI 异步客户端"""
    client = AsyncMock()
    return client
