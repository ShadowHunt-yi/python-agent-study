"""API 接口测试"""
import pytest
from unittest.mock import AsyncMock, patch
from httpx import AsyncClient, ASGITransport


@pytest.fixture
async def api_client():
    from mini_agent.main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.mark.asyncio
async def test_health(api_client):
    resp = await api_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_chat_sync(api_client):
    with patch("mini_agent.agent.Agent.run", new_callable=AsyncMock) as mock:
        mock.return_value = {"reply": "测试回复", "tool_calls": [], "steps": 1}
        resp = await api_client.post("/agent/chat", json={"message": "测试"})
        assert resp.status_code == 200
        assert resp.json()["reply"] == "测试回复"


@pytest.mark.asyncio
async def test_chat_validation(api_client):
    """空消息应返回 422"""
    resp = await api_client.post("/agent/chat", json={"message": ""})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_list_tools(api_client):
    resp = await api_client.get("/agent/tools")
    assert resp.status_code == 200
    tools = resp.json()["tools"]
    assert len(tools) == 3
    assert tools[0]["name"] == "calculate"
