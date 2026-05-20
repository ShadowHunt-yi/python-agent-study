"""Agent 核心逻辑测试"""
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from mini_agent.agent import Agent


def make_mock_response(content=None, tool_calls=None):
    """创建 Mock 的 OpenAI 响应"""
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls or []
    response = MagicMock()
    response.choices = [MagicMock(message=message)]
    return response


@pytest.fixture
def agent():
    """创建 Agent 实例（Mock 客户端）"""
    with patch("mini_agent.agent.AsyncOpenAI") as MockClient:
        a = Agent()
        a.client = AsyncMock()
        yield a


class TestAgentSync:
    """Agent 同步模式测试"""

    @pytest.mark.asyncio
    async def test_direct_answer(self, agent):
        """测试直接回复（不调用工具）"""
        agent.client.chat.completions.create = AsyncMock(
            return_value=make_mock_response(content="你好！有什么可以帮你的？")
        )

        result = await agent.run("你好")
        assert result["reply"] == "你好！有什么可以帮你的？"
        assert result["steps"] == 1
        assert result["tool_calls"] == []

    @pytest.mark.asyncio
    async def test_single_tool_call(self, agent):
        """测试单次工具调用"""
        # 第一次返回工具调用，第二次返回最终回复
        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "calculate"
        tool_call.function.arguments = json.dumps({"expression": "2+3"})

        agent.client.chat.completions.create = AsyncMock(side_effect=[
            make_mock_response(content="", tool_calls=[tool_call]),
            make_mock_response(content="2+3 等于 5"),
        ])

        result = await agent.run("2+3等于几？")
        assert "5" in result["reply"]
        assert len(result["tool_calls"]) == 1
        assert result["tool_calls"][0]["name"] == "calculate"
        assert result["steps"] == 2

    @pytest.mark.asyncio
    async def test_max_steps_limit(self, agent):
        """测试最大步数限制"""
        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "get_current_time"
        tool_call.function.arguments = "{}"

        # 每次都返回工具调用，模拟无限循环
        agent.client.chat.completions.create = AsyncMock(
            return_value=make_mock_response(content="", tool_calls=[tool_call])
        )
        agent.max_steps = 3

        result = await agent.run("测试")
        assert result["steps"] == 3
        assert "限制" in result["reply"]


class TestAgentStream:
    """Agent 流式模式测试"""

    @pytest.mark.asyncio
    async def test_stream_tokens(self, agent):
        """测试流式输出"""
        agent.client.chat.completions.create = AsyncMock(
            return_value=make_mock_response(content="Hello World")
        )

        events = []
        async for event in agent.stream("Hi"):
            events.append(event)

        # 应该有 token 事件和 done 事件
        token_events = [e for e in events if e.type == "token"]
        done_events = [e for e in events if e.type == "done"]

        assert len(token_events) == len("Hello World")
        assert len(done_events) == 1

    @pytest.mark.asyncio
    async def test_stream_with_tool(self, agent):
        """测试流式输出带工具调用"""
        tool_call = MagicMock()
        tool_call.id = "call_1"
        tool_call.function.name = "get_current_time"
        tool_call.function.arguments = "{}"

        agent.client.chat.completions.create = AsyncMock(side_effect=[
            make_mock_response(content="", tool_calls=[tool_call]),
            make_mock_response(content="现在是 2026-05-20"),
        ])

        events = []
        async for event in agent.stream("几点了？"):
            events.append(event)

        event_types = [e.type for e in events]
        assert "tool_call" in event_types
        assert "tool_result" in event_types
        assert "token" in event_types
        assert "done" in event_types
