# 03 — Agent 测试实战

## 一、测试策略分层

```
单元测试（快，多）→ 工具函数、Pydantic 模型、工具选择逻辑
集成测试（中）→ Agent + Mock LLM
端到端测试（慢，少）→ 真实 LLM 调用
```

## 二、Mock OpenAI 客户端

```python
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

def make_completion(content: str = "", tool_calls: list = None):
    """创建 Mock 的 ChatCompletion"""
    message = ChatCompletionMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
    )
    return ChatCompletion(
        id="chatcmpl-test",
        choices=[Choice(index=0, message=message, finish_reason="stop")],
        created=1234567890,
        model="gpt-4o",
        object="chat.completion",
    )

@pytest.fixture
def mock_openai():
    """Mock OpenAI 异步客户端"""
    with patch("your_module.AsyncOpenAI") as MockClient:
        client = AsyncMock()
        MockClient.return_value = client
        yield client
```

## 三、测试工具选择逻辑

```python
@pytest.mark.asyncio
async def test_agent_chooses_calculator(mock_openai):
    """测试 Agent 对数学问题选择计算器工具"""
    from openai.types.chat import ChatCompletionMessageToolCall

    tool_call = ChatCompletionMessageToolCall(
        id="call_1",
        type="function",
        function={"name": "calculate", "arguments": '{"expression": "2+3"}'},
    )

    mock_openai.chat.completions.create = AsyncMock(side_effect=[
        make_completion(tool_calls=[tool_call]),
        make_completion(content="2+3 等于 5"),
    ])

    agent = Agent()
    result = await agent.run("2+3等于几？")

    assert "5" in result["reply"]
    assert result["tool_calls"][0]["name"] == "calculate"
```

## 四、测试错误处理

```python
@pytest.mark.asyncio
async def test_agent_handles_tool_error(mock_openai):
    """测试工具执行失败时的处理"""
    from openai.types.chat import ChatCompletionMessageToolCall

    tool_call = ChatCompletionMessageToolCall(
        id="call_1",
        type="function",
        function={"name": "calculate", "arguments": '{"expression": "1/0"}'},
    )

    mock_openai.chat.completions.create = AsyncMock(side_effect=[
        make_completion(tool_calls=[tool_call]),
        make_completion(content="计算出错，除数不能为零"),
    ])

    agent = Agent()
    result = await agent.run("1除以0")

    # Agent 应该能处理错误并给出合理回复
    assert result["steps"] == 2
```

## 五、测试流式输出

```python
@pytest.mark.asyncio
async def test_stream_output(mock_openai):
    """测试流式输出的事件序列"""
    from openai.types.chat import ChatCompletionMessageToolCall

    mock_openai.chat.completions.create = AsyncMock(
        return_value=make_completion(content="Hello")
    )

    agent = Agent()
    events = []
    async for event in agent.stream("Hi"):
        events.append(event)

    # 验证事件序列
    assert events[-1].type == "done"
    assert all(e.type == "token" for e in events[:-1])
    assert "".join(e.content for e in events[:-1]) == "Hello"
```

## 六、集成测试：真实 LLM

```python
import os
import pytest

@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="需要 OPENAI_API_KEY",
)
@pytest.mark.asyncio
async def test_agent_e2e():
    """端到端测试（真实 LLM 调用）"""
    agent = Agent()
    result = await agent.run("2+3等于几？")

    assert "5" in result["reply"]
    assert result["steps"] >= 1
    assert result["steps"] <= 5
```

## 七、FastAPI 测试

```python
import pytest
from httpx import AsyncClient, ASGITransport
from unittest.mock import AsyncMock, patch

@pytest.fixture
def api_client():
    from mini_agent.main import app
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")

@pytest.mark.asyncio
async def test_chat_endpoint(api_client):
    with patch("mini_agent.agent.Agent.run", new_callable=AsyncMock) as mock:
        mock.return_value = {"reply": "test", "tool_calls": [], "steps": 1}
        resp = await api_client.post("/agent/chat", json={"message": "hello"})
        assert resp.status_code == 200
        assert resp.json()["reply"] == "test"

@pytest.mark.asyncio
async def test_chat_validation(api_client):
    """测试请求校验"""
    resp = await api_client.post("/agent/chat", json={"message": ""})
    assert resp.status_code == 422  # Pydantic 验证失败

@pytest.mark.asyncio
async def test_health(api_client):
    resp = await api_client.get("/health")
    assert resp.status_code == 200
```

## 八、conftest.py 最佳实践

```python
# tests/conftest.py
import pytest
from unittest.mock import AsyncMock, patch

@pytest.fixture(autouse=True)
def mock_env_vars():
    """自动 Mock 环境变量"""
    with patch.dict("os.environ", {
        "OPENAI_API_KEY": "test-key",
        "OPENAI_MODEL": "gpt-4o",
    }):
        yield

@pytest.fixture
def mock_openai():
    """Mock OpenAI 客户端"""
    with patch("your_module.AsyncOpenAI") as Mock:
        client = AsyncMock()
        Mock.return_value = client
        yield client

@pytest.fixture
def sample_chat_request():
    """示例请求数据"""
    return {"message": "你好"}
```

## 九、面试高频问题

**Q: 怎么测试 Agent 的工具选择？**
A: Mock LLM 返回特定的 tool_calls，验证 Agent 正确执行了工具并处理了结果。

**Q: 什么时候用真实 LLM 测试？**
A: 只在端到端测试中使用，且要标记为可跳过（需要 API Key）。大多数测试应该 Mock。

**Q: 怎么测试流式输出？**
A: 收集所有事件到列表，验证事件类型序列和内容。

**Q: conftest.py 放在哪里？**
A: 项目根目录或 tests 目录。同级及子目录的测试自动发现其中的 fixture。
