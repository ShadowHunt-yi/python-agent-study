# 02 — Agent 核心逻辑

> 本项目使用 OpenAI SDK 直接调用 LLM，不依赖 LangChain。
> 这样可以更清楚地理解 Agent 的底层原理。

## agent.py —— ReAct 循环核心

```python
"""Agent 核心逻辑 —— ReAct 循环实现

ReAct 循环：
1. LLM 思考 → 决定是否调用工具
2. 如果调用工具 → 执行工具 → 将结果反馈给 LLM → 回到 1
3. 如果不调用工具 → 输出最终回复
"""
import json
import logging
from typing import AsyncGenerator

from openai import AsyncOpenAI

from .config import settings
from .models import StreamEvent
from .tools import TOOLS, TOOL_FUNCTIONS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """你是一个有帮助的 AI 助手。你可以使用以下工具来回答问题：

1. calculate - 计算数学表达式（支持 +, -, *, /, **, math 函数）
2. search_knowledge - 搜索知识库获取信息
3. get_current_time - 获取当前日期和时间

请根据用户的问题，决定是否需要调用工具。
- 如果需要工具，直接调用即可
- 如果不需要工具，直接用你的知识回答
- 回答要简洁、准确
"""


class Agent:
    """ReAct Agent

    实现完整的 ReAct 循环：
    - 支持多轮工具调用
    - 支持同步和流式输出
    - 自动管理对话历史
    """

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self.model = settings.openai_model
        self.tools = TOOLS                # OpenAI 格式的 tools schema
        self.tool_functions = TOOL_FUNCTIONS  # 工具名 → 执行函数
        self.max_steps = settings.max_agent_steps

    def _build_messages(self, history: list[dict]) -> list[dict]:
        """构建 OpenAI 格式的消息列表"""
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        return messages

    def _execute_tool(self, tool_call: dict) -> str:
        """执行单个工具调用

        Args:
            tool_call: OpenAI 格式的 tool_call 对象
        """
        name = tool_call["function"]["name"]
        args_str = tool_call["function"]["arguments"]

        logger.info(f"调用工具: {name}({args_str})")

        func = self.tool_functions.get(name)
        if not func:
            return f"错误：未知工具 '{name}'"

        try:
            args = json.loads(args_str)
            result = func(args)
            logger.info(f"工具结果: {result[:200]}")
            return str(result)
        except Exception as e:
            error_msg = f"工具执行错误: {e}"
            logger.error(error_msg)
            return error_msg

    async def run(self, query: str, history: list[dict] | None = None) -> dict:
        """同步运行 Agent（非流式）

        Args:
            query: 用户输入
            history: 对话历史（可选）

        Returns:
            包含 reply, tool_calls, steps 的字典
        """
        history = list(history or [])
        history.append({"role": "user", "content": query})

        all_tool_calls = []
        steps = 0

        for step in range(self.max_steps):
            steps += 1
            messages = self._build_messages(history)

            # 调用 LLM
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
            )

            message = response.choices[0].message

            # 没有工具调用 → 最终回复
            if not message.tool_calls:
                reply = message.content or ""
                history.append({"role": "assistant", "content": reply})
                return {
                    "reply": reply,
                    "tool_calls": all_tool_calls,
                    "steps": steps,
                }

            # 有工具调用 → 逐个执行
            # 先将 assistant 消息（含 tool_calls）加入历史
            assistant_msg = {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            }
            history.append(assistant_msg)

            # 执行每个工具调用
            for tc in message.tool_calls:
                all_tool_calls.append({
                    "name": tc.function.name,
                    "args": json.loads(tc.function.arguments),
                })

                result = self._execute_tool({
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                })

                # 将工具结果作为 tool message 加入历史
                history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        # 达到最大步数
        return {
            "reply": "抱歉，达到最大思考步数限制，未能完成任务。",
            "tool_calls": all_tool_calls,
            "steps": steps,
        }

    async def stream(
        self, query: str, history: list[dict] | None = None
    ) -> AsyncGenerator[StreamEvent, None]:
        """流式运行 Agent

        Yields:
            StreamEvent 事件流
        """
        history = list(history or [])
        history.append({"role": "user", "content": query})

        for step in range(self.max_steps):
            messages = self._build_messages(history)

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
            )

            message = response.choices[0].message

            # 没有工具调用 → 流式输出最终回复
            if not message.tool_calls:
                reply = message.content or ""
                for char in reply:
                    yield StreamEvent(type="token", content=char)
                history.append({"role": "assistant", "content": reply})
                yield StreamEvent(type="done")
                return

            # 有工具调用 → 执行工具
            assistant_msg = {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            }
            history.append(assistant_msg)

            for tc in message.tool_calls:
                tool_name = tc.function.name
                tool_args = tc.function.arguments

                # 推送工具调用事件
                yield StreamEvent(
                    type="tool_call",
                    name=tool_name,
                    content=tool_args,
                )

                # 执行工具
                result = self._execute_tool({
                    "function": {
                        "name": tool_name,
                        "arguments": tool_args,
                    }
                })

                # 推送工具结果事件
                yield StreamEvent(
                    type="tool_result",
                    name=tool_name,
                    content=result,
                )

                history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        yield StreamEvent(
            type="error",
            content="达到最大步数限制，未能完成任务。",
        )
```

## api/routes.py

```python
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..models import ChatRequest, ChatResponse
from ..agent import Agent

router = APIRouter(prefix="/agent", tags=["Agent"])
agent = Agent()

@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """同步聊天"""
    result = await agent.run(req.message)
    return ChatResponse(**result)

@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """流式聊天（SSE）"""
    async def event_generator():
        async for event in agent.stream(req.message):
            yield f"data: {event.model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
    )

@router.get("/tools")
async def list_tools():
    """列出所有可用工具"""
    from ..tools import TOOLS
    return {
        "tools": [
            {
                "name": t["function"]["name"],
                "description": t["function"]["description"],
            }
            for t in TOOLS
        ]
    }
```

## main.py

```python
import logging
from fastapi import FastAPI
from .config import settings
from .api.routes import router

logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

app = FastAPI(title="Mini Agent", version="0.1.0")
app.include_router(router)

@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.openai_model}
```

## 核心流程图

```
用户输入 "2+3等于几？"
    ↓
Agent.run("2+3等于几？")
    ↓
LLM 调用（tools=TOOLS, tool_choice="auto"）
    ↓
LLM 返回 tool_calls: [{name: "calculate", args: {expression: "2+3"}}]
    ↓
执行 calculate("2+3") → "5"
    ↓
将工具结果加入历史，再次调用 LLM
    ↓
LLM 返回 content: "2+3 等于 5"（无 tool_calls）
    ↓
返回 {"reply": "2+3 等于 5", "tool_calls": [...], "steps": 2}
```
