"""API 路由定义"""
import json

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from ..models import ChatRequest, ChatResponse, StreamEvent
from ..agent import Agent

router = APIRouter(prefix="/agent", tags=["Agent"])

# 全局 Agent 实例
agent = Agent()


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """同步聊天接口

    等待 Agent 完成所有步骤后返回最终结果。
    """
    result = await agent.run(req.message)
    return ChatResponse(**result)


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """流式聊天接口（SSE）

    返回 Server-Sent Events 流，前端可用 EventSource 接收。
    """
    async def event_generator():
        async for event in agent.stream(req.message):
            yield f"data: {event.model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
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
