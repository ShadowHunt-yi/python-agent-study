"""Pydantic 数据模型"""
from enum import Enum
from pydantic import BaseModel, Field


# ===== 请求/响应模型 =====

class ChatRequest(BaseModel):
    """聊天请求"""
    message: str = Field(min_length=1, max_length=10000, description="用户消息")
    thread_id: str | None = Field(default=None, description="对话线程ID（可选）")


class ChatResponse(BaseModel):
    """聊天响应"""
    reply: str = Field(description="Agent 回复")
    tool_calls: list[dict] = Field(default_factory=list, description="工具调用记录")
    steps: int = Field(default=0, description="执行步数")


class StreamEvent(BaseModel):
    """流式事件"""
    type: str = Field(description="事件类型: thinking/tool_call/tool_result/token/done/error")
    content: str = Field(default="", description="事件内容")
    name: str = Field(default="", description="工具名称（tool_call 时有值）")


# ===== 工具相关模型 =====

class ToolCall(BaseModel):
    """工具调用"""
    id: str = Field(description="调用ID")
    name: str = Field(description="工具名称")
    args: dict = Field(description="工具参数")


class ToolResult(BaseModel):
    """工具执行结果"""
    tool_call_id: str = Field(description="对应的调用ID")
    name: str = Field(description="工具名称")
    content: str = Field(description="执行结果")


# ===== 对话历史模型 =====

class Role(str, Enum):
    """消息角色"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class Message(BaseModel):
    """对话消息"""
    role: Role
    content: str
    tool_call_id: str | None = None
