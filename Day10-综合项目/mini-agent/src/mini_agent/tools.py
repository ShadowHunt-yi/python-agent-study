"""工具定义

每个工具用 @tool 装饰器定义，docstring 就是 LLM 看到的工具描述。
写清楚 docstring 非常重要，LLM 靠它决定调用哪个工具。
"""
import math
from datetime import datetime

from pydantic import BaseModel, Field


# ===== 工具参数 Schema（Pydantic） =====

class CalculateInput(BaseModel):
    """计算数学表达式的参数"""
    expression: str = Field(description="数学表达式，如 '2+3*4' 或 'math.sqrt(16)'")


class SearchInput(BaseModel):
    """搜索知识库的参数"""
    query: str = Field(description="搜索关键词")


# ===== 工具实现 =====

def calculate(expression: str) -> str:
    """计算数学表达式。支持基本运算（+, -, *, /, **）和 math 模块函数。

    使用示例:
    - "2 + 3 * 4"
    - "math.sqrt(144)"
    - "math.log10(1000)"
    """
    try:
        # 只允许安全的内置函数和 math 模块
        allowed_names = {
            "math": math,
            "abs": abs,
            "round": round,
            "int": int,
            "float": float,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
        }
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"


def search_knowledge(query: str) -> str:
    """搜索知识库，返回与查询相关的知识信息。

    可以搜索的主题包括：Python、AI、Agent、LangChain、FastAPI 等。
    """
    # Mock 知识库
    knowledge_base = {
        "Python": (
            "Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年创建。"
            "Python 以其简洁的语法和丰富的生态系统著称，广泛应用于 Web 开发、"
            "数据科学、AI/ML 和自动化脚本。"
        ),
        "AI": (
            "人工智能(AI)是计算机科学的一个分支，致力于创建能模拟人类智能的系统。"
            "现代 AI 主要基于深度学习和大语言模型(LLM)技术。"
        ),
        "Agent": (
            "AI Agent 是能自主执行任务的智能体，通常结合 LLM、工具调用和记忆系统。"
            "Agent 可以感知环境、制定计划、执行动作并根据结果调整策略。"
            "主流框架包括 LangChain、LangGraph、CrewAI 等。"
        ),
        "LangChain": (
            "LangChain 是一个用于构建 LLM 应用的 Python 框架。"
            "核心概念包括：@tool 装饰器、LCEL 管道、Agent 创建等。"
            "LangGraph 是其图编排扩展，用于构建复杂的 Agent 工作流。"
        ),
        "FastAPI": (
            "FastAPI 是一个现代 Python Web 框架，基于类型提示自动生成 API 文档。"
            "特点：高性能（基于 Starlette）、自动校验（基于 Pydantic）、"
            "依赖注入、原生异步支持。"
        ),
    }

    results = []
    for key, value in knowledge_base.items():
        if key.lower() in query.lower():
            results.append(f"【{key}】{value}")

    if results:
        return "\n\n".join(results)
    return f"未找到关于 '{query}' 的相关信息。可搜索的主题：Python、AI、Agent、LangChain、FastAPI"


def get_current_time() -> str:
    """获取当前的日期和时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S (%A)")


# ===== 工具注册表 =====

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": calculate.__doc__,
            "parameters": CalculateInput.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": search_knowledge.__doc__,
            "parameters": SearchInput.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": get_current_time.__doc__,
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

# 工具名称 → 实现函数的映射
TOOL_FUNCTIONS = {
    "calculate": lambda args: calculate(args["expression"]),
    "search_knowledge": lambda args: search_knowledge(args["query"]),
    "get_current_time": lambda args: get_current_time(),
}
