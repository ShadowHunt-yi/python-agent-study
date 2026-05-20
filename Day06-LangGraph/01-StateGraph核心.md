# 01 — LangGraph StateGraph 核心

## 一、StateGraph —— 核心抽象

LangGraph 把 Agent 工作流建模为有向图：节点是计算单元，边是状态流转。

```python
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]   # Reducer: 追加到列表
    llm_calls: int                             # 自定义计数器
```

### Reducer（归并器）

```python
# Annotated[list, operator.add] = 节点返回 {"messages": [new]} 时追加而非替换
# 默认行为 = 节点返回 {"key": value} 时直接替换
```

## 二、节点（Node）

```python
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, ToolMessage

model = init_chat_model("openai:gpt-4o", temperature=0)

@tool
def multiply(a: int, b: int) -> int:
    """两数相乘"""
    return a * b

tools = [multiply]
tools_by_name = {t.name: t for t in tools}
model_with_tools = model.bind_tools(tools)

def llm_call(state: AgentState):
    """LLM 节点：决定是否调用工具"""
    response = model_with_tools.invoke(
        [SystemMessage(content="你是数学助手。")] + state["messages"]
    )
    return {
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }

def tool_node(state: AgentState):
    """工具节点：执行工具调用"""
    results = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        results.append(ToolMessage(
            content=str(observation),
            tool_call_id=tool_call["id"],
        ))
    return {"messages": results}
```

## 三、边（Edge）

```python
from typing import Literal

def should_continue(state: AgentState) -> Literal["tool_node", "__end__"]:
    """条件路由：有工具调用 → tool_node，否则 → END"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return "__end__"

# 构建图
builder = StateGraph(AgentState)
builder.add_node("llm_call", llm_call)
builder.add_node("tool_node", tool_node)

builder.add_edge(START, "llm_call")                           # 起始 → LLM
builder.add_conditional_edges("llm_call", should_continue)    # LLM → 条件路由
builder.add_edge("tool_node", "llm_call")                     # 工具 → 回到 LLM

agent = builder.compile()
```

### 图结构

```
START → llm_call → (有工具调用?) → tool_node → llm_call → ...
                      ↓ (无)
                     END
```

## 四、执行

```python
result = agent.invoke({
    "messages": [{"role": "user", "content": "3乘以4等于几？"}]
})
print(result["messages"][-1].content)   # "12"
print(result["llm_calls"])              # 2（一次思考+调用，一次整合结果）
```

## 五、Checkpointing —— 状态持久化

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
agent = builder.compile(checkpointer=memory)

# 用 thread_id 维护对话上下文
config = {"configurable": {"thread_id": "conversation-1"}}

# 第一轮对话
result = agent.invoke(
    {"messages": [{"role": "user", "content": "我叫小明"}]},
    config=config,
)

# 第二轮 —— 记住上下文
result = agent.invoke(
    {"messages": [{"role": "user", "content": "我叫什么？"}]},
    config=config,
)
print(result["messages"][-1].content)   # "你叫小明"
```

### 生产环境 Checkpointer

```python
# 开发用 MemorySaver（内存）
memory = MemorySaver()

# 生产用 SqliteSaver 或 PostgreSQL
from langgraph.checkpoint.sqlite import SqliteSaver
sqlite = SqliteSaver.from_conn_string(":memory:")

# PostgreSQL（推荐生产）
from langgraph.checkpoint.postgres import PostgresSaver
pg = PostgresSaver.from_conn_string("postgresql://...")
```

## 六、Human-in-the-Loop

```python
from langgraph.types import interrupt, Command

@tool
def send_email(to: str, subject: str, body: str):
    """发送邮件。执行前需要人工确认。"""
    response = interrupt({
        "action": "send_email",
        "to": to,
        "subject": subject,
        "message": "确认发送此邮件？",
    })
    if response.get("action") == "approve":
        return f"邮件已发送至 {to}"
    return "邮件已取消"

# Agent 暂停后，恢复执行：
resumed = agent.invoke(
    Command(resume={"action": "approve"}),
    config=config,
)
```

## 七、多 Agent —— Supervisor 模式

```python
from langgraph.graph import StateGraph, START, END, MessagesState

def supervisor(state: MessagesState):
    """主管：决定调用哪个 Agent"""
    # LLM 决定：route to "researcher" / "coder" / FINISH
    ...

def researcher(state: MessagesState):
    """研究员 Agent"""
    ...

def coder(state: MessagesState):
    """程序员 Agent"""
    ...

builder = StateGraph(MessagesState)
builder.add_node("supervisor", supervisor)
builder.add_node("researcher", researcher)
builder.add_node("coder", coder)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route_decision)
builder.add_edge("researcher", "supervisor")
builder.add_edge("coder", "supervisor")
```

```
START → supervisor → researcher → supervisor → coder → supervisor → END
```

## 八、Retry 策略

```python
from langgraph.types import RetryPolicy

builder.add_node(
    "api_call",
    api_call_function,
    retry_policy=RetryPolicy(max_attempts=3),
)
```

## 九、可视化

```python
# 生成 Mermaid 图
print(agent.get_graph().draw_mermaid())

# 生成 PNG 图片（需要安装依赖）
agent.get_graph().draw_mermaid_png()
```

## 十、面试高频问题

**Q: LangChain Agent 和 LangGraph Agent 的区别？**
A: 旧 AgentExecutor 是简单循环。LangGraph 把工作流建模为有向图，支持显式状态、条件路由、Checkpointing、Human-in-the-Loop。生产推荐 LangGraph。

**Q: Reducer 是什么？**
A: 定义状态更新如何合并。`Annotated[list, operator.add]` 表示追加到列表。默认是替换。

**Q: Checkpointing 有什么用？**
A: 保存图状态快照。支持对话持久化、时间旅行调试、中断后恢复。

**Q: 什么时候用条件边？**
A: ReAct 循环中 LLM 可能调用工具也可能直接回答。条件边根据状态动态路由。
