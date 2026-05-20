# 03 — 多 Agent 模式

## 一、Supervisor 模式

一个主管 Agent 负责分配任务给多个专业 Agent。

```python
from typing import Literal
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage

model = init_chat_model("openai:gpt-4o", temperature=0)

def supervisor(state: MessagesState) -> dict:
    """主管节点：决定下一步由谁执行"""
    response = model.invoke([
        {"role": "system", "content": """你是一个任务主管。根据对话内容，决定下一步由谁执行：
- researcher: 需要搜索信息时
- coder: 需要写代码时
- FINISH: 任务完成时

只回复一个单词：researcher / coder / FINISH"""},
        *state["messages"],
    ])
    return {"messages": [AIMessage(content=response.content, name="supervisor")]}

def researcher(state: MessagesState) -> dict:
    """研究员节点：搜索和整理信息"""
    response = model.invoke([
        {"role": "system", "content": "你是研究员，负责搜索和整理信息。"},
        *state["messages"],
    ])
    return {"messages": [AIMessage(content=response.content, name="researcher")]}

def coder(state: MessagesState) -> dict:
    """程序员节点：写代码"""
    response = model.invoke([
        {"role": "system", "content": "你是程序员，负责写代码。"},
        *state["messages"],
    ])
    return {"messages": [AIMessage(content=response.content, name="coder")]}

def route_supervisor(state: MessagesState) -> Literal["researcher", "coder", "__end__"]:
    """路由函数"""
    last_msg = state["messages"][-1].content.strip().lower()
    if "finish" in last_msg:
        return "__end__"
    elif "researcher" in last_msg:
        return "researcher"
    elif "coder" in last_msg:
        return "coder"
    return "__end__"

# 构建图
builder = StateGraph(MessagesState)
builder.add_node("supervisor", supervisor)
builder.add_node("researcher", researcher)
builder.add_node("coder", coder)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", route_supervisor)
builder.add_edge("researcher", "supervisor")
builder.add_edge("coder", "supervisor")

graph = builder.compile()
```

```
        ┌──────────┐
        │ supervisor│
        └────┬─────┘
             │
    ┌────────┼────────┐
    ↓        ↓        ↓
researcher  coder    END
    │        │
    └────────┘
    (回到 supervisor)
```

## 二、Swarm 模式

多个 Agent 之间直接交接（handoff），不需要主管。

```python
from langgraph.prebuilt import create_react_agent

def transfer_to_coder():
    """转交给程序员"""
    return "coder"

def transfer_to_researcher():
    """转交给研究员"""
    return "researcher"

researcher = create_react_agent(
    model="openai:gpt-4o",
    tools=[search_tool, transfer_to_coder],
    prompt="你是研究员。需要写代码时转交给 coder。",
)

coder = create_react_agent(
    model="openai:gpt-4o",
    tools=[code_tool, transfer_to_researcher],
    prompt="你是程序员。需要查资料时转交给 researcher。",
)
```

## 三、并行 Agent

多个 Agent 独立工作，最后汇总。

```python
async def parallel_agents(state: dict):
    """并行执行多个 Agent"""
    import asyncio

    results = await asyncio.gather(
        researcher_agent.run(state["query"]),
        coder_agent.run(state["query"]),
        writer_agent.run(state["query"]),
    )
    return {
        "research": results[0],
        "code": results[1],
        "draft": results[2],
    }
```

## 四、流水线模式

Agent 按顺序处理，前一个的输出是后一个的输入。

```python
builder = StateGraph(AgentState)
builder.add_node("researcher", researcher)   # 第 1 步：研究
builder.add_node("writer", writer)           # 第 2 步：写报告
builder.add_node("reviewer", reviewer)       # 第 3 步：审阅

builder.add_edge(START, "researcher")
builder.add_edge("researcher", "writer")
builder.add_edge("writer", "reviewer")
builder.add_edge("reviewer", END)
```

```
START → researcher → writer → reviewer → END
```

## 五、带共享记忆的多 Agent

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()

# Agent A 写入记忆
@tool
def save_research(key: str, content: str, runtime: ToolRuntime):
    runtime.store.put(("research",), key, {"content": content})

# Agent B 读取记忆
@tool
def get_research(key: str, runtime: ToolRuntime):
    data = runtime.store.get(("research",), key)
    return data.value["content"] if data else "未找到"
```

## 六、面试高频问题

**Q: Supervisor 和 Swarm 的区别？**
A: Supervisor 有中央主管分配任务。Swarm 没有主管，Agent 之间直接交接。Supervisor 更可控，Swarm 更灵活。

**Q: 多 Agent 什么时候用并行，什么时候用串行？**
A: 任务独立时用并行（提高效率）。任务有依赖关系时用串行（如先研究再写作）。

**Q: 多 Agent 之间怎么共享信息？**
A: 通过共享 State、共享 Store（记忆）、或通过 Supervisor 传递消息。
