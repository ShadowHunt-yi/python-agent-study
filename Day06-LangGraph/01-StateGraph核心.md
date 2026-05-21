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

### TypedDict 详解

```python
# TypedDict = 带类型注解的字典（不是 class，是 dict 的子类）
from typing import TypedDict

class Point(TypedDict):
    x: float
    y: float

p: Point = {"x": 1.0, "y": 2.0}   # ✅ 创建方式是字典，不是 Point(x=1, y=2)
p["x"]                              # ✅ 字典访问
# p.x                              # ❌ 不能用属性访问（它不是 dataclass）

# JS 对比：类似 TypeScript 的 interface
# interface Point { x: number; y: number; }
```

### Annotated 详解

```python
from typing import Annotated

# Annotated[类型, 元数据...] = 给类型附加额外信息
# 格式：Annotated[原始类型, 附加信息1, 附加信息2, ...]

# LangGraph 用 Annotated 来指定 Reducer（归并策略）
messages: Annotated[list, operator.add]
#                  ↑        ↑
#                  |        └─ Reducer 函数：新旧值如何合并
#                  └────────── 基础类型

# operator.add 就是 lambda a, b: a + b
# 节点返回 {"messages": [msg1]} → 状态中的 messages 变为 old_messages + [msg1]
```

### Reducer（归并器）

```python
# 没有 Reducer = 替换
class State1(TypedDict):
    count: int                          # 节点返回 {"count": 5} → 直接替换为 5

# 有 Reducer = 合并
class State2(TypedDict):
    messages: Annotated[list, operator.add]   # 节点返回 {"messages": [new]} → 追加

# 为什么需要 Reducer？
# 多个节点可能同时更新同一个 key，Reducer 定义如何合并它们的输出
# messages 用 operator.add（追加），因为每个节点都有新消息要添加
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

### Literal 类型

```python
from typing import Literal

# Literal = 变量只能是指定的几个值之一
direction: Literal["left", "right"] = "left"   # ✅
# direction = "up"                              # ❌ 类型检查报错

# JS 对比：TypeScript 的联合字面量类型
# let direction: "left" | "right" = "left";
```

### `__end__` vs END

```python
from langgraph.graph import END

# END 是 LangGraph 预定义的常量，值就是字符串 "__end__"
print(END)           # "__end__"
print(type(END))     # <class 'str'>

# 以下两种写法等价：
return "__end__"     # 直接写字符串
return END           # 用常量（推荐，更清晰，重构时不会出错）
```

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

### interrupt + Command 完整流程

```python
from langgraph.types import interrupt, Command

@tool
def send_email(to: str, subject: str, body: str):
    """发送邮件。执行前需要人工确认。"""
    # interrupt() 暂停图执行，把数据返回给调用方
    response = interrupt({
        "action": "send_email",
        "to": to,
        "subject": subject,
        "message": "确认发送此邮件？",
    })
    # 恢复执行后，response 是 Command(resume=...) 中的值
    if response.get("action") == "approve":
        return f"邮件已发送至 {to}"
    return "邮件已取消"
```

### 执行流程图

```
第一次调用 agent.invoke({"messages": [...]})
┌──────────────────────────────────────────────────┐
│ 1. 用户发消息                                      │
│ 2. LLM 决定调用 send_email(to="boss@corp.com")    │
│ 3. 进入 send_email 工具函数                        │
│ 4. 执行到 interrupt({...})                         │
│ 5. ★ 图暂停 ★，interrupt 数据返回给调用方            │
│    返回值: {"__interrupt": [{value: {...}, ...}]}  │
└──────────────────────────────────────────────────┘

调用方收到中断，展示给用户，等待用户决策

第二次调用 agent.invoke(Command(resume={...}), config)
┌──────────────────────────────────────────────────┐
│ 1. Command(resume={"action": "approve"})          │
│ 2. 图从 interrupt 处恢复执行                        │
│ 3. response = {"action": "approve"}               │
│ 4. send_email 返回 "邮件已发送至 boss@corp.com"     │
│ 5. LLM 整合结果，图执行到 END                       │
└──────────────────────────────────────────────────┘
```

### Command 详解

```python
from langgraph.types import Command

# Command 是恢复执行时传入的指令对象
# resume 的值会成为 interrupt() 表达式的返回值

# 恢复执行
agent.invoke(
    Command(resume={"action": "approve"}),   # resume 的值 → interrupt() 的返回值
    config=config,                            # 必须传同一个 thread_id
)

# JS 类比：类似 Generator 的 gen.send(value)
# interrupt() ≈ yield（暂停，返回数据）
# Command(resume=value) ≈ gen.send(value)（恢复，传入数据）
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

**Q: `__end__` 和 END 的关系？**
A: `END` 是 LangGraph 预定义的常量，值为字符串 `"__end__"`。推荐用 `END` 而不是手写字符串，更清晰且重构安全。

**Q: interrupt 和 Command 怎么配合？**
A: `interrupt(data)` 暂停图执行并返回 data 给调用方。`Command(resume=value)` 恢复执行，value 成为 `interrupt()` 的返回值。类似 Generator 的 `yield` + `send()`。

**Q: TypedDict 和 dataclass 的区别？**
A: TypedDict 是带类型的字典，用 `d["key"]` 访问。dataclass 是类，用 `d.key` 访问。LangGraph 用 TypedDict 因为状态本质上是字典，方便序列化和 Reducer 合并。

**Q: TaskGroup 和 gather 的区别？**
A: gather 等所有任务完成，失败时需要 `return_exceptions=True`。TaskGroup 实现结构化并发，一个任务失败自动取消其他任务，抛出 ExceptionGroup。
