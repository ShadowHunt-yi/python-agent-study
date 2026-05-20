# 03 — LCEL 进阶

## 一、RunnablePassthrough —— 透传数据

```python
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# 透传输入
chain = RunnablePassthrough() | RunnableLambda(lambda x: f"收到: {x}")
result = chain.invoke("hello")   # "收到: hello"

# 透传 + 处理
chain = {
    "question": RunnablePassthrough(),  # 原样透传
    "context": retriever,               # 检索上下文
} | prompt | model | parser
```

## 二、RunnableParallel —— 并行执行

```python
from langchain_core.runnables import RunnableParallel

chain = RunnableParallel(
    summary=summary_chain,
    sentiment=sentiment_chain,
    keywords=keywords_chain,
)

result = chain.invoke({"text": "这是一篇文章..."})
# result = {"summary": "...", "sentiment": "正面", "keywords": [...]}
```

## 三、RunnableLambda —— 自定义函数

```python
from langchain_core.runnables import RunnableLambda

def preprocess(text: str) -> str:
    return text.strip().lower()

preprocessor = RunnableLambda(preprocess)
chain = preprocessor | model | parser
```

## 四、条件路由

```python
from langchain_core.runnables import RunnableBranch, RunnableLambda

def is_math(question: str) -> bool:
    return any(op in question for op in ["+", "-", "*", "/", "计算", "等于"])

branch = RunnableBranch(
    (is_math, math_chain),         # 条件匹配 → math_chain
    general_chain,                  # 默认 → general_chain
)

result = branch.invoke("2+3等于几？")
```

## 五、重试和回退

```python
from langchain_core.runnables import RunnableWithFallbacks

chain = primary_model.with_fallbacks([fallback_model])

# 重试
chain = model.with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True,
)
```

## 六、链式组合实战

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI

# 1. 简单链
simple = prompt | model | StrOutputParser()

# 2. 带预处理的链
chain = (
    RunnableLambda(lambda x: x.strip())
    | prompt
    | model
    | StrOutputParser()
)

# 3. 多步链（第一个链的输出作为第二个链的输入）
first = prompt1 | model | StrOutputParser()
second = prompt2 | model | StrOutputParser()
full = first | second

# 4. RAG 链
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

answer = rag_chain.invoke("Python 是谁创建的？")
```

## 七、获取中间结果

```python
chain = prompt | model | parser

# 获取每一步的输出
for chunk in chain.stream({"topic": "猫"}):
    print(chunk, end="")

# 获取 token 级别的流式
for token in model.stream(prompt.invoke({"topic": "猫"})):
    print(token.content, end="")
```

## 八、面试高频问题

**Q: LCEL 的 `|` 操作符做了什么？**
A: 创建一个 RunnableSequence，前一个的输出作为后一个的输入。支持自动的流式、异步、批处理。

**Q: RunnableParallel 的作用？**
A: 并行执行多个 Runnable，结果合并为 dict。常用于 RAG（同时检索和透传问题）。

**Q: 怎么在 LCEL 中加入自定义逻辑？**
A: 用 `RunnableLambda(func)` 包装普通函数。
