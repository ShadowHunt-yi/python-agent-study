# RAG 专题（检索增强生成）

> RAG 是 Agent 开发的核心技术，面试高频考点。本专题从原理到实战全覆盖。

---

## 一、RAG 完整流程

```
┌─────────────────────────────────────────────────────────┐
│                    离线索引阶段                           │
│  文档加载 → 文本分块 → Embedding → 存入向量数据库          │
└─────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│                    在线查询阶段                           │
│  用户问题 → Embedding → 向量检索 → 拼接上下文 → LLM 生成   │
└─────────────────────────────────────────────────────────┘
```

### 为什么需要 RAG？

| 问题 | 解决方案 |
|------|---------|
| LLM 知识截止日期 | RAG 注入最新信息 |
| LLM 幻觉（编造事实） | RAG 提供真实上下文 |
| 企业私有数据 | RAG 注入内部知识库 |
| LLM 上下文窗口限制 | RAG 只检索最相关的片段 |

### RAG vs 微调

| 维度 | RAG | 微调 |
|------|-----|------|
| 知识更新 | 实时更新文档即可 | 需要重新训练 |
| 成本 | 低（只需向量数据库） | 高（需要 GPU） |
| 可解释性 | 高（可以追溯来源） | 低（黑盒） |
| 适用场景 | 知识密集型问答 | 风格/格式调整 |
| 延迟 | 较高（检索 + 生成） | 较低（直接生成） |

---

## 二、Embedding 原理

### 什么是 Embedding？

Embedding 是将文本映射为高维向量（如 1536 维），向量之间的距离代表语义相似度。

```
"退款" → [0.12, -0.34, 0.56, ...]  ┐ 距离近（语义相似）
"退钱" → [0.11, -0.35, 0.55, ...]  ┘

"退款" → [0.12, -0.34, 0.56, ...]  ┐ 距离远（语义不同）
"天气" → [-0.45, 0.78, -0.12, ...] ┘
```

### 距离度量

```python
import numpy as np

# 余弦相似度（最常用）
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# 欧氏距离
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

# 点积（向量已归一化时等价于余弦相似度）
def dot_product(a, b):
    return np.dot(a, b)
```

### Embedding 模型选择

| 模型 | 维度 | 特点 | 适用场景 |
|------|------|------|---------|
| text-embedding-3-small | 1536 | OpenAI，性价比高 | 通用场景 |
| text-embedding-3-large | 3072 | OpenAI，最高精度 | 高精度需求 |
| bge-m3 | 1024 | 开源，多语言 | 中文场景 |
| e5-mistral-7b | 4096 | 开源，大模型 | 高精度开源需求 |

```python
from openai import OpenAI

client = OpenAI()

# 单个文本
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="Python 是最好的编程语言",
)
vector = response.data[0].embedding
print(f"维度: {len(vector)}")  # 1536

# 批量文本
texts = ["退款流程", "退货政策", "天气预报"]
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts,
)
vectors = [item.embedding for item in response.data]
```

---

## 三、向量数据库对比

| 数据库 | 类型 | 特点 | 适用场景 |
|--------|------|------|---------|
| **FAISS** | 库 | Facebook 开源，纯内存 | 原型开发、小数据集 |
| **Chroma** | 嵌入式 | 轻量级，简单易用 | 本地开发、小项目 |
| **Milvus** | 分布式 | 生产级，支持万亿向量 | 大规模生产环境 |
| **Qdrant** | 服务 | Rust 实现，高性能 | 性能敏感场景 |
| **Pinecone** | SaaS | 全托管，无需运维 | 不想运维 |
| **Weaviate** | 服务 | 支持混合搜索 | 需要混合检索 |

### FAISS 快速上手

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 创建向量库
texts = ["Python 是编程语言", "AI 是人工智能", "Agent 是智能体"]
vectorstore = FAISS.from_texts(texts, embeddings)

# 检索
results = vectorstore.similarity_search("编程", k=2)
for doc in results:
    print(doc.page_content)
```

### Chroma 快速上手

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# 创建向量库（持久化到磁盘）
vectorstore = Chroma.from_texts(
    texts=["Python 是编程语言", "AI 是人工智能"],
    embedding=embeddings,
    persist_directory="./chroma_db",
)

# 检索
results = vectorstore.similarity_search("编程", k=2)
```

---

## 四、文档加载与分块

### 文档加载器

```python
from langchain_community.document_loaders import (
    PyPDFLoader,           # PDF
    TextLoader,            # 纯文本
    DirectoryLoader,       # 目录
    UnstructuredMarkdownLoader,  # Markdown
    WebBaseLoader,         # 网页
)

# 加载 PDF
loader = PyPDFLoader("document.pdf")
documents = loader.load()

# 加载目录下所有 txt
loader = DirectoryLoader("./docs", glob="**/*.txt")
documents = loader.load()

# 加载网页
loader = WebBaseLoader("https://example.com/article")
documents = loader.load()
```

### 分块策略

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownTextSplitter,
)

# 1. 递归字符分割（最常用）
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # 每块最大字符数
    chunk_overlap=200,      # 重叠字符数（保持语义连续）
    length_function=len,
    separators=["\n\n", "\n", " ", ""],  # 分割优先级
)
chunks = splitter.split_documents(documents)

# 2. Markdown 分割（按标题）
splitter = MarkdownTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.split_documents(documents)

# 3. Token 分割（精确控制 token 数）
splitter = TokenTextSplitter(
    chunk_size=500,         # 每块最大 token 数
    chunk_overlap=50,
)
chunks = splitter.split_documents(documents)
```

### 分块参数调优

| 参数 | 太小 | 太大 | 推荐值 |
|------|------|------|--------|
| chunk_size | 检索精准但丢失上下文 | 检索不精准但上下文完整 | 500-1500 字符 |
| chunk_overlap | 块间语义断裂 | 重复内容太多 | 10%-20% of chunk_size |

### Parent-Child Chunk 策略

```python
# 小块检索，大块送给 LLM
# 优势：提高召回率的同时保持上下文

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

# 小块（用于检索）
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

# 大块（送给 LLM）
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

store = InMemoryStore()
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
```

---

## 五、检索策略

### 1. 向量检索（基础）

```python
# 相似度检索
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},  # 返回 top 3
)

# MMR（最大边际相关性）—— 增加结果多样性
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 10},  # 从 10 个中选 3 个最多样化的
)
```

### 2. 混合检索（向量 + 关键词）

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# BM25 关键词检索
bm25_retriever = BM25Retriever.from_texts(texts)
bm25_retriever.k = 3

# 向量检索
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 混合检索（各占 50% 权重）
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5],
)

results = ensemble_retriever.invoke("退款流程")
```

### 3. 重排序（Re-ranking）

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain_cohere import CohereRerank

# 使用 Cohere 重排序
reranker = CohereRerank(model="rerank-v3.5", top_n=3)

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=retriever,
)

results = compression_retriever.invoke("退款流程")
```

### 4. 多查询检索（Multi-Query）

```python
from langchain.retrievers import MultiQueryRetriever
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")

# 自动生成多个查询变体，提高召回率
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm,
)

# 用户问："退款流程是什么？"
# 自动生成：
# - "如何申请退款？"
# - "退货退款的步骤"
# - "退款需要多长时间？"
```

### 5. HyDE（假设文档嵌入）

```python
from langchain.chains import HypotheticalDocumentEmbedder
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# HyDE：先让 LLM 生成假设答案，用假设答案去检索
hyde_embeddings = HypotheticalDocumentEmbedder.from_llm(
    llm=llm,
    base_embeddings=embeddings,
    prompt_key="web_search",  # 使用预定义的 prompt
)

# 用户问："Python 的 GIL 是什么？"
# LLM 生成假设答案："GIL 是 Python 的全局解释器锁..."
# 用假设答案的 embedding 去检索，找到更相关的文档
```

---

## 六、RAG 优化技巧

### 1. 上下文压缩

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# 只保留与问题相关的部分
compressor = LLMChainExtractor.from_llm(ChatOpenAI(model="gpt-4o"))

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever,
)
```

### 2. 查询改写

```python
from langchain.prompts import ChatPromptTemplate

# 查询改写 prompt
rewrite_prompt = ChatPromptTemplate.from_template("""
你是一个查询优化专家。请将用户的原始问题改写为更适合检索的形式。

原始问题: {question}

改写后的查询:
""")

# 示例：
# 原始："退款"
# 改写："退款流程是什么？如何申请退款？退款需要多长时间？"
```

### 3. 答案验证

```python
from langchain.prompts import ChatPromptTemplate

# 验证答案是否基于上下文
verify_prompt = ChatPromptTemplate.from_template("""
请判断以下答案是否基于给定的上下文。如果答案中有上下文未提及的信息，请指出。

上下文: {context}
问题: {question}
答案: {answer}

判断（是/否）和解释:
""")
```

---

## 七、Agentic RAG

### 普通 RAG vs Agentic RAG

| 维度 | 普通 RAG | Agentic RAG |
|------|---------|-------------|
| 流程 | 固定（检索 → 生成） | 动态（Agent 决定是否检索） |
| 查询优化 | 无 | Agent 可以改写查询 |
| 多跳推理 | 不支持 | 支持（先查 A，再查 B） |
| 答案验证 | 无 | Agent 可以判断答案质量 |

### Agentic RAG 实现

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class RAGState(TypedDict):
    query: str
    documents: list
    answer: str
    needs_retrieval: bool

def decide_retrieval(state: RAGState):
    """Agent 决定是否需要检索"""
    # LLM 判断是否需要外部知识
    response = llm.invoke([
        {"role": "system", "content": "判断以下问题是否需要检索外部知识库。"},
        {"role": "user", "content": state["query"]},
    ])
    needs_retrieval = "需要" in response.content
    return {"needs_retrieval": needs_retrieval}

def retrieve(state: RAGState):
    """检索相关文档"""
    docs = retriever.invoke(state["query"])
    return {"documents": docs}

def generate(state: RAGState):
    """生成答案"""
    if state.get("documents"):
        context = "\n".join([doc.page_content for doc in state["documents"]])
        prompt = f"基于以下上下文回答问题：\n\n{context}\n\n问题：{state['query']}"
    else:
        prompt = state["query"]

    answer = llm.invoke(prompt)
    return {"answer": answer.content}

def route_after_decision(state: RAGState):
    if state["needs_retrieval"]:
        return "retrieve"
    return "generate"

# 构建图
builder = StateGraph(RAGState)
builder.add_node("decide", decide_retrieval)
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)

builder.add_edge(START, "decide")
builder.add_conditional_edges("decide", route_after_decision)
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

graph = builder.compile()
```

---

## 八、RAG 评估

### 评估指标

| 指标 | 说明 | 计算方式 |
|------|------|---------|
| **Faithfulness** | 答案是否忠实于上下文 | LLM 判断答案是否基于检索结果 |
| **Answer Relevancy** | 答案是否回答了问题 | LLM 判断答案与问题的相关性 |
| **Context Recall** | 检索是否覆盖了正确答案 | 对比检索结果与标准答案 |
| **Context Precision** | 检索是否精准 | 检索结果中相关文档的比例 |

### RAGAS 评估框架

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

# 准备评估数据
eval_data = {
    "question": ["Python 的 GIL 是什么？"],
    "answer": ["GIL 是全局解释器锁..."],
    "contexts": [["GIL（Global Interpreter Lock）是 Python 的..."]],
    "ground_truth": ["GIL 是 Python 的全局解释器锁，限制同一时刻只有一个线程执行字节码"],
}

# 评估
result = evaluate(
    dataset=eval_data,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
)
print(result)
```

### 评估 Checklist

- [ ] 答案是否准确（与标准答案对比）
- [ ] 答案是否基于检索结果（Faithfulness）
- [ ] 检索结果是否覆盖了正确答案（Context Recall）
- [ ] 检索结果是否精准（Context Precision）
- [ ] 延迟是否可接受（< 3s）
- [ ] Token 消耗是否合理

---

## 九、生产环境部署

### 架构设计

```
用户查询
    ↓
查询改写（可选）
    ↓
混合检索（向量 + BM25）
    ↓
重排序（Re-ranking）
    ↓
上下文压缩（可选）
    ↓
LLM 生成
    ↓
答案验证（可选）
    ↓
返回答案 + 引用来源
```

### 性能优化

1. **缓存**：相似查询直接返回缓存答案
2. **异步**：检索和 LLM 调用并行化
3. **批量**：批量 Embedding 和检索
4. **索引优化**：选择合适的向量索引类型（HNSW、IVF）

### 监控指标

- 检索延迟（P50、P99）
- LLM 生成延迟
- 答案准确率
- 用户满意度
- Token 消耗

---

## 十、面试题

### Q1: RAG 的完整流程是什么？

**答**：RAG 分为离线索引和在线查询两个阶段：
- **离线索引**：文档加载 → 文本分块 → Embedding → 存入向量数据库
- **在线查询**：用户问题 → Embedding → 向量检索 → 拼接上下文 → LLM 生成答案

### Q2: 为什么需要 RAG？

**答**：RAG 解决了 LLM 的三个核心问题：
1. **知识截止**：RAG 注入最新信息
2. **幻觉**：RAG 提供真实上下文，减少编造
3. **私有数据**：RAG 注入企业内部知识库

### Q3: RAG 和微调怎么选？

**答**：
- **RAG**：适合知识密集型问答，知识更新频繁，需要可解释性
- **微调**：适合调整输出风格/格式，知识相对稳定
- **两者结合**：先微调适应领域，再用 RAG 注入具体知识

### Q4: 文本分块策略有哪些？

**答**：
1. **固定大小**：按字符数切分，简单但可能切断语义
2. **递归字符**：按 `\n\n` → `\n` → ` ` 递归切分，保持段落完整性
3. **语义分块**：用 Embedding 判断语义边界，质量最好但慢
4. **文档结构**：按 Markdown 标题、HTML 标签切分

### Q5: chunk_size 和 chunk_overlap 怎么调？

**答**：
- **chunk_size**：太小丢失上下文，太大检索不精准。推荐 500-1500 字符
- **chunk_overlap**：保持块间语义连续。推荐 chunk_size 的 10%-20%

### Q6: 什么是混合检索？

**答**：混合检索结合向量检索和关键词检索（BM25）：
- **向量检索**：理解语义，"退款" 能找到 "退钱"
- **BM25**：精确匹配，适合专有名词
- **混合**：两者结合，取长补短

### Q7: 什么是重排序（Re-ranking）？

**答**：重排序是对初步检索结果进行二次排序：
- 初步检索（向量/BM25）快速召回 top-k 个候选
- 重排序（Cross-Encoder）精确计算相关性，选出 top-n
- 优势：提高精度，减少 LLM 上下文中的噪声

### Q8: Agentic RAG 和普通 RAG 的区别？

**答**：
- **普通 RAG**：固定流程，单次检索 → 生成
- **Agentic RAG**：Agent 决定是否需要检索、检索什么、检索后是否需要再检索
- **优势**：可以多跳推理、判断检索结果是否足够、优化查询词

### Q9: 如何评估 RAG 系统效果？

**答**：四个核心指标：
1. **Faithfulness**：答案是否忠实于检索到的上下文
2. **Answer Relevancy**：答案是否回答了问题
3. **Context Recall**：检索是否覆盖了正确答案
4. **Context Precision**：检索是否精准

### Q10: RAG 的常见优化手段？

**答**：
1. **Query 改写**：HyDE、Multi-Query
2. **混合检索**：向量 + BM25
3. **Re-ranking**：Cross-Encoder 重排序
4. **上下文压缩**：只保留相关部分
5. **Parent-Child Chunk**：小块检索，大块送给 LLM
