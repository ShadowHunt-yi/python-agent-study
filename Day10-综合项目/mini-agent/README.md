# Mini Agent

一个用于学习的 AI Agent 项目，综合运用 Python、Pydantic、OpenAI SDK、FastAPI、asyncio 等技术栈。

## 功能

- ReAct 模式的 Agent 循环
- 工具调用（计算器、知识库搜索、时间查询）
- 同步和流式 API
- Pydantic 数据校验
- 异步并发

## 快速开始

```bash
# 1. 安装依赖
pip install -e ".[dev]"

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env 填入 OPENAI_API_KEY

# 3. 启动服务
uvicorn mini_agent.main:app --reload --port 8000

# 4. 测试
curl -X POST http://localhost:8000/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "2+3等于几？"}'

# 5. 运行测试
pytest tests/ -v
```

## 项目结构

```
mini-agent/
├── pyproject.toml
├── .env.example
├── src/mini_agent/
│   ├── __init__.py
│   ├── main.py          # FastAPI 入口
│   ├── config.py         # 配置管理
│   ├── models.py         # Pydantic 数据模型
│   ├── tools.py          # 工具定义
│   └── agent.py          # Agent 核心逻辑
└── tests/
    ├── conftest.py
    ├── test_tools.py
    └── test_agent.py
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /health | 健康检查 |
| POST | /agent/chat | 同步聊天 |
| POST | /agent/chat/stream | 流式聊天（SSE） |
