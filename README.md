# Python Agent 开发面试速通计划

> 面向：熟悉 JavaScript、需要转 Python Agent 开发岗位的开发者
> 周期：10 天（每天 4-6 小时）
> 目标：通过 Python 基础检验 + 掌握 Agent 开发核心栈
> 共 **60+ 个 Markdown 文件**，覆盖概念、代码、练习、面试题

## 学习路线

```
Day 01  Python 基础语法（JS 对比）         ← 语法层，1天够        (11 文件)
Day 02  Python 进阶特性                    ← 装饰器/生成器/异步    (10 文件)
Day 03  Pydantic v2                       ← 数据校验核心          (5 文件)
Day 04  OpenAI Python SDK                 ← 调用 LLM             (4 文件)
Day 05  LangChain 基础                    ← Agent 框架入门        (4 文件)
Day 06  LangGraph                         ← 生产级 Agent 编排     (4 文件)
Day 07  FastAPI                           ← 暴露 Agent API        (6 文件)
Day 08  asyncio 异步编程                  ← 并发调用必备          (4 文件)
Day 09  pytest 测试                       ← 测试 Agent            (4 文件)
Day 10  综合项目 + 面试准备               ← 完整 Mini Agent       (6 文件)
参考    速查表/面试题/文档链接                                    (4 文件)
```

## 技术栈版本（2025-2026）

| 包 | 版本 | 说明 |
|---|---|---|
| Python | 3.11+ | match 语法 + 新类型提示 |
| langchain | 0.3.x | `create_agent()` 新 API |
| langgraph | 1.0.x | 生产推荐的 Agent 编排 |
| pydantic | 2.x | Rust 核心，5-50x 性能提升 |
| openai | 1.x/2.x | Responses API |
| fastapi | 0.115+ | 依赖注入 |
| pytest | 8.x | pytest-asyncio 0.23+ |

## 每日结构

每个 Day 文件夹包含：
- `README.md` — 当日学习目标和时间安排
- 多个专题 MD — 全量学习内容（含代码示例、JS 对比、面试题）
- `练习题.md` — 动手练习，巩固知识

## 参考资料（resources/）

| 文件 | 内容 |
|------|------|
| `Python速查表.md` | Python vs JS 语法对照 |
| `Agent架构速查.md` | Agent 核心组件和模式 |
| `常见面试题.md` | 30+ 面试高频问题及答案 |
| `官方文档.md` | 所有技术栈的文档链接 |

## Day 10 综合项目

`Day10-综合项目/mini-agent/` 是一个完整的可运行项目：
- Pydantic 数据模型
- OpenAI SDK 调用 LLM
- ReAct 循环（Agent 核心逻辑）
- FastAPI API 服务（同步 + 流式）
- pytest 完整测试
- Docker 部署配置

## 面试重点排序

1. **Python 基础语法** — 必过门槛
2. **Pydantic** — Agent 项目几乎必用
3. **LangChain / LangGraph** — 核心框架
4. **OpenAI SDK** — 调用 LLM 的基础
5. **FastAPI** — 暴露 Agent API
6. **asyncio** — 异步调用 LLM 必须
7. **pytest** — 测试能力证明

## 网站部署（GitHub Pages）

本项目支持通过 MkDocs 生成静态网站，部署到 GitHub Pages。

### 本地预览

```bash
pip install -r requirements-mkdocs.txt
mkdocs serve
# 访问 http://localhost:8000
```

### 部署到 GitHub Pages

1. 创建 GitHub 仓库并推送代码
2. 进入仓库 Settings → Pages → Source 选择 "GitHub Actions"
3. 推送到 main 分支会自动触发部署
4. 访问 `https://你的用户名.github.io/python-agent-study/`

### 自定义域名

在 `mkdocs.yml` 中修改 `site_url`，并在仓库 Settings → Pages 中配置 Custom domain。
