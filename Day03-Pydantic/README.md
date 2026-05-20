# Day 03 — Pydantic v2

## 学习目标

- [ ] 掌握 BaseModel 基础用法
- [ ] 掌握 Field 验证和 ConfigDict
- [ ] 掌握 field_validator 和 model_validator
- [ ] 掌握 model_dump / model_validate 序列化
- [ ] 掌握 TypeAdapter 非模型验证
- [ ] 了解 computed_field 和 Annotated 类型

## 时间安排（约 4 小时）

| 时段 | 内容 | 时长 |
|------|------|------|
| 上午 | BaseModel + Field + ConfigDict | 1h |
| 上午 | 验证器（field_validator / model_validator） | 1h |
| 下午 | 序列化（model_dump / model_validate） | 1h |
| 下午 | TypeAdapter + 练习 | 1h |

## 为什么 Pydantic 对 Agent 开发至关重要

- LangChain 的工具定义用 Pydantic
- OpenAI SDK 的结构化输出用 Pydantic
- FastAPI 的请求/响应用 Pydantic
- Agent 的状态管理用 Pydantic
- 面试中几乎必问
