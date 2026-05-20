"""FastAPI 应用入口"""
import logging

from fastapi import FastAPI

from .config import settings
from .api.routes import router

# 配置日志
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

app = FastAPI(
    title="Mini Agent",
    description="一个用于学习的 AI Agent API",
    version="0.1.0",
)

# 注册路由
app.include_router(router)


@app.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "model": settings.openai_model,
    }
