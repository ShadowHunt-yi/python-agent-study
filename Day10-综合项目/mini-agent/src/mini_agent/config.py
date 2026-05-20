"""配置管理"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """应用配置，从环境变量或 .env 文件读取"""

    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_base_url: str = "https://api.openai.com/v1"
    max_agent_steps: int = 10
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
