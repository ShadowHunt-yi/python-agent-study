FROM python:3.12-slim

WORKDIR /app

# 先复制依赖文件，利用 Docker 缓存（依赖不变时跳过 pip install）
COPY pyproject.toml .
COPY src/ src/

# pip install . 需要 src/ 目录（pyproject.toml 引用了它）
RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "mini_agent.main:app", "--host", "0.0.0.0", "--port", "8000"]
