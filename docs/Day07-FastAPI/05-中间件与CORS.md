# 05 — 中间件与 CORS

## 一、中间件执行顺序

```
请求 → Middleware A → Middleware B → 路由处理 → Middleware B → Middleware A → 响应
       （洋葱模型：先入后出）
```

```python
from fastapi import FastAPI, Request
import time

app = FastAPI()

@app.middleware("http")
async def add_timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    response.headers["X-Process-Time"] = f"{elapsed:.4f}"
    return response

@app.middleware("http")
async def add_request_id(request: Request, call_next):
    import uuid
    request_id = str(uuid.uuid4())[:8]
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response
```

## 二、CORS 中间件

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://myapp.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
    max_age=600,  # 预检请求缓存时间（秒）
)
```

### CORS 常见问题

```python
# 开发环境允许所有
allow_origins=["*"]

# 生产环境限制具体域名
allow_origins=["https://myapp.com", "https://admin.myapp.com"]

# 注意：allow_origins=["*"] 时不能 allow_credentials=True
```

## 三、请求日志中间件

```python
import logging
from fastapi import Request

logger = logging.getLogger("api")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"→ {request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"← {request.method} {request.url.path} {response.status_code}")
    return response
```

## 四、认证中间件

```python
from fastapi import Request, HTTPException

# 公开路由白名单
PUBLIC_ROUTES = {"/", "/health", "/docs", "/openapi.json"}

@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    # 跳过公开路由
    if request.url.path in PUBLIC_ROUTES:
        return await call_next(request)

    # 跳过 OPTIONS 预检请求
    if request.method == "OPTIONS":
        return await call_next(request)

    # 验证 Token
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="缺少认证信息")

    token = auth[7:]
    # 验证 token 逻辑...
    # request.state.user = decoded_user

    return await call_next(request)
```

## 五、限流中间件

```python
from fastapi import Request
from collections import defaultdict
import time

# 简单的内存限流器
rate_limit_store: dict[str, list[float]] = defaultdict(list)

@app.middleware("http")
async def rate_limit(request: Request, call_next):
    client_ip = request.client.host
    now = time.time()
    window = 60  # 1 分钟窗口
    max_requests = 100  # 最大请求数

    # 清理过期记录
    rate_limit_store[client_ip] = [
        t for t in rate_limit_store[client_ip] if now - t < window
    ]

    if len(rate_limit_store[client_ip]) >= max_requests:
        raise HTTPException(status_code=429, detail="请求过于频繁")

    rate_limit_store[client_ip].append(now)
    return await call_next(request)
```

## 六、异常处理中间件

```python
from fastapi import Request
from fastapi.responses import JSONResponse

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "内部服务器错误",
            "detail": str(exc) if app.debug else None,
        },
    )
```

## 七、面试高频问题

**Q: 中间件的执行顺序？**
A: 洋葱模型。先添加的中间件最后执行（请求阶段从外到内，响应阶段从内到外）。

**Q: 中间件和依赖注入的区别？**
A: 中间件作用于所有请求（全局）。依赖注入作用于特定路由（局部）。认证适合用中间件，数据库连接适合用依赖注入。

**Q: CORS 是什么？怎么配置？**
A: 跨域资源共享。配置 `allow_origins`（允许的域名）、`allow_methods`（允许的方法）、`allow_headers`（允许的头部）。
