# 01 — pytest 基础

## 一、基本断言

```python
# test_basic.py
def test_addition():
    assert 1 + 1 == 2

def test_string():
    assert "hello".upper() == "HELLO"

def test_list():
    assert [1, 2, 3].reverse() is None  # reverse() 返回 None

def test_exception():
    import pytest
    with pytest.raises(ZeroDivisionError):
        1 / 0

def test_approx():
    import pytest
    assert 0.1 + 0.2 == pytest.approx(0.3)
```

```bash
# 运行测试
pytest                    # 运行所有测试
pytest test_basic.py      # 运行指定文件
pytest -v                 # 详细输出
pytest -x                 # 第一个失败就停止
pytest -k "test_add"      # 按名称过滤
pytest --tb=short         # 简化错误输出
```

## 二、Fixture —— 测试前置/后置

```python
import pytest

@pytest.fixture
def sample_data():
    """每个测试都会重新执行"""
    return {"name": "Alice", "age": 30}

@pytest.fixture
def db_connection():
    """带清理的 fixture"""
    conn = create_connection()
    yield conn       # yield 之前是 setup，之后是 teardown
    conn.close()

def test_user_name(sample_data):
    assert sample_data["name"] == "Alice"

def test_user_age(sample_data):
    assert sample_data["age"] == 30

def test_db_query(db_connection):
    result = db_connection.execute("SELECT 1")
    assert result is not None
```

### Fixture 作用域

```python
@pytest.fixture(scope="function")   # 默认：每个函数重新创建
def func_fixture(): ...

@pytest.fixture(scope="class")      # 每个测试类创建一次
def class_fixture(): ...

@pytest.fixture(scope="module")     # 每个模块创建一次
def module_fixture(): ...

@pytest.fixture(scope="session")    # 整个测试会话创建一次
def session_fixture(): ...
```

### conftest.py —— 共享 Fixture

```python
# conftest.py —— 放在项目根目录或测试目录
import pytest

@pytest.fixture
def api_client():
    return httpx.AsyncClient(base_url="http://localhost:8000")

@pytest.fixture
def auth_headers():
    return {"Authorization": "Bearer test-token"}

# 所有测试文件自动可用，无需 import
```

## 三、参数化

```python
import pytest

@pytest.mark.parametrize("input,expected", [
    ("hello", "HELLO"),
    ("world", "WORLD"),
    ("Python", "PYTHON"),
])
def test_upper(input, expected):
    assert input.upper() == expected

# 多参数
@pytest.mark.parametrize("a,b,expected", [
    (1, 2, 3),
    (0, 0, 0),
    (-1, 1, 0),
])
def test_add(a, b, expected):
    assert a + b == expected
```

## 四、异步测试

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_func()
    assert result == "expected"

# 需要安装 pytest-asyncio
# pip install pytest-asyncio
```

```python
# conftest.py
import pytest
import httpx

@pytest.fixture
async def async_client():
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        yield client

@pytest.mark.asyncio
async def test_api(async_client):
    resp = await async_client.get("/health")
    assert resp.status_code == 200
```

## 五、Mock / Patch

```python
from unittest.mock import AsyncMock, patch, MagicMock
import pytest

# Mock 一个函数
def test_with_mock():
    with patch("module.some_function") as mock_func:
        mock_func.return_value = "mocked"
        result = module.some_function()
        assert result == "mocked"
        mock_func.assert_called_once()

# Mock 异步函数
@pytest.mark.asyncio
async def test_async_mock():
    with patch("module.async_func", new_callable=AsyncMock) as mock:
        mock.return_value = "mocked"
        result = await module.async_func()
        assert result == "mocked"

# Mock 类方法
def test_mock_method():
    obj = SomeClass()
    with patch.object(obj, "method", return_value="mocked"):
        assert obj.method() == "mocked"
```

### Mock LLM 调用

```python
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_agent_with_mock_llm():
    mock_llm = AsyncMock()
    mock_llm.invoke.return_value = MagicMock(content="42")

    agent = create_agent(llm=mock_llm, tools=[...])
    result = await agent.run("2+2等于几？")

    assert "42" in result
    mock_llm.invoke.assert_called_once()
```

## 六、Agent 测试策略

```python
# 1. 工具测试（单元测试）
def test_multiply_tool():
    assert multiply(3, 4) == 12

# 2. Agent 决策测试（集成测试）
@pytest.mark.asyncio
async def test_agent_chooses_correct_tool():
    # Mock LLM，验证 Agent 选择正确的工具
    ...

# 3. 端到端测试（慢，谨慎使用）
@pytest.mark.asyncio
async def test_agent_e2e():
    # 真实 LLM 调用
    result = await agent.run("3乘以4等于几？")
    assert "12" in result
```

## 七、调试技巧

```python
# pdb —— Python 调试器
import pdb; pdb.set_trace()   # 设置断点

# 常用命令
# n — 下一行
# s — 进入函数
# c — 继续执行
# p expr — 打印表达式
# l — 显示当前位置代码
# q — 退出

# pytest 中使用
pytest --pdb                 # 失败时进入 pdb
pytest -s                    # 不捕获 print 输出
```

```python
# 日志调试
import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_something():
    logger.debug("测试开始")
    result = do_something()
    logger.debug(f"结果: {result}")
    assert result == expected
```

## 八、面试高频问题

**Q: fixture 的 scope 有哪些？**
A: function（默认）、class、module、session。scope 越大，创建次数越少。

**Q: conftest.py 的作用？**
A: 放置共享 fixture 和 hooks，自动被同级及子目录的测试发现，无需 import。

**Q: 怎么测试 Agent？**
A: 三层：单元测试工具函数、Mock LLM 测试决策逻辑、可选的端到端测试。

**Q: Mock 和 patch 的区别？**
A: Mock 创建模拟对象。patch 替换模块中的真实对象。
