# 01 — BaseModel 基础

## 一、基本用法

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    name: str
    age: int
    email: str
    is_active: bool = True       # 默认值

# 创建实例 —— 自动验证类型
user = User(name="Alice", age=30, email="alice@example.com")
print(user.name)         # "Alice"
print(user.model_dump()) # {'name': 'Alice', 'age': 30, 'email': '...', 'is_active': True}

# 类型自动转换（宽松模式）
user2 = User(name="Bob", age="25", email="bob@test.com")  # "25" → 25
print(user2.age)         # 25（int）

# 验证失败
try:
    User(name="Charlie", age="not_a_number", email="c@test.com")
except ValidationError as e:
    print(e)
```

## 二、Field 约束

```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    price: float = Field(gt=0, le=99999)         # >0, <=99999
    quantity: int = Field(ge=0)                    # >=0
    description: str = Field(default="", max_length=1000)
    tags: list[str] = Field(default_factory=list)  # 可变默认值

p = Product(name="Widget", price=9.99, quantity=100)

try:
    Product(name="", price=-1, quantity=-5)
except ValidationError as e:
    print(e)  # 多个错误
```

### Field 参数速查

| 参数 | 说明 | 适用类型 |
|------|------|---------|
| `default` | 默认值 | 所有 |
| `default_factory` | 默认值工厂 | 可变类型 |
| `min_length` / `max_length` | 长度限制 | str, list |
| `gt` / `ge` / `lt` / `le` | 数值范围 | int, float |
| `pattern` | 正则匹配 | str |
| `description` | 字段描述 | 所有 |
| `alias` | 别名 | 所有 |

## 三、ConfigDict

```python
from pydantic import BaseModel, ConfigDict

class User(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,   # 自动去首尾空白
        extra="forbid",              # 禁止额外字段
        frozen=True,                 # 不可变实例
        from_attributes=True,        # ORM 兼容
    )

    name: str
    age: int

user = User(name="  Alice  ", age=30)
print(user.name)    # "Alice"（自动去空白）

try:
    User(name="Bob", age=25, extra_field="oops")
except ValidationError as e:
    print(e)  # extra 字段被拒绝

# frozen=True 时不可修改
try:
    user.name = "Bob"
except ValidationError as e:
    print(e)
```

### ConfigDict 常用选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `str_strip_whitespace` | 自动去空白 | False |
| `str_min_length` | 全局最小长度 | - |
| `extra` | "allow"/"forbid"/"ignore" | "ignore" |
| `frozen` | 不可变 | False |
| `from_attributes` | 支持 ORM 对象 | False |
| `populate_by_name` | 允许字段名（非 alias） | False |

## 四、嵌套模型

```python
from pydantic import BaseModel

class Address(BaseModel):
    street: str
    city: str
    zip_code: str

class Company(BaseModel):
    name: str
    address: Address           # 嵌套模型
    employees: list[str] = []

company = Company(
    name="Acme",
    address={"street": "123 Main", "city": "NYC", "zip_code": "10001"},
    employees=["Alice", "Bob"],
)
print(company.address.city)    # "NYC"
print(company.model_dump())
```

## 五、model_dump —— 序列化

```python
user = User(name="Alice", age=30)

# 转为 dict
user.model_dump()                          # {'name': 'Alice', 'age': 30}
user.model_dump(exclude={"age"})           # {'name': 'Alice'}
user.model_dump(include={"name"})          # {'name': 'Alice'}

# 转为 JSON 字符串
user.model_dump_json()                     # '{"name":"Alice","age":30}'
user.model_dump_json(indent=2)             # 带缩进的 JSON

# 排除 None 值
user.model_dump(exclude_none=True)
```

## 六、model_validate —— 反序列化

```python
# 从 dict 创建
data = {"name": "Alice", "age": 30}
user = User.model_validate(data)

# 从 JSON 字符串创建
json_str = '{"name": "Bob", "age": 25}'
user = User.model_validate_json(json_str)

# 从 ORM 对象创建（需要 from_attributes=True）
user = User.model_validate(orm_object)
```

## 七、model_json_schema —— JSON Schema

```python
schema = User.model_json_schema()
print(schema)
# {
#   "properties": {
#     "name": {"title": "Name", "type": "string"},
#     "age": {"title": "Age", "type": "integer"}
#   },
#   "required": ["name", "age"],
#   "title": "User",
#   "type": "object"
# }
```

## 八、面试高频问题

**Q: Pydantic v1 和 v2 的主要区别？**
A: v2 用 Rust 核心（pydantic-core），性能提升 5-50x。API 变化：`.dict()` → `.model_dump()`，`.json()` → `.model_dump_json()`，`@validator` → `@field_validator`。

**Q: `model_validate` 和 `model_validate_json` 的区别？**
A: `model_validate` 接收 Python dict，`model_validate_json` 接收 JSON 字符串。

**Q: Pydantic 在 Agent 开发中有什么用？**
A: 定义工具参数 schema、结构化 LLM 输出、API 请求/响应校验、配置管理。
