# Python 速查表（JS 开发者版）

## 语法对照

| 操作 | Python | JavaScript |
|------|--------|-----------|
| 变量 | `x = 1` | `let x = 1` |
| 常量 | `X = 1`（约定） | `const X = 1` |
| 字符串 | `"hello"` / `'hello'` | `"hello"` / `'hello'` |
| 模板字符串 | `f"Hello, {name}"` | `` `Hello, ${name}` `` |
| 多行字符串 | `"""..."""` | `` `...` `` |
| 数组 | `[1, 2, 3]` | `[1, 2, 3]` |
| 字典 | `{"a": 1}` | `{a: 1}` |
| 空值 | `None` | `null` / `undefined` |
| 布尔 | `True` / `False` | `true` / `false` |
| 类型 | `int`, `str`, `float` | `number`, `string` |

## 函数

```python
# Python
def add(a: int, b: int = 0) -> int:
    return a + b

# lambda
square = lambda x: x ** 2

# 可变参数
def func(*args, **kwargs):
    print(args)    # tuple
    print(kwargs)  # dict
```

```javascript
// JavaScript
function add(a, b = 0) {
    return a + b;
}
const square = x => x ** 2;

function func(...args) {
    console.log(args); // array
}
```

## 列表操作

```python
nums = [1, 2, 3, 4, 5]

# map
squared = [x**2 for x in nums]

# filter
evens = [x for x in nums if x % 2 == 0]

# reduce
from functools import reduce
total = reduce(lambda a, b: a + b, nums)

# find
first = next((x for x in nums if x > 3), None)

# slice
first_three = nums[:3]
last_two = nums[-2:]
```

```javascript
const nums = [1, 2, 3, 4, 5];
const squared = nums.map(x => x ** 2);
const evens = nums.filter(x => x % 2 === 0);
const total = nums.reduce((a, b) => a + b, 0);
const first = nums.find(x => x > 3);
const firstThree = nums.slice(0, 3);
```

## 字典操作

```python
d = {"a": 1, "b": 2, "c": 3}

# 访问
d["a"]           # 1
d.get("x", 0)    # 0（默认值）

# 遍历
for key, value in d.items():
    print(key, value)

# 推导式
squares = {k: v**2 for k, v in d.items()}

# 合并
merged = {**d1, **d2}  # Python 3.5+
merged = d1 | d2        # Python 3.9+
```

```javascript
const d = {a: 1, b: 2, c: 3};
d.a               // 1
d.x ?? 0          // 0
Object.entries(d).forEach(([k, v]) => console.log(k, v));
const merged = {...d1, ...d2};
```

## 异步

```python
import asyncio

async def fetch_data():
    await asyncio.sleep(1)
    return "data"

# 运行
result = asyncio.run(fetch_data())

# 并发
results = await asyncio.gather(
    fetch_data(),
    fetch_data(),
)
```

```javascript
async function fetchData() {
    await new Promise(r => setTimeout(r, 1000));
    return "data";
}

const results = await Promise.all([
    fetchData(),
    fetchData(),
]);
```

## 类

```python
from dataclasses import dataclass

@dataclass
class User:
    name: str
    age: int = 0

    def greet(self) -> str:
        return f"Hi, I'm {self.name}"

user = User(name="Alice", age=30)
```

```javascript
class User {
    constructor(name, age = 0) {
        this.name = name;
        this.age = age;
    }
    greet() {
        return `Hi, I'm ${this.name}`;
    }
}
const user = new User("Alice", 30);
```

## 错误处理

```python
try:
    result = 1 / 0
except ZeroDivisionError as e:
    print(f"错误: {e}")
except Exception as e:
    print(f"其他错误: {e}")
finally:
    print("清理")
```

```javascript
try {
    result = 1 / 0;
} catch (e) {
    console.log(`错误: ${e}`);
} finally {
    console.log("清理");
}
```

## 模块导入

```python
# 整个模块
import os
os.path.join("a", "b")

# 部分导入
from os.path import join
join("a", "b")

# 别名
import numpy as np
```

```javascript
import os from "os";
import { join } from "path";
import np from "numpy";
```

## 装饰器 vs 高阶函数

```python
# Python 装饰器
def log(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@log
def greet(name):
    return f"Hello, {name}"
```

```javascript
// JS 高阶函数
const log = fn => (...args) => {
    console.log(`Calling ${fn.name}`);
    return fn(...args);
};

const greet = log(name => `Hello, ${name}`);
```
