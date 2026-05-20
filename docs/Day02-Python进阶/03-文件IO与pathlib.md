# 03 — 文件 I/O 与 pathlib

## 一、读取文件

```python
# with 语句自动关闭文件
with open("data.txt", "r", encoding="utf-8") as f:
    content = f.read()          # 整个文件作为字符串
    # 或
    lines = f.readlines()       # 行列表
    # 或
    for line in f:              # 逐行迭代（内存高效）
        print(line.strip())
```

```javascript
// JS (Node.js)
import { readFile } from "fs/promises";
const content = await readFile("data.txt", "utf-8");
```

## 二、写入文件

```python
# 覆盖写入
with open("output.txt", "w", encoding="utf-8") as f:
    f.write("Hello, World!\n")
    f.write("Second line\n")

# 追加写入
with open("output.txt", "a", encoding="utf-8") as f:
    f.write("Appended line\n")

# 写入多行
lines = ["line 1\n", "line 2\n", "line 3\n"]
with open("output.txt", "w") as f:
    f.writelines(lines)
```

### 文件打开模式

| 模式 | 说明 |
|------|------|
| `"r"` | 只读（默认） |
| `"w"` | 覆盖写入（文件不存在则创建） |
| `"a"` | 追加写入 |
| `"x"` | 排他创建（文件已存在则报错） |
| `"b"` | 二进制模式（如 `"rb"`） |
| `"t"` | 文本模式（默认，如 `"rt"`） |
| `"+"` | 读写模式（如 `"r+"`） |

## 三、pathlib（推荐）

```python
from pathlib import Path

# 创建路径（跨平台）
p = Path("data") / "files" / "output.txt"
# Windows: data\files\output.txt
# Linux: data/files/output.txt

# 路径信息
p.name           # "output.txt"
p.stem           # "output"（无扩展名）
p.suffix         # ".txt"
p.parent         # Path("data/files")
p.parts          # ('data', 'files', 'output.txt')
p.resolve()      # 绝对路径

# 判断
p.exists()       # True/False
p.is_file()      # True/False
p.is_dir()       # True/False

# 读写
p.read_text(encoding="utf-8")        # 读取全部
p.write_text("content", encoding="utf-8")  # 写入

# 创建目录
Path("logs").mkdir(parents=True, exist_ok=True)

# 遍历
for f in Path(".").glob("*.py"):     # 当前目录的 .py 文件
    print(f)

for f in Path(".").rglob("*.py"):    # 递归搜索
    print(f)

# 列出目录
list(Path(".").iterdir())

# 删除/重命名
p.unlink()                     # 删除文件
p.rename("new_name.txt")       # 重命名
```

```javascript
// JS (Node.js)
import path from "path";
import fs from "fs/promises";

const p = path.join("data", "files", "output.txt");
const exists = await fs.access(p).then(() => true).catch(() => false);
const content = await fs.readFile(p, "utf-8");
```

## 四、JSON 读写

```python
import json

# 写入 JSON
data = {"name": "Alice", "age": 30, "scores": [90, 85, 95]}
with open("data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

# 读取 JSON
with open("data.json", "r", encoding="utf-8") as f:
    loaded = json.load(f)

# 字符串转换
json_str = json.dumps(data, indent=2)
parsed = json.loads(json_str)
```

## 五、CSV 读写

```python
import csv

# 写入
with open("data.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["name", "age", "city"])
    writer.writerow(["Alice", 30, "NYC"])
    writer.writerows([
        ["Bob", 25, "LA"],
        ["Charlie", 35, "SF"],
    ])

# 读取
with open("data.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        print(row)   # ['Alice', '30', 'NYC']

# DictReader/DictWriter —— 按列名访问
with open("data.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row["name"], row["age"])
```

## 六、编码注意事项

```python
# Windows 默认编码可能是 GBK，不是 UTF-8
# 总是显式指定 encoding

# ✓ 正确
with open("file.txt", "r", encoding="utf-8") as f:
    content = f.read()

# ✗ 危险 —— 在 Windows 上可能失败
with open("file.txt", "r") as f:
    content = f.read()
```

## 七、面试高频问题

**Q: `with open()` 比 `open()` 好在哪里？**
A: `with` 保证文件在块结束时关闭，即使发生异常。不用 `with` 需要 try/finally 手动关闭。

**Q: pathlib 比 os.path 好在哪里？**
A: pathlib 面向对象、跨平台（`/` 运算符拼接路径）、API 更直观。os.path 是函数式的。

**Q: 如何高效读取大文件？**
A: 用 `for line in f:` 逐行迭代，不要 `f.read()` 一次全部读入内存。
