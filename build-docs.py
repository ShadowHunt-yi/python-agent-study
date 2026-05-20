"""构建 MkDocs 文档目录

将项目中的 .md 文件复制到 docs/ 目录，保持目录结构。
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
DOCS = ROOT / "docs"

# 需要包含的目录
INCLUDE_DIRS = [
    "Day01-Python基础语法",
    "Day02-Python进阶",
    "Day03-Pydantic",
    "Day04-OpenAI-SDK",
    "Day05-LangChain基础",
    "Day06-LangGraph",
    "Day07-FastAPI",
    "Day08-asyncio",
    "Day09-测试与调试",
    "Day10-综合项目",
    "resources",
]

# 需要排除的文件
EXCLUDE_FILES = {"README.md"}  # 根目录 README 不需要


def build():
    # 清空 docs/
    if DOCS.exists():
        shutil.rmtree(DOCS)
    DOCS.mkdir()

    # 复制 index.md
    index_src = ROOT / "index.md"
    if index_src.exists():
        shutil.copy2(index_src, DOCS / "index.md")

    # 复制各目录的 md 文件
    for dir_name in INCLUDE_DIRS:
        src_dir = ROOT / dir_name
        if not src_dir.exists():
            continue
        dst_dir = DOCS / dir_name
        dst_dir.mkdir(parents=True)
        for md_file in src_dir.glob("*.md"):
            shutil.copy2(md_file, dst_dir / md_file.name)

    print(f"已生成 docs/ 目录，包含 {len(list(DOCS.rglob('*.md')))} 个文件")


if __name__ == "__main__":
    build()
