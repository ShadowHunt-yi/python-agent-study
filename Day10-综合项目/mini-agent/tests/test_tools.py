"""工具单元测试"""
import pytest
from mini_agent.tools import calculate, search_knowledge, get_current_time


class TestCalculate:
    """计算器工具测试"""

    def test_basic_operations(self):
        assert calculate("2 + 3") == "5"
        assert calculate("10 - 4") == "6"
        assert calculate("3 * 7") == "21"
        assert calculate("10 / 2") == "5.0"

    def test_power(self):
        assert calculate("2 ** 10") == "1024"

    def test_math_functions(self):
        assert calculate("math.sqrt(144)") == "12.0"
        assert calculate("math.log10(1000)") == "3.0"
        assert calculate("abs(-42)") == "42"

    def test_complex_expression(self):
        result = calculate("(2 + 3) * 4 ** 2")
        assert result == "80"

    def test_error_handling(self):
        result = calculate("1 / 0")
        assert "错误" in result

    def test_invalid_expression(self):
        result = calculate("invalid_func()")
        assert "错误" in result


class TestSearchKnowledge:
    """知识库搜索测试"""

    def test_search_python(self):
        result = search_knowledge("Python")
        assert "Python" in result
        assert "1991" in result

    def test_search_agent(self):
        result = search_knowledge("Agent")
        assert "Agent" in result

    def test_search_not_found(self):
        result = search_knowledge("不存在的主题")
        assert "未找到" in result

    def test_case_insensitive(self):
        result = search_knowledge("python")
        assert "Python" in result


class TestGetCurrentTime:
    """时间工具测试"""

    def test_returns_string(self):
        result = get_current_time()
        assert isinstance(result, str)

    def test_format(self):
        result = get_current_time()
        # 格式: "2026-05-20 10:30:00 (Tuesday)"
        assert len(result) >= 19
        assert "(" in result
        assert ")" in result
