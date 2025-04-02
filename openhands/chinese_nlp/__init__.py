"""
中文自然语言处理模块。

本模块提供中文自然语言处理功能，包括分词、词性标注、命名实体识别等。
"""

from openhands.chinese_nlp.processor import ChineseProcessor, get_chinese_processor
from openhands.chinese_nlp.parser import ChineseParser
from openhands.chinese_nlp.enhancer import ChineseEnhancer

__all__ = [
    "ChineseProcessor",
    "get_chinese_processor",
    "ChineseParser",
    "ChineseEnhancer"
]