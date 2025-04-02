"""
中文解析器模块。

本模块提供中文文本解析功能，包括意图识别、实体提取等。
"""

import os
import json
import logging
import re
from typing import Dict, List, Any, Optional, Union, Set, Tuple

from openhands.chinese_nlp.processor import ChineseProcessor, get_chinese_processor

logger = logging.getLogger("chinese_parser")

class ChineseParser:
    """中文解析器类，用于解析中文文本。"""
    
    def __init__(self, processor: Optional[ChineseProcessor] = None):
        """
        初始化中文解析器。
        
        参数:
            processor: 中文处理器实例
        """
        self.processor = processor or get_chinese_processor()
        
        # 加载意图模式
        self.intent_patterns = self._load_intent_patterns()
        
        # 加载实体模式
        self.entity_patterns = self._load_entity_patterns()
        
        logger.info("中文解析器初始化完成")
    
    def _load_intent_patterns(self) -> Dict[str, List[str]]:
        """
        加载意图模式。
        
        返回:
            意图模式字典
        """
        # 默认意图模式
        intent_patterns = {
            "greeting": [
                "你好", "您好", "早上好", "中午好", "晚上好", "嗨", "哈喽", "hello", "hi"
            ],
            "farewell": [
                "再见", "拜拜", "回头见", "下次见", "明天见", "晚安", "goodbye", "bye"
            ],
            "thanks": [
                "谢谢", "感谢", "多谢", "非常感谢", "谢谢你", "谢谢您", "thank", "thanks"
            ],
            "help": [
                "帮助", "帮忙", "怎么用", "使用方法", "说明", "指南", "help", "guide"
            ],
            "query": [
                "什么", "怎么", "如何", "为什么", "哪里", "何时", "是否", "能否", "可以", "能不能"
            ],
            "command": [
                "打开", "关闭", "启动", "停止", "开始", "结束", "执行", "运行", "安装", "卸载"
            ],
            "confirm": [
                "是的", "对的", "没错", "确认", "确定", "好的", "可以", "行", "yes", "ok"
            ],
            "deny": [
                "不是", "不对", "错了", "否认", "不行", "不可以", "不好", "no", "nope"
            ]
        }
        
        # 尝试从文件加载意图模式
        try:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            intent_file = os.path.join(module_dir, "data", "intents.json")
            
            if os.path.exists(intent_file):
                with open(intent_file, "r", encoding="utf-8") as f:
                    file_patterns = json.load(f)
                
                # 合并模式
                for intent, patterns in file_patterns.items():
                    if intent in intent_patterns:
                        intent_patterns[intent].extend(patterns)
                    else:
                        intent_patterns[intent] = patterns
                
                logger.info(f"从{intent_file}加载了意图模式")
        
        except Exception as e:
            logger.error(f"加载意图模式失败: {e}")
        
        return intent_patterns
    
    def _load_entity_patterns(self) -> Dict[str, Dict[str, Any]]:
        """
        加载实体模式。
        
        返回:
            实体模式字典
        """
        # 默认实体模式
        entity_patterns = {
            "date": {
                "regex": r"(\d{4}年)?\d{1,2}月\d{1,2}日|(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})|今天|明天|后天|昨天|前天"
            },
            "time": {
                "regex": r"\d{1,2}[点时:]\d{1,2}分?(\d{1,2}秒?)?|上午|中午|下午|晚上|凌晨"
            },
            "number": {
                "regex": r"\d+(\.\d+)?|[一二三四五六七八九十百千万亿]+"
            },
            "phone": {
                "regex": r"1[3-9]\d{9}|(\d{3,4}-?)?\d{7,8}"
            },
            "email": {
                "regex": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
            },
            "url": {
                "regex": r"https?://[^\s]+"
            }
        }
        
        # 尝试从文件加载实体模式
        try:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            entity_file = os.path.join(module_dir, "data", "entities.json")
            
            if os.path.exists(entity_file):
                with open(entity_file, "r", encoding="utf-8") as f:
                    file_patterns = json.load(f)
                
                # 合并模式
                for entity, pattern in file_patterns.items():
                    entity_patterns[entity] = pattern
                
                logger.info(f"从{entity_file}加载了实体模式")
        
        except Exception as e:
            logger.error(f"加载实体模式失败: {e}")
        
        return entity_patterns
    
    def parse(self, text: str) -> Dict[str, Any]:
        """
        解析文本。
        
        参数:
            text: 待解析文本
            
        返回:
            解析结果
        """
        # 纠错
        corrected_text = self.processor.correct(text)
        
        # 分词
        tokens = self.processor.tokenize(corrected_text)
        
        # 词性标注
        pos_tags = self.processor.pos_tag(corrected_text)
        
        # 命名实体识别
        entities = self.processor.ner(corrected_text)
        
        # 提取关键词
        keywords = self.processor.extract_keywords(corrected_text)
        
        # 识别意图
        intent = self.recognize_intent(corrected_text)
        
        # 提取实体
        extracted_entities = self.extract_entities(corrected_text)
        
        # 情感分析
        sentiment = self.processor.get_sentiment(corrected_text)
        
        return {
            "original_text": text,
            "corrected_text": corrected_text,
            "tokens": tokens,
            "pos_tags": pos_tags,
            "entities": entities,
            "keywords": keywords,
            "intent": intent,
            "extracted_entities": extracted_entities,
            "sentiment": sentiment
        }
    
    def recognize_intent(self, text: str) -> Dict[str, float]:
        """
        识别意图。
        
        参数:
            text: 文本
            
        返回:
            意图及其置信度
        """
        intents = {}
        
        # 对每个意图计算匹配度
        for intent, patterns in self.intent_patterns.items():
            score = 0.0
            
            for pattern in patterns:
                if pattern in text:
                    # 如果模式是文本的一部分，增加分数
                    score += 0.5
                    
                    # 如果模式在文本开头，额外增加分数
                    if text.startswith(pattern):
                        score += 0.3
                    
                    # 如果模式与文本完全匹配，额外增加分数
                    if text == pattern:
                        score += 0.2
            
            if score > 0:
                intents[intent] = min(score, 1.0)
        
        # 如果没有匹配的意图，尝试基于关键词的匹配
        if not intents:
            tokens = self.processor.tokenize(text)
            
            for intent, patterns in self.intent_patterns.items():
                score = 0.0
                
                for pattern in patterns:
                    pattern_tokens = self.processor.tokenize(pattern)
                    
                    for token in pattern_tokens:
                        if token in tokens:
                            score += 0.2
                
                if score > 0:
                    intents[intent] = min(score, 1.0)
        
        return intents
    
    def extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        提取实体。
        
        参数:
            text: 文本
            
        返回:
            提取的实体
        """
        entities = {}
        
        # 使用正则表达式提取实体
        for entity_type, pattern in self.entity_patterns.items():
            regex = pattern["regex"]
            matches = re.finditer(regex, text)
            
            entity_list = []
            for match in matches:
                entity_list.append({
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end()
                })
            
            if entity_list:
                entities[entity_type] = entity_list
        
        return entities
    
    def extract_time_expressions(self, text: str) -> List[Dict[str, Any]]:
        """
        提取时间表达式。
        
        参数:
            text: 文本
            
        返回:
            提取的时间表达式
        """
        time_expressions = []
        
        # 提取日期
        if "date" in self.entity_patterns:
            date_regex = self.entity_patterns["date"]["regex"]
            date_matches = re.finditer(date_regex, text)
            
            for match in date_matches:
                time_expressions.append({
                    "type": "date",
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end()
                })
        
        # 提取时间
        if "time" in self.entity_patterns:
            time_regex = self.entity_patterns["time"]["regex"]
            time_matches = re.finditer(time_regex, text)
            
            for match in time_matches:
                time_expressions.append({
                    "type": "time",
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end()
                })
        
        return time_expressions
    
    def extract_location_expressions(self, text: str) -> List[Dict[str, Any]]:
        """
        提取地点表达式。
        
        参数:
            text: 文本
            
        返回:
            提取的地点表达式
        """
        location_expressions = []
        
        # 使用命名实体识别提取地点
        entities = self.processor.ner(text)
        
        for entity, entity_type, start, end in entities:
            if entity_type == "LOC":
                location_expressions.append({
                    "type": "location",
                    "value": entity,
                    "start": start,
                    "end": end
                })
        
        return location_expressions
    
    def extract_person_expressions(self, text: str) -> List[Dict[str, Any]]:
        """
        提取人名表达式。
        
        参数:
            text: 文本
            
        返回:
            提取的人名表达式
        """
        person_expressions = []
        
        # 使用命名实体识别提取人名
        entities = self.processor.ner(text)
        
        for entity, entity_type, start, end in entities:
            if entity_type == "PER":
                person_expressions.append({
                    "type": "person",
                    "value": entity,
                    "start": start,
                    "end": end
                })
        
        return person_expressions
    
    def extract_number_expressions(self, text: str) -> List[Dict[str, Any]]:
        """
        提取数字表达式。
        
        参数:
            text: 文本
            
        返回:
            提取的数字表达式
        """
        number_expressions = []
        
        # 提取数字
        if "number" in self.entity_patterns:
            number_regex = self.entity_patterns["number"]["regex"]
            number_matches = re.finditer(number_regex, text)
            
            for match in number_matches:
                number_expressions.append({
                    "type": "number",
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end()
                })
        
        return number_expressions