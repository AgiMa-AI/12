"""
中文增强器模块。

本模块提供中文文本增强功能，用于增强模型对中文的理解能力。
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union, Callable

from openhands.chinese_nlp.processor import ChineseProcessor, get_chinese_processor
from openhands.chinese_nlp.parser import ChineseParser

logger = logging.getLogger("chinese_enhancer")

class ChineseEnhancer:
    """中文增强器类，用于增强模型对中文的理解能力。"""
    
    def __init__(self, processor: Optional[ChineseProcessor] = None, parser: Optional[ChineseParser] = None):
        """
        初始化中文增强器。
        
        参数:
            processor: 中文处理器实例
            parser: 中文解析器实例
        """
        self.processor = processor or get_chinese_processor()
        self.parser = parser or ChineseParser(self.processor)
        
        # 加载增强规则
        self.enhancement_rules = self._load_enhancement_rules()
        
        # 加载中文习语
        self.idioms = self._load_idioms()
        
        # 加载中文成语
        self.proverbs = self._load_proverbs()
        
        logger.info("中文增强器初始化完成")
    
    def _load_enhancement_rules(self) -> Dict[str, Callable]:
        """
        加载增强规则。
        
        返回:
            增强规则字典
        """
        # 默认增强规则
        enhancement_rules = {
            "expand_abbreviations": self._expand_abbreviations,
            "explain_idioms": self._explain_idioms,
            "add_pinyin": self._add_pinyin,
            "add_traditional": self._add_traditional,
            "add_context": self._add_context
        }
        
        return enhancement_rules
    
    def _load_idioms(self) -> Dict[str, str]:
        """
        加载中文习语。
        
        返回:
            习语字典
        """
        idioms = {}
        
        # 尝试从文件加载习语
        try:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            idioms_file = os.path.join(module_dir, "data", "idioms.json")
            
            if os.path.exists(idioms_file):
                with open(idioms_file, "r", encoding="utf-8") as f:
                    idioms = json.load(f)
                
                logger.info(f"从{idioms_file}加载了{len(idioms)}个习语")
        
        except Exception as e:
            logger.error(f"加载习语失败: {e}")
        
        return idioms
    
    def _load_proverbs(self) -> Dict[str, str]:
        """
        加载中文成语。
        
        返回:
            成语字典
        """
        proverbs = {}
        
        # 尝试从文件加载成语
        try:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            proverbs_file = os.path.join(module_dir, "data", "proverbs.json")
            
            if os.path.exists(proverbs_file):
                with open(proverbs_file, "r", encoding="utf-8") as f:
                    proverbs = json.load(f)
                
                logger.info(f"从{proverbs_file}加载了{len(proverbs)}个成语")
        
        except Exception as e:
            logger.error(f"加载成语失败: {e}")
        
        return proverbs
    
    def enhance(self, text: str, rules: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        增强文本。
        
        参数:
            text: 待增强文本
            rules: 要应用的规则列表，如果为None则应用所有规则
            
        返回:
            增强结果
        """
        # 解析文本
        parse_result = self.parser.parse(text)
        
        # 应用增强规则
        enhancements = {}
        
        if rules is None:
            rules = list(self.enhancement_rules.keys())
        
        for rule in rules:
            if rule in self.enhancement_rules:
                try:
                    enhancement = self.enhancement_rules[rule](text, parse_result)
                    if enhancement:
                        enhancements[rule] = enhancement
                except Exception as e:
                    logger.error(f"应用规则{rule}失败: {e}")
        
        # 合并结果
        result = {
            "original_text": text,
            "parse_result": parse_result,
            "enhancements": enhancements
        }
        
        return result
    
    def _expand_abbreviations(self, text: str, parse_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        展开缩写。
        
        参数:
            text: 文本
            parse_result: 解析结果
            
        返回:
            展开结果
        """
        # 常见缩写
        abbreviations = {
            "北大": "北京大学",
            "清华": "清华大学",
            "人大": "中国人民大学",
            "北师大": "北京师范大学",
            "复旦": "复旦大学",
            "上交": "上海交通大学",
            "浙大": "浙江大学",
            "南大": "南京大学",
            "武大": "武汉大学",
            "中科大": "中国科学技术大学",
            "国务院": "中华人民共和国国务院",
            "工信部": "中华人民共和国工业和信息化部",
            "发改委": "中华人民共和国国家发展和改革委员会",
            "央行": "中国人民银行",
            "证监会": "中国证券监督管理委员会",
            "银保监会": "中国银行保险监督管理委员会",
            "两会": "全国人民代表大会和中国人民政治协商会议",
            "一带一路": "丝绸之路经济带和21世纪海上丝绸之路",
            "京津冀": "北京、天津、河北",
            "长三角": "长江三角洲地区",
            "珠三角": "珠江三角洲地区"
        }
        
        expanded = {}
        
        for abbr, full in abbreviations.items():
            if abbr in text:
                expanded[abbr] = full
        
        if expanded:
            return {
                "expanded": expanded,
                "explanation": "文本中包含以下缩写，已展开为全称"
            }
        
        return None
    
    def _explain_idioms(self, text: str, parse_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        解释习语和成语。
        
        参数:
            text: 文本
            parse_result: 解析结果
            
        返回:
            解释结果
        """
        explained = {}
        
        # 检查习语
        for idiom, explanation in self.idioms.items():
            if idiom in text:
                explained[idiom] = explanation
        
        # 检查成语
        for proverb, explanation in self.proverbs.items():
            if proverb in text:
                explained[proverb] = explanation
        
        if explained:
            return {
                "explained": explained,
                "explanation": "文本中包含以下习语或成语，已提供解释"
            }
        
        return None
    
    def _add_pinyin(self, text: str, parse_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        添加拼音。
        
        参数:
            text: 文本
            parse_result: 解析结果
            
        返回:
            拼音结果
        """
        try:
            from pypinyin import pinyin, Style
            
            # 获取拼音
            pinyin_result = pinyin(text, style=Style.TONE)
            pinyin_text = ' '.join([''.join(p) for p in pinyin_result])
            
            return {
                "pinyin": pinyin_text,
                "explanation": "文本的拼音表示"
            }
        except ImportError:
            logger.warning("pypinyin未安装，无法添加拼音")
            return None
        except Exception as e:
            logger.error(f"添加拼音失败: {e}")
            return None
    
    def _add_traditional(self, text: str, parse_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        添加繁体字。
        
        参数:
            text: 文本
            parse_result: 解析结果
            
        返回:
            繁体字结果
        """
        try:
            from opencc import OpenCC
            
            # 转换为繁体
            cc = OpenCC('s2t')
            traditional = cc.convert(text)
            
            return {
                "traditional": traditional,
                "explanation": "文本的繁体字表示"
            }
        except ImportError:
            logger.warning("opencc-python-reimplemented未安装，无法添加繁体字")
            return None
        except Exception as e:
            logger.error(f"添加繁体字失败: {e}")
            return None
    
    def _add_context(self, text: str, parse_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        添加上下文信息。
        
        参数:
            text: 文本
            parse_result: 解析结果
            
        返回:
            上下文信息
        """
        context = {}
        
        # 提取时间表达式
        time_expressions = self.parser.extract_time_expressions(text)
        if time_expressions:
            context["time"] = time_expressions
        
        # 提取地点表达式
        location_expressions = self.parser.extract_location_expressions(text)
        if location_expressions:
            context["location"] = location_expressions
        
        # 提取人名表达式
        person_expressions = self.parser.extract_person_expressions(text)
        if person_expressions:
            context["person"] = person_expressions
        
        # 提取数字表达式
        number_expressions = self.parser.extract_number_expressions(text)
        if number_expressions:
            context["number"] = number_expressions
        
        if context:
            return {
                "context": context,
                "explanation": "文本中提取的上下文信息"
            }
        
        return None
    
    def enhance_query(self, query: str) -> str:
        """
        增强查询。
        
        参数:
            query: 查询文本
            
        返回:
            增强后的查询
        """
        # 解析查询
        parse_result = self.parser.parse(query)
        
        # 提取关键词
        keywords = [word for word, _ in parse_result["keywords"]]
        
        # 提取实体
        entities = []
        for entity_list in parse_result["extracted_entities"].values():
            for entity in entity_list:
                entities.append(entity["value"])
        
        # 识别意图
        intents = parse_result["intent"]
        top_intent = max(intents.items(), key=lambda x: x[1])[0] if intents else None
        
        # 构建增强查询
        enhanced_query = query
        
        # 添加关键词
        if keywords:
            enhanced_query += f"\n关键词: {', '.join(keywords)}"
        
        # 添加实体
        if entities:
            enhanced_query += f"\n实体: {', '.join(entities)}"
        
        # 添加意图
        if top_intent:
            enhanced_query += f"\n意图: {top_intent}"
        
        return enhanced_query
    
    def enhance_response(self, query: str, response: str) -> str:
        """
        增强响应。
        
        参数:
            query: 查询文本
            response: 响应文本
            
        返回:
            增强后的响应
        """
        # 解析查询
        query_parse = self.parser.parse(query)
        
        # 解析响应
        response_parse = self.parser.parse(response)
        
        # 提取查询中的关键词
        query_keywords = [word for word, _ in query_parse["keywords"]]
        
        # 检查响应是否包含查询关键词
        missing_keywords = []
        for keyword in query_keywords:
            if keyword not in response:
                missing_keywords.append(keyword)
        
        # 如果缺少关键词，添加补充信息
        if missing_keywords:
            supplement = f"\n\n补充信息: 您的问题涉及到 {', '.join(missing_keywords)}，以上回答可能未完全覆盖这些方面。如需更多相关信息，请告诉我。"
            response += supplement
        
        # 检查是否需要解释习语或成语
        idioms_result = self._explain_idioms(response, response_parse)
        if idioms_result and idioms_result["explained"]:
            explanation = "\n\n名词解释:"
            for term, desc in idioms_result["explained"].items():
                explanation += f"\n- {term}: {desc}"
            response += explanation
        
        return response