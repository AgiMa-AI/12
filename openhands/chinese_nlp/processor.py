"""
中文处理器模块。

本模块提供中文文本处理功能，包括分词、词性标注、命名实体识别等。
"""

import os
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Union, Set, Tuple

logger = logging.getLogger("chinese_processor")

class ChineseProcessor:
    """中文处理器类，用于处理中文文本。"""
    
    def __init__(self, use_jieba: bool = True, use_hanlp: bool = False):
        """
        初始化中文处理器。
        
        参数:
            use_jieba: 是否使用jieba分词
            use_hanlp: 是否使用HanLP
        """
        self.use_jieba = use_jieba
        self.use_hanlp = use_hanlp
        
        # 加载分词器
        self._load_tokenizers()
        
        # 加载停用词
        self.stopwords = self._load_stopwords()
        
        # 加载同义词
        self.synonyms = self._load_synonyms()
        
        # 加载情感词典
        self.sentiment_dict = self._load_sentiment_dict()
        
        # 线程锁
        self.lock = threading.RLock()
        
        logger.info("中文处理器初始化完成")
    
    def _load_tokenizers(self):
        """加载分词器。"""
        if self.use_jieba:
            try:
                import jieba
                import jieba.posseg as pseg
                self.jieba = jieba
                self.jieba_pseg = pseg
                logger.info("jieba分词器加载成功")
            except ImportError:
                logger.warning("jieba未安装，将不使用jieba分词")
                self.use_jieba = False
        
        if self.use_hanlp:
            try:
                import hanlp
                self.hanlp = hanlp
                self.hanlp_tokenizer = hanlp.load('CTB6_CONVSEG')
                self.hanlp_tagger = hanlp.load('CTB5_POS_RNN_FASTTEXT_ZH')
                self.hanlp_ner = hanlp.load('MSRA_NER_BERT_BASE_ZH')
                logger.info("HanLP加载成功")
            except ImportError:
                logger.warning("HanLP未安装，将不使用HanLP")
                self.use_hanlp = False
    
    def _load_stopwords(self) -> Set[str]:
        """
        加载停用词。
        
        返回:
            停用词集合
        """
        stopwords = set()
        
        # 默认停用词
        default_stopwords = {
            "的", "了", "和", "是", "就", "都", "而", "及", "与", "着",
            "或", "一个", "没有", "我们", "你们", "他们", "她们", "它们",
            "这个", "那个", "这些", "那些", "这样", "那样", "如此", "什么",
            "哪些", "怎样", "怎么", "什么样", "啊", "呀", "哦", "哎", "唉"
        }
        
        stopwords.update(default_stopwords)
        
        # 尝试从文件加载停用词
        try:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            stopwords_file = os.path.join(module_dir, "data", "stopwords.txt")
            
            if os.path.exists(stopwords_file):
                with open(stopwords_file, "r", encoding="utf-8") as f:
                    for line in f:
                        word = line.strip()
                        if word:
                            stopwords.add(word)
                
                logger.info(f"从{stopwords_file}加载了{len(stopwords)}个停用词")
        
        except Exception as e:
            logger.error(f"加载停用词失败: {e}")
        
        return stopwords
    
    def _load_synonyms(self) -> Dict[str, List[str]]:
        """
        加载同义词。
        
        返回:
            同义词字典
        """
        synonyms = {}
        
        # 尝试从文件加载同义词
        try:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            synonyms_file = os.path.join(module_dir, "data", "synonyms.json")
            
            if os.path.exists(synonyms_file):
                with open(synonyms_file, "r", encoding="utf-8") as f:
                    synonyms = json.load(f)
                
                logger.info(f"从{synonyms_file}加载了{len(synonyms)}组同义词")
        
        except Exception as e:
            logger.error(f"加载同义词失败: {e}")
        
        return synonyms
    
    def _load_sentiment_dict(self) -> Dict[str, float]:
        """
        加载情感词典。
        
        返回:
            情感词典
        """
        sentiment_dict = {}
        
        # 尝试从文件加载情感词典
        try:
            module_dir = os.path.dirname(os.path.abspath(__file__))
            sentiment_file = os.path.join(module_dir, "data", "sentiment.json")
            
            if os.path.exists(sentiment_file):
                with open(sentiment_file, "r", encoding="utf-8") as f:
                    sentiment_dict = json.load(f)
                
                logger.info(f"从{sentiment_file}加载了{len(sentiment_dict)}个情感词")
        
        except Exception as e:
            logger.error(f"加载情感词典失败: {e}")
        
        return sentiment_dict
    
    def tokenize(self, text: str) -> List[str]:
        """
        分词。
        
        参数:
            text: 待分词文本
            
        返回:
            分词结果
        """
        with self.lock:
            if self.use_jieba:
                return list(self.jieba.cut(text))
            elif self.use_hanlp:
                return self.hanlp_tokenizer(text)
            else:
                # 简单的字符分割
                return list(text)
    
    def pos_tag(self, text: str) -> List[Tuple[str, str]]:
        """
        词性标注。
        
        参数:
            text: 待标注文本
            
        返回:
            词性标注结果，格式为[(词, 词性), ...]
        """
        with self.lock:
            if self.use_jieba:
                return [(word, pos) for word, pos in self.jieba_pseg.cut(text)]
            elif self.use_hanlp:
                words = self.hanlp_tokenizer(text)
                tags = self.hanlp_tagger(words)
                return list(zip(words, tags))
            else:
                # 简单的字符分割，默认词性为n（名词）
                return [(char, 'n') for char in text]
    
    def ner(self, text: str) -> List[Tuple[str, str, int, int]]:
        """
        命名实体识别。
        
        参数:
            text: 待识别文本
            
        返回:
            命名实体识别结果，格式为[(实体, 类型, 开始位置, 结束位置), ...]
        """
        with self.lock:
            if self.use_hanlp:
                entities = []
                result = self.hanlp_ner(text)
                
                for entity, entity_type, start, end in result:
                    entities.append((entity, entity_type, start, end))
                
                return entities
            else:
                # 简单的规则匹配
                entities = []
                
                # 匹配常见地名
                for location in ["北京", "上海", "广州", "深圳", "杭州", "南京", "成都", "重庆", "武汉", "西安"]:
                    start = 0
                    while True:
                        start = text.find(location, start)
                        if start == -1:
                            break
                        entities.append((location, "LOC", start, start + len(location)))
                        start += len(location)
                
                # 匹配常见人名前缀
                for prefix in ["张", "王", "李", "赵", "刘", "陈", "杨", "黄", "周", "吴"]:
                    start = 0
                    while True:
                        start = text.find(prefix, start)
                        if start == -1:
                            break
                        
                        # 检查是否是人名（简单假设：前缀后面跟着1-2个字）
                        if start + 1 < len(text) and start + 3 >= len(text):
                            name = text[start:start+2] if start + 2 < len(text) else text[start:start+1]
                            entities.append((name, "PER", start, start + len(name)))
                        
                        start += 1
                
                return entities
    
    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """
        去除停用词。
        
        参数:
            tokens: 分词结果
            
        返回:
            去除停用词后的分词结果
        """
        return [token for token in tokens if token not in self.stopwords]
    
    def get_synonyms(self, word: str) -> List[str]:
        """
        获取同义词。
        
        参数:
            word: 词语
            
        返回:
            同义词列表
        """
        return self.synonyms.get(word, [])
    
    def get_sentiment(self, text: str) -> float:
        """
        获取情感倾向。
        
        参数:
            text: 文本
            
        返回:
            情感倾向值，正值表示积极，负值表示消极，0表示中性
        """
        tokens = self.tokenize(text)
        sentiment = 0.0
        
        for token in tokens:
            sentiment += self.sentiment_dict.get(token, 0.0)
        
        return sentiment
    
    def extract_keywords(self, text: str, topK: int = 5) -> List[Tuple[str, float]]:
        """
        提取关键词。
        
        参数:
            text: 文本
            topK: 返回的关键词数量
            
        返回:
            关键词列表，格式为[(关键词, 权重), ...]
        """
        with self.lock:
            if self.use_jieba:
                import jieba.analyse
                return jieba.analyse.extract_tags(text, topK=topK, withWeight=True)
            else:
                # 简单的TF统计
                tokens = self.tokenize(text)
                tokens = self.remove_stopwords(tokens)
                
                # 计算词频
                freq = {}
                for token in tokens:
                    freq[token] = freq.get(token, 0) + 1
                
                # 按词频排序
                keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)
                
                # 归一化权重
                total = sum(freq.values())
                keywords = [(word, count / total) for word, count in keywords[:topK]]
                
                return keywords
    
    def summarize(self, text: str, ratio: float = 0.2) -> str:
        """
        文本摘要。
        
        参数:
            text: 文本
            ratio: 摘要占原文比例
            
        返回:
            摘要文本
        """
        # 分割句子
        sentences = []
        for sep in ["。", "！", "？", "\n"]:
            text = text.replace(sep, sep + "[SEP]")
        
        sentences = [s.strip() for s in text.split("[SEP]") if s.strip()]
        
        if not sentences:
            return ""
        
        # 如果句子数量少于3，直接返回第一句
        if len(sentences) < 3:
            return sentences[0]
        
        # 提取关键词
        keywords = self.extract_keywords(text, topK=10)
        keywords_dict = {word: weight for word, weight in keywords}
        
        # 计算句子权重
        sentence_weights = []
        for sentence in sentences:
            tokens = self.tokenize(sentence)
            weight = sum(keywords_dict.get(token, 0) for token in tokens)
            sentence_weights.append((sentence, weight))
        
        # 按权重排序
        sentence_weights.sort(key=lambda x: x[1], reverse=True)
        
        # 选择权重最高的句子
        num_sentences = max(1, int(len(sentences) * ratio))
        summary_sentences = [s[0] for s in sentence_weights[:num_sentences]]
        
        # 按原文顺序排序
        summary_sentences.sort(key=lambda s: sentences.index(s))
        
        return "。".join(summary_sentences) + "。"
    
    def correct(self, text: str) -> str:
        """
        文本纠错。
        
        参数:
            text: 文本
            
        返回:
            纠错后的文本
        """
        # 简单的规则纠错
        corrections = {
            "泸州": "泸州",
            "挨个": "挨个",
            "妳": "你",
            "裏": "里",
            "喂": "喂",
            "麽": "么",
            "這": "这",
            "哪裡": "哪里",
            "甚麼": "什么"
        }
        
        for wrong, correct in corrections.items():
            text = text.replace(wrong, correct)
        
        return text

# 全局中文处理器实例
_chinese_processor = None

def get_chinese_processor(use_jieba: bool = True, use_hanlp: bool = False) -> ChineseProcessor:
    """
    获取全局中文处理器实例。
    
    参数:
        use_jieba: 是否使用jieba分词
        use_hanlp: 是否使用HanLP
        
    返回:
        中文处理器实例
    """
    global _chinese_processor
    
    if _chinese_processor is None:
        _chinese_processor = ChineseProcessor(use_jieba, use_hanlp)
    
    return _chinese_processor