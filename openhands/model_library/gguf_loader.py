"""
GGUF模型加载器。

本模块提供GGUF模型的加载和推理功能，支持各种开源大语言模型。
"""

import os
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger("gguf_loader")

class GGUFLoader:
    """GGUF模型加载器类。"""
    
    def __init__(self, model_path: str, context_window: int = 4096, n_threads: Optional[int] = None, n_gpu_layers: int = -1):
        """
        初始化GGUF模型加载器。
        
        参数:
            model_path: 模型文件路径
            context_window: 上下文窗口大小
            n_threads: 线程数，默认为CPU核心数
            n_gpu_layers: GPU层数，-1表示全部使用GPU
        """
        self.model_path = model_path
        self.context_window = context_window
        self.n_threads = n_threads or os.cpu_count() or 4
        self.n_gpu_layers = n_gpu_layers
        
        # 模型实例
        self.model = None
        
        # 模型信息
        self.model_info = self._get_model_info()
        
        # 线程锁
        self.lock = threading.RLock()
        
        logger.info(f"初始化GGUF模型加载器: {model_path}")
    
    def _get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息。
        
        返回:
            模型信息字典
        """
        # 基本信息
        info = {
            "path": self.model_path,
            "filename": os.path.basename(self.model_path),
            "size": os.path.getsize(self.model_path) if os.path.exists(self.model_path) else 0,
            "context_window": self.context_window
        }
        
        # 尝试从文件名解析模型信息
        filename = info["filename"]
        name_parts = filename.split("-")
        
        if len(name_parts) >= 2:
            info["name"] = name_parts[0]
            
            # 尝试解析参数大小
            for part in name_parts:
                if part.endswith("b") and part[:-1].isdigit():
                    info["parameters"] = int(part[:-1]) * 1_000_000_000
                elif part.endswith("m") and part[:-1].isdigit():
                    info["parameters"] = int(part[:-1]) * 1_000_000
        else:
            info["name"] = filename.split(".")[0]
        
        # 尝试从同名JSON文件加载额外信息
        json_path = os.path.splitext(self.model_path)[0] + ".json"
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    extra_info = json.load(f)
                    info.update(extra_info)
            except Exception as e:
                logger.warning(f"无法加载模型信息文件 {json_path}: {e}")
        
        return info
    
    def load(self) -> bool:
        """
        加载模型。
        
        返回:
            是否成功加载
        """
        with self.lock:
            if self.model is not None:
                logger.info("模型已加载")
                return True
            
            try:
                from llama_cpp import Llama
                
                logger.info(f"加载模型: {self.model_path}")
                logger.info(f"上下文窗口: {self.context_window}")
                logger.info(f"线程数: {self.n_threads}")
                logger.info(f"GPU层数: {self.n_gpu_layers}")
                
                self.model = Llama(
                    model_path=self.model_path,
                    n_ctx=self.context_window,
                    n_threads=self.n_threads,
                    n_gpu_layers=self.n_gpu_layers
                )
                
                logger.info(f"模型加载成功: {self.model_path}")
                return True
            
            except ImportError:
                logger.error("未安装llama-cpp-python，请安装后再使用")
                return False
            
            except Exception as e:
                logger.error(f"加载模型失败: {e}")
                return False
    
    def unload(self) -> bool:
        """
        卸载模型。
        
        返回:
            是否成功卸载
        """
        with self.lock:
            if self.model is None:
                logger.info("模型未加载")
                return True
            
            try:
                # 释放模型
                self.model = None
                
                # 手动触发垃圾回收
                import gc
                gc.collect()
                
                logger.info(f"模型卸载成功: {self.model_path}")
                return True
            
            except Exception as e:
                logger.error(f"卸载模型失败: {e}")
                return False
    
    def is_loaded(self) -> bool:
        """
        检查模型是否已加载。
        
        返回:
            是否已加载
        """
        return self.model is not None
    
    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9, top_k: int = 40, stop: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        生成文本。
        
        参数:
            prompt: 提示文本
            max_tokens: 最大生成token数
            temperature: 温度
            top_p: Top-p采样
            top_k: Top-k采样
            stop: 停止词列表
            
        返回:
            生成结果
        """
        with self.lock:
            if self.model is None:
                if not self.load():
                    return {
                        "text": "模型加载失败，无法生成文本",
                        "error": "模型未加载"
                    }
            
            try:
                # 生成文本
                result = self.model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    stop=stop or []
                )
                
                return {
                    "text": result["choices"][0]["text"],
                    "usage": {
                        "prompt_tokens": result["usage"]["prompt_tokens"],
                        "completion_tokens": result["usage"]["completion_tokens"],
                        "total_tokens": result["usage"]["total_tokens"]
                    },
                    "finish_reason": result["choices"][0]["finish_reason"]
                }
            
            except Exception as e:
                logger.error(f"生成文本失败: {e}")
                return {
                    "text": f"生成文本失败: {str(e)}",
                    "error": str(e)
                }
    
    def chat(self, messages: List[Dict[str, str]], max_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9, top_k: int = 40, stop: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        聊天生成。
        
        参数:
            messages: 消息列表，格式为[{"role": "user", "content": "..."}, ...]
            max_tokens: 最大生成token数
            temperature: 温度
            top_p: Top-p采样
            top_k: Top-k采样
            stop: 停止词列表
            
        返回:
            生成结果
        """
        with self.lock:
            if self.model is None:
                if not self.load():
                    return {
                        "content": "模型加载失败，无法生成回复",
                        "error": "模型未加载"
                    }
            
            try:
                # 构建提示文本
                prompt = self._build_chat_prompt(messages)
                
                # 生成文本
                result = self.generate(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    stop=stop
                )
                
                # 提取回复内容
                content = result.get("text", "")
                
                # 处理可能的格式问题
                if content.startswith("assistant:") or content.startswith("Assistant:"):
                    content = content.split(":", 1)[1].strip()
                
                return {
                    "content": content,
                    "usage": result.get("usage", {}),
                    "finish_reason": result.get("finish_reason", "stop"),
                    "model": self.model_info.get("name", os.path.basename(self.model_path))
                }
            
            except Exception as e:
                logger.error(f"生成聊天回复失败: {e}")
                return {
                    "content": f"生成聊天回复失败: {str(e)}",
                    "error": str(e)
                }
    
    def _build_chat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        构建聊天提示文本。
        
        参数:
            messages: 消息列表
            
        返回:
            提示文本
        """
        prompt = ""
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                prompt += f"system: {content}\n\n"
            elif role == "user":
                prompt += f"user: {content}\n\n"
            elif role == "assistant":
                prompt += f"assistant: {content}\n\n"
            else:
                prompt += f"{role}: {content}\n\n"
        
        # 添加助手角色提示
        prompt += "assistant: "
        
        return prompt
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取模型信息。
        
        返回:
            模型信息
        """
        return self.model_info.copy()

# 模型缓存
_model_cache: Dict[str, GGUFLoader] = {}
_cache_lock = threading.RLock()

def get_gguf_model(model_path: str, context_window: int = 4096, n_threads: Optional[int] = None, n_gpu_layers: int = -1) -> GGUFLoader:
    """
    获取GGUF模型。
    
    参数:
        model_path: 模型文件路径
        context_window: 上下文窗口大小
        n_threads: 线程数
        n_gpu_layers: GPU层数
        
    返回:
        GGUF模型加载器
    """
    with _cache_lock:
        # 规范化路径
        model_path = os.path.abspath(model_path)
        
        # 检查缓存
        if model_path in _model_cache:
            return _model_cache[model_path]
        
        # 创建新模型
        model = GGUFLoader(model_path, context_window, n_threads, n_gpu_layers)
        
        # 添加到缓存
        _model_cache[model_path] = model
        
        return model