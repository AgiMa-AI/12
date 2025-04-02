"""
模型更新器。

本模块提供模型自动更新功能，支持从各种来源下载和更新模型。
"""

import os
import json
import logging
import threading
import datetime
import urllib.request
import urllib.error
import hashlib
import shutil
from typing import Dict, List, Any, Optional, Union, Tuple, Callable

from openhands.model_library.model import Model, ModelType, ModelInfo
from openhands.model_library.library import get_model_library

logger = logging.getLogger("model_updater")

class ModelUpdater:
    """模型更新器类。"""
    
    def __init__(self, update_sources: Optional[List[Dict[str, Any]]] = None):
        """
        初始化模型更新器。
        
        参数:
            update_sources: 更新源列表
        """
        # 获取模型库
        self.library = get_model_library()
        
        # 更新源
        self.update_sources = update_sources or []
        
        # 加载默认更新源
        if not self.update_sources:
            self._load_default_sources()
        
        # 更新历史
        self.update_history = []
        
        # 线程锁
        self.lock = threading.RLock()
        
        # 正在更新的模型
        self.updating_models = set()
        
        logger.info(f"初始化模型更新器，{len(self.update_sources)}个更新源")
    
    def _load_default_sources(self) -> None:
        """加载默认更新源。"""
        # 默认更新源
        default_sources = [
            {
                "name": "本地模型目录",
                "type": "local",
                "url": os.path.join(os.path.expanduser("~"), "models"),
                "enabled": True
            },
            {
                "name": "HuggingFace模型",
                "type": "huggingface",
                "url": "https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads",
                "enabled": True
            }
        ]
        
        # 尝试从文件加载更新源
        try:
            config_dir = os.path.join(os.path.expanduser("~"), "openhands", "config")
            os.makedirs(config_dir, exist_ok=True)
            
            sources_file = os.path.join(config_dir, "update_sources.json")
            
            if os.path.exists(sources_file):
                with open(sources_file, "r", encoding="utf-8") as f:
                    sources = json.load(f)
                
                if isinstance(sources, list) and sources:
                    self.update_sources = sources
                    logger.info(f"从{sources_file}加载了{len(sources)}个更新源")
                    return
        
        except Exception as e:
            logger.error(f"加载更新源失败: {e}")
        
        # 使用默认更新源
        self.update_sources = default_sources
        
        # 保存默认更新源
        try:
            config_dir = os.path.join(os.path.expanduser("~"), "openhands", "config")
            os.makedirs(config_dir, exist_ok=True)
            
            sources_file = os.path.join(config_dir, "update_sources.json")
            
            with open(sources_file, "w", encoding="utf-8") as f:
                json.dump(default_sources, f, ensure_ascii=False, indent=2)
            
            logger.info(f"保存了默认更新源到{sources_file}")
        
        except Exception as e:
            logger.error(f"保存默认更新源失败: {e}")
    
    def add_update_source(self, name: str, source_type: str, url: str, enabled: bool = True, auth: Optional[Dict[str, str]] = None) -> bool:
        """
        添加更新源。
        
        参数:
            name: 更新源名称
            source_type: 更新源类型
            url: 更新源URL
            enabled: 是否启用
            auth: 认证信息
            
        返回:
            是否成功
        """
        with self.lock:
            # 检查是否已存在
            for source in self.update_sources:
                if source["name"] == name:
                    logger.warning(f"更新源已存在: {name}")
                    return False
            
            # 添加更新源
            source = {
                "name": name,
                "type": source_type,
                "url": url,
                "enabled": enabled
            }
            
            if auth:
                source["auth"] = auth
            
            self.update_sources.append(source)
            
            # 保存更新源
            self._save_update_sources()
            
            logger.info(f"添加了更新源: {name}")
            return True
    
    def remove_update_source(self, name: str) -> bool:
        """
        删除更新源。
        
        参数:
            name: 更新源名称
            
        返回:
            是否成功
        """
        with self.lock:
            # 查找更新源
            for i, source in enumerate(self.update_sources):
                if source["name"] == name:
                    # 删除更新源
                    del self.update_sources[i]
                    
                    # 保存更新源
                    self._save_update_sources()
                    
                    logger.info(f"删除了更新源: {name}")
                    return True
            
            logger.warning(f"更新源不存在: {name}")
            return False
    
    def enable_update_source(self, name: str, enabled: bool = True) -> bool:
        """
        启用或禁用更新源。
        
        参数:
            name: 更新源名称
            enabled: 是否启用
            
        返回:
            是否成功
        """
        with self.lock:
            # 查找更新源
            for source in self.update_sources:
                if source["name"] == name:
                    # 更新状态
                    source["enabled"] = enabled
                    
                    # 保存更新源
                    self._save_update_sources()
                    
                    logger.info(f"{'启用' if enabled else '禁用'}了更新源: {name}")
                    return True
            
            logger.warning(f"更新源不存在: {name}")
            return False
    
    def _save_update_sources(self) -> bool:
        """
        保存更新源。
        
        返回:
            是否成功
        """
        try:
            config_dir = os.path.join(os.path.expanduser("~"), "openhands", "config")
            os.makedirs(config_dir, exist_ok=True)
            
            sources_file = os.path.join(config_dir, "update_sources.json")
            
            with open(sources_file, "w", encoding="utf-8") as f:
                json.dump(self.update_sources, f, ensure_ascii=False, indent=2)
            
            logger.info(f"保存了更新源到{sources_file}")
            return True
        
        except Exception as e:
            logger.error(f"保存更新源失败: {e}")
            return False
    
    def check_for_updates(self, model_type: Optional[ModelType] = None, callback: Optional[Callable[[str, float], None]] = None) -> List[Dict[str, Any]]:
        """
        检查更新。
        
        参数:
            model_type: 模型类型
            callback: 进度回调函数
            
        返回:
            可更新的模型列表
        """
        with self.lock:
            updates = []
            
            # 获取已有模型
            existing_models = {}
            for model in self.library.get_all_models():
                model_info = model.get_info()
                existing_models[model_info.name] = model_info
            
            # 检查每个更新源
            for i, source in enumerate(self.update_sources):
                if not source["enabled"]:
                    continue
                
                try:
                    # 更新进度
                    if callback:
                        callback(f"正在检查更新源: {source['name']}", i / len(self.update_sources) * 100)
                    
                    # 根据类型检查更新
                    if source["type"] == "local":
                        source_updates = self._check_local_updates(source, existing_models, model_type)
                    elif source["type"] == "huggingface":
                        source_updates = self._check_huggingface_updates(source, existing_models, model_type)
                    else:
                        logger.warning(f"不支持的更新源类型: {source['type']}")
                        source_updates = []
                    
                    # 添加更新
                    updates.extend(source_updates)
                
                except Exception as e:
                    logger.error(f"检查更新源失败: {source['name']}, {e}")
            
            # 更新进度
            if callback:
                callback("更新检查完成", 100)
            
            logger.info(f"检查更新完成，发现{len(updates)}个可更新模型")
            return updates
    
    def _check_local_updates(self, source: Dict[str, Any], existing_models: Dict[str, ModelInfo], model_type: Optional[ModelType] = None) -> List[Dict[str, Any]]:
        """
        检查本地更新源。
        
        参数:
            source: 更新源
            existing_models: 已有模型
            model_type: 模型类型
            
        返回:
            可更新的模型列表
        """
        updates = []
        
        # 检查目录是否存在
        if not os.path.exists(source["url"]) or not os.path.isdir(source["url"]):
            logger.warning(f"更新源目录不存在: {source['url']}")
            return updates
        
        # 遍历目录
        for root, dirs, files in os.walk(source["url"]):
            for file in files:
                # 检查文件扩展名
                if not file.endswith((".gguf", ".bin", ".pt", ".pth", ".onnx", ".safetensors")):
                    continue
                
                # 获取文件路径
                file_path = os.path.join(root, file)
                
                # 获取文件名（不含扩展名）
                file_name = os.path.splitext(file)[0]
                
                # 检查是否已存在
                if file_name in existing_models:
                    # 检查文件大小和修改时间
                    existing_size = existing_models[file_name].size
                    existing_path = existing_models[file_name].path
                    
                    new_size = os.path.getsize(file_path)
                    new_mtime = os.path.getmtime(file_path)
                    
                    # 如果新文件更大或更新，则添加更新
                    if new_size > existing_size or (new_size == existing_size and new_mtime > os.path.getmtime(existing_path)):
                        # 确定模型类型
                        if model_type is None or existing_models[file_name].model_type == model_type:
                            updates.append({
                                "name": file_name,
                                "source": source["name"],
                                "path": file_path,
                                "size": new_size,
                                "existing_size": existing_size,
                                "model_type": existing_models[file_name].model_type.value
                            })
                else:
                    # 新模型
                    # 确定模型类型
                    if model_type is None:
                        updates.append({
                            "name": file_name,
                            "source": source["name"],
                            "path": file_path,
                            "size": os.path.getsize(file_path),
                            "existing_size": 0,
                            "model_type": "other"
                        })
        
        return updates
    
    def _check_huggingface_updates(self, source: Dict[str, Any], existing_models: Dict[str, ModelInfo], model_type: Optional[ModelType] = None) -> List[Dict[str, Any]]:
        """
        检查HuggingFace更新源。
        
        参数:
            source: 更新源
            existing_models: 已有模型
            model_type: 模型类型
            
        返回:
            可更新的模型列表
        """
        # 这里只是一个示例，实际实现需要使用HuggingFace API
        # 由于API调用需要token，这里只返回一个空列表
        logger.info("HuggingFace更新源需要API token，请在配置中添加token")
        return []
    
    def update_model(self, update_info: Dict[str, Any], callback: Optional[Callable[[str, float], None]] = None) -> bool:
        """
        更新模型。
        
        参数:
            update_info: 更新信息
            callback: 进度回调函数
            
        返回:
            是否成功
        """
        with self.lock:
            # 检查是否已在更新
            if update_info["name"] in self.updating_models:
                logger.warning(f"模型正在更新中: {update_info['name']}")
                return False
            
            # 标记为正在更新
            self.updating_models.add(update_info["name"])
            
            try:
                # 更新进度
                if callback:
                    callback(f"正在更新模型: {update_info['name']}", 0)
                
                # 获取模型路径
                model_path = update_info["path"]
                
                # 检查文件是否存在
                if not os.path.exists(model_path):
                    logger.error(f"模型文件不存在: {model_path}")
                    return False
                
                # 更新进度
                if callback:
                    callback(f"正在复制模型文件: {update_info['name']}", 20)
                
                # 添加到模型库
                model_type = ModelType.from_string(update_info.get("model_type", "other"))
                model = self.library.add_model(model_path, model_type)
                
                if not model:
                    logger.error(f"添加模型失败: {update_info['name']}")
                    return False
                
                # 更新进度
                if callback:
                    callback(f"模型更新完成: {update_info['name']}", 100)
                
                # 添加到更新历史
                self.update_history.append({
                    "name": update_info["name"],
                    "source": update_info["source"],
                    "time": datetime.datetime.now().isoformat(),
                    "success": True
                })
                
                logger.info(f"模型更新成功: {update_info['name']}")
                return True
            
            except Exception as e:
                logger.error(f"更新模型失败: {update_info['name']}, {e}")
                
                # 添加到更新历史
                self.update_history.append({
                    "name": update_info["name"],
                    "source": update_info["source"],
                    "time": datetime.datetime.now().isoformat(),
                    "success": False,
                    "error": str(e)
                })
                
                return False
            
            finally:
                # 移除正在更新标记
                self.updating_models.remove(update_info["name"])
    
    def get_update_sources(self) -> List[Dict[str, Any]]:
        """
        获取更新源列表。
        
        返回:
            更新源列表
        """
        return self.update_sources.copy()
    
    def get_update_history(self) -> List[Dict[str, Any]]:
        """
        获取更新历史。
        
        返回:
            更新历史
        """
        return self.update_history.copy()

# 全局模型更新器实例
_model_updater = None

def get_model_updater() -> ModelUpdater:
    """
    获取全局模型更新器实例。
    
    返回:
        模型更新器实例
    """
    global _model_updater
    
    if _model_updater is None:
        _model_updater = ModelUpdater()
    
    return _model_updater