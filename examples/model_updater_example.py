#!/usr/bin/env python3
"""
模型更新器示例。

本脚本展示了如何使用模型更新器检查和更新模型。
"""

import os
import sys
import logging
import argparse
import time
from typing import Dict, List, Any

# 添加父目录到路径，以便导入openhands
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openhands.model_library import (
    get_model_library,
    ModelType,
    get_model_updater
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("model_updater_example")

def progress_callback(message: str, progress: float) -> None:
    """
    进度回调函数。
    
    参数:
        message: 消息
        progress: 进度（0-100）
    """
    # 计算进度条
    bar_length = 40
    filled_length = int(bar_length * progress / 100)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    
    # 打印进度
    print(f"\r{message} [{bar}] {progress:.1f}%", end='')
    
    # 如果完成，换行
    if progress >= 100:
        print()

def main():
    """主函数。"""
    parser = argparse.ArgumentParser(description="模型更新器示例")
    parser.add_argument("--add-source", help="添加更新源", action="store_true")
    parser.add_argument("--source-name", help="更新源名称")
    parser.add_argument("--source-type", help="更新源类型")
    parser.add_argument("--source-url", help="更新源URL")
    parser.add_argument("--check", help="检查更新", action="store_true")
    parser.add_argument("--update", help="更新模型", action="store_true")
    parser.add_argument("--model-type", help="模型类型")
    parser.add_argument("--list-sources", help="列出更新源", action="store_true")
    
    args = parser.parse_args()
    
    try:
        # 获取模型库
        library = get_model_library()
        
        # 获取模型更新器
        updater = get_model_updater()
        
        # 列出更新源
        if args.list_sources:
            print("\n=== 更新源列表 ===")
            for i, source in enumerate(updater.get_update_sources()):
                print(f"{i+1}. {source['name']} ({source['type']})")
                print(f"   URL: {source['url']}")
                print(f"   状态: {'启用' if source['enabled'] else '禁用'}")
                print()
        
        # 添加更新源
        if args.add_source:
            if not args.source_name or not args.source_type or not args.source_url:
                print("添加更新源需要指定名称、类型和URL")
                return
            
            success = updater.add_update_source(args.source_name, args.source_type, args.source_url)
            
            if success:
                print(f"成功添加更新源: {args.source_name}")
            else:
                print(f"添加更新源失败: {args.source_name}")
        
        # 检查更新
        if args.check:
            print("\n=== 检查更新 ===")
            
            # 确定模型类型
            model_type = None
            if args.model_type:
                model_type = ModelType.from_string(args.model_type)
                print(f"检查模型类型: {model_type.value}")
            
            # 检查更新
            updates = updater.check_for_updates(model_type, progress_callback)
            
            if updates:
                print(f"\n找到{len(updates)}个可更新模型:")
                for i, update in enumerate(updates):
                    print(f"{i+1}. {update['name']}")
                    print(f"   来源: {update['source']}")
                    print(f"   路径: {update['path']}")
                    print(f"   大小: {update['size'] / (1024 * 1024):.2f} MB")
                    if update['existing_size'] > 0:
                        print(f"   现有大小: {update['existing_size'] / (1024 * 1024):.2f} MB")
                    print(f"   类型: {update.get('model_type', 'other')}")
                    print()
                
                # 更新模型
                if args.update:
                    print("\n=== 更新模型 ===")
                    
                    for i, update in enumerate(updates):
                        print(f"\n正在更新 {i+1}/{len(updates)}: {update['name']}")
                        
                        success = updater.update_model(update, progress_callback)
                        
                        if success:
                            print(f"成功更新模型: {update['name']}")
                        else:
                            print(f"更新模型失败: {update['name']}")
                        
                        # 暂停一下，避免过快更新
                        time.sleep(1)
            else:
                print("\n没有找到可更新的模型")
        
        # 打印更新历史
        update_history = updater.get_update_history()
        if update_history:
            print("\n=== 更新历史 ===")
            for i, history in enumerate(update_history):
                print(f"{i+1}. {history['name']}")
                print(f"   来源: {history['source']}")
                print(f"   时间: {history['time']}")
                print(f"   结果: {'成功' if history['success'] else '失败'}")
                if not history['success'] and 'error' in history:
                    print(f"   错误: {history['error']}")
                print()
        
    except KeyboardInterrupt:
        print("\n已中断")
    
    except Exception as e:
        logger.error(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()