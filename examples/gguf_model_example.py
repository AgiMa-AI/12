#!/usr/bin/env python3
"""
GGUF模型示例。

本脚本展示了如何使用GGUF模型进行文本生成和聊天。
"""

import os
import sys
import logging
import argparse
from typing import Dict, List, Any

# 添加父目录到路径，以便导入openhands
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openhands.model_library.gguf_loader import get_gguf_model

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("gguf_model_example")

def interactive_chat(model_path: str, context_window: int = 4096, temperature: float = 0.7):
    """
    交互式聊天。
    
    参数:
        model_path: 模型文件路径
        context_window: 上下文窗口大小
        temperature: 温度
    """
    # 获取模型
    model = get_gguf_model(model_path, context_window)
    
    # 加载模型
    if not model.is_loaded():
        if not model.load():
            print("模型加载失败，退出")
            return
    
    # 打印模型信息
    model_info = model.get_model_info()
    print("\n=== 模型信息 ===")
    print(f"名称: {model_info.get('name', '未知')}")
    print(f"路径: {model_info['path']}")
    print(f"大小: {model_info['size'] / (1024 * 1024):.2f} MB")
    print(f"上下文窗口: {model_info['context_window']}")
    if "parameters" in model_info:
        print(f"参数数量: {model_info['parameters'] / 1_000_000_000:.1f}B")
    
    # 初始化消息历史
    messages = []
    
    # 添加系统消息
    system_message = "你是一个智能助手，使用中文回答用户的问题。请保持回答简洁、准确、有帮助。"
    messages.append({"role": "system", "content": system_message})
    
    print("\n=== 开始聊天 ===")
    print("输入问题与AI对话，输入'exit'或'quit'退出")
    
    while True:
        # 获取用户输入
        user_input = input("\n> ")
        
        # 检查是否退出
        if user_input.lower() in ["exit", "quit", "退出", "再见"]:
            print("再见！")
            break
        
        # 添加用户消息
        messages.append({"role": "user", "content": user_input})
        
        # 生成回复
        print("生成中...")
        response = model.chat(messages, temperature=temperature)
        
        # 打印回复
        print(f"\n{response['content']}")
        
        # 添加助手消息
        messages.append({"role": "assistant", "content": response["content"]})

def main():
    """主函数。"""
    parser = argparse.ArgumentParser(description="GGUF模型示例")
    parser.add_argument("--model", help="模型文件路径", required=True)
    parser.add_argument("--context-window", help="上下文窗口大小", type=int, default=4096)
    parser.add_argument("--temperature", help="温度", type=float, default=0.7)
    
    args = parser.parse_args()
    
    try:
        # 检查模型文件是否存在
        if not os.path.exists(args.model):
            print(f"模型文件不存在: {args.model}")
            return
        
        # 启动交互式聊天
        interactive_chat(args.model, args.context_window, args.temperature)
        
    except KeyboardInterrupt:
        print("\n已中断")
    
    except Exception as e:
        logger.error(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()