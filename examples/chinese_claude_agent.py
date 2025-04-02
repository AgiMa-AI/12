#!/usr/bin/env python3
"""
中文增强的Claude Agent示例。

本脚本展示了如何使用中文增强的Claude Agent，包括中文解析、中文增强等功能。
"""

import os
import sys
import json
import logging
import argparse
from typing import Dict, List, Any

# 添加父目录到路径，以便导入openhands
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openhands.claude_agent import (
    ChineseClaudeAgent,
    get_chinese_claude_agent,
    initialize_environment
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("chinese_claude_agent")

def print_analysis(analysis: Dict[str, Any]) -> None:
    """
    打印分析结果。
    
    参数:
        analysis: 分析结果
    """
    print("\n=== 文本分析 ===")
    
    # 打印原始文本
    print(f"原始文本: {analysis['original_text']}")
    
    # 打印纠错后的文本
    if analysis['original_text'] != analysis['parse_result']['corrected_text']:
        print(f"纠错后文本: {analysis['parse_result']['corrected_text']}")
    
    # 打印分词结果
    print(f"分词结果: {', '.join(analysis['parse_result']['tokens'])}")
    
    # 打印关键词
    print("关键词:")
    for keyword, weight in analysis['parse_result']['keywords']:
        print(f"  {keyword}: {weight:.4f}")
    
    # 打印意图
    print("意图:")
    for intent, confidence in analysis['parse_result']['intent'].items():
        print(f"  {intent}: {confidence:.4f}")
    
    # 打印实体
    if analysis['parse_result']['extracted_entities']:
        print("实体:")
        for entity_type, entities in analysis['parse_result']['extracted_entities'].items():
            print(f"  {entity_type}:")
            for entity in entities:
                print(f"    {entity['value']}")
    
    # 打印情感
    sentiment = analysis['parse_result']['sentiment']
    sentiment_text = "积极" if sentiment > 0 else "消极" if sentiment < 0 else "中性"
    print(f"情感: {sentiment_text} ({sentiment:.4f})")
    
    # 打印增强结果
    if analysis['enhancements']:
        print("\n=== 增强结果 ===")
        for rule, enhancement in analysis['enhancements'].items():
            print(f"{rule}:")
            if rule == "expand_abbreviations" and "expanded" in enhancement:
                for abbr, full in enhancement["expanded"].items():
                    print(f"  {abbr} -> {full}")
            elif rule == "explain_idioms" and "explained" in enhancement:
                for idiom, explanation in enhancement["explained"].items():
                    print(f"  {idiom}: {explanation}")
            elif rule == "add_pinyin" and "pinyin" in enhancement:
                print(f"  拼音: {enhancement['pinyin']}")
            elif rule == "add_traditional" and "traditional" in enhancement:
                print(f"  繁体: {enhancement['traditional']}")
            elif rule == "add_context" and "context" in enhancement:
                for context_type, context_items in enhancement["context"].items():
                    print(f"  {context_type}:")
                    for item in context_items:
                        print(f"    {item['value']}")

def interactive_mode(agent: ChineseClaudeAgent) -> None:
    """
    交互模式。
    
    参数:
        agent: 中文增强的Claude Agent实例
    """
    print("\n=== 中文增强的Claude Agent交互模式 ===")
    print("输入问题与AI对话，输入'exit'或'quit'退出，输入'analyze'分析上一次回答")
    
    last_response = None
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n> ")
            
            # 检查是否退出
            if user_input.lower() in ["exit", "quit", "退出", "再见"]:
                print("再见！")
                break
            
            # 检查是否分析上一次回答
            if user_input.lower() in ["analyze", "分析"]:
                if last_response:
                    analysis = agent.analyze_text(last_response)
                    print_analysis(analysis)
                else:
                    print("没有可分析的回答")
                continue
            
            # 处理用户输入
            response = agent.process_input(user_input)
            
            # 打印回答
            if hasattr(response, "content"):
                print(f"\n{response.content}")
                last_response = response.content
            else:
                print("\n无法获取回答")
        
        except KeyboardInterrupt:
            print("\n已中断")
            break
        
        except Exception as e:
            logger.error(f"错误: {e}")
            print(f"\n发生错误: {e}")

def main():
    """主函数。"""
    parser = argparse.ArgumentParser(description="中文增强的Claude Agent示例")
    parser.add_argument("--config", help="配置文件路径", default="agent_config.json")
    parser.add_argument("--local", help="使用本地模型", action="store_true")
    parser.add_argument("--local-model", help="本地模型路径", default=None)
    parser.add_argument("--analyze", help="要分析的文本", default=None)
    
    args = parser.parse_args()
    
    # 设置本地模型环境变量
    if args.local:
        os.environ["USE_LOCAL_MODEL"] = "true"
    if args.local_model:
        os.environ["LOCAL_MODEL_PATH"] = args.local_model
    
    try:
        # 初始化环境
        env = initialize_environment(
            config_file=args.config,
            with_examples=True
        )
        
        # 获取中文增强的Claude Agent
        agent = get_chinese_claude_agent(config_file=args.config)
        
        # 如果提供了要分析的文本，则分析文本
        if args.analyze:
            analysis = agent.analyze_text(args.analyze)
            print_analysis(analysis)
            return
        
        # 否则进入交互模式
        interactive_mode(agent)
        
    except Exception as e:
        logger.error(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()