#!/usr/bin/env python3
"""
构建可执行文件。

本脚本使用PyInstaller构建OpenHands的可执行文件。
"""

import os
import sys
import shutil
import subprocess
import argparse
import platform

def check_requirements():
    """检查是否安装了必要的依赖"""
    try:
        import PyInstaller
        print("PyInstaller已安装")
    except ImportError:
        print("正在安装PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    # 检查其他依赖
    requirements = [
        "jieba",
        "requests",
        "tqdm",
        "numpy",
        "pillow"
    ]
    
    for req in requirements:
        try:
            __import__(req)
            print(f"{req}已安装")
        except ImportError:
            print(f"正在安装{req}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", req])

def build_executable(output_dir, one_file=False, console=True, icon=None):
    """构建可执行文件"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建命令
    cmd = [
        "pyinstaller",
        "--clean",
        "--name", "OpenHands",
        "--distpath", output_dir
    ]
    
    # 添加图标
    if icon and os.path.exists(icon):
        cmd.extend(["--icon", icon])
    
    # 单文件模式
    if one_file:
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")
    
    # 控制台模式
    if not console:
        cmd.append("--noconsole")
    
    # 添加数据文件
    data_files = [
        ("openhands/chinese_nlp/data/*.json", "openhands/chinese_nlp/data"),
        ("models/*.gguf", "models"),
        ("models/*.json", "models"),
        ("web/*", "web")
    ]
    
    for src, dst in data_files:
        cmd.extend(["--add-data", f"{src}{os.pathsep}{dst}"])
    
    # 添加主脚本
    cmd.append("start_openhands.py")
    
    # 执行构建
    print("正在构建可执行文件...")
    print(" ".join(cmd))
    subprocess.check_call(cmd)
    
    print(f"构建完成，输出目录: {output_dir}")

def copy_models(output_dir, one_file):
    """复制模型文件"""
    # 确定目标目录
    if one_file:
        target_dir = os.path.join(output_dir, "models")
    else:
        target_dir = os.path.join(output_dir, "OpenHands", "models")
    
    # 创建目录
    os.makedirs(target_dir, exist_ok=True)
    
    # 复制模型文件
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    if os.path.exists(models_dir):
        for file in os.listdir(models_dir):
            if file.endswith((".gguf", ".json")):
                src = os.path.join(models_dir, file)
                dst = os.path.join(target_dir, file)
                print(f"复制模型文件: {src} -> {dst}")
                shutil.copy2(src, dst)

def create_readme(output_dir, one_file):
    """创建README文件"""
    # 确定目标目录
    if one_file:
        target_dir = output_dir
    else:
        target_dir = os.path.join(output_dir, "OpenHands")
    
    # 创建README文件
    readme_path = os.path.join(target_dir, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write("""OpenHands - 智能健康管家

使用说明：
1. 双击OpenHands.exe启动程序
2. 在浏览器中访问 http://localhost:8000
3. 开始与您的智能健康管家对话

命令行参数：
--port PORT       指定HTTP服务器端口（默认8000）
--no-browser      不自动打开浏览器
--local-model PATH 指定本地模型路径

如果您想使用自己的模型，请将模型文件（.gguf格式）放在models目录中。

祝您使用愉快！
""")
    
    print(f"创建README文件: {readme_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="构建OpenHands可执行文件")
    parser.add_argument("--output", help="输出目录", default="dist")
    parser.add_argument("--onefile", help="生成单文件可执行文件", action="store_true")
    parser.add_argument("--noconsole", help="不显示控制台窗口", action="store_true")
    parser.add_argument("--icon", help="图标文件路径")
    
    args = parser.parse_args()
    
    # 检查依赖
    check_requirements()
    
    # 构建可执行文件
    build_executable(args.output, args.onefile, not args.noconsole, args.icon)
    
    # 复制模型文件
    copy_models(args.output, args.onefile)
    
    # 创建README文件
    create_readme(args.output, args.onefile)
    
    print("构建完成！")

if __name__ == "__main__":
    main()