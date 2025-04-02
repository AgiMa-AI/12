# OpenHands - 智能健康管家

OpenHands是一个智能健康管家系统，基于先进的大语言模型技术，为用户提供健康咨询、饮食指导和运动建议。

## 功能特点

- **中文自然语言处理**：专为中文用户设计，支持中文分词、词性标注、命名实体识别等
- **多模型支持**：集成多种开源大语言模型，包括健康领域和编程领域的专业模型
- **本地部署**：支持完全本地部署，保护用户隐私
- **易于使用**：简洁直观的Web界面，无需复杂设置
- **可扩展性**：支持添加新模型和功能

## 安装方法

### 方法一：使用可执行文件（推荐）

1. 从[发布页面](https://github.com/yourusername/OpenHands/releases)下载最新版本
2. 解压缩下载的文件
3. 双击`OpenHands.exe`启动程序
4. 在浏览器中访问 http://localhost:8000

### 方法二：从源代码安装

```bash
# 克隆代码库
git clone https://github.com/yourusername/OpenHands.git

# 进入目录
cd OpenHands

# 安装依赖
pip install -r requirements.txt

# 启动程序
python start_openhands.py
```

## 使用自定义模型

OpenHands支持使用自定义的GGUF格式模型。要使用自定义模型：

1. 将模型文件（.gguf格式）放在`models`目录中
2. 创建同名的JSON文件，描述模型信息（可选）
3. 启动程序时指定模型路径：`python start_openhands.py --local-model "models/your-model.gguf"`

## 命令行参数

```
--port PORT       指定HTTP服务器端口（默认8000）
--config FILE     指定配置文件路径（默认agent_config.json）
--local           使用本地模型
--local-model PATH 指定本地模型路径
--no-browser      不自动打开浏览器
--check-updates   检查更新
--auto-update     自动更新
```

## 构建可执行文件

要构建可执行文件，请运行：

```bash
python build_exe.py
```

可选参数：
```
--output DIR      指定输出目录（默认dist）
--onefile         生成单文件可执行文件
--noconsole       不显示控制台窗口
--icon PATH       指定图标文件路径
```

## 许可证

本项目采用MIT许可证。详见[LICENSE](LICENSE)文件。

## 贡献

欢迎贡献代码、报告问题或提出建议。请查看[贡献指南](CONTRIBUTING.md)了解更多信息。