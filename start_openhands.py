#!/usr/bin/env python3
"""
启动OpenHands智能健康管家。

本脚本启动OpenHands智能健康管家，包括GUI界面和后端服务。
"""

import os
import sys
import logging
import argparse
import threading
import time
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler

# 添加当前目录到路径，以便导入openhands
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from openhands.claude_agent import (
    initialize_environment,
    get_chinese_claude_agent
)
from openhands.model_library import (
    get_model_library,
    get_model_updater,
    get_gguf_model,
    ModelType
)
from openhands.context_awareness import (
    get_context_manager
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("openhands")

class OpenHandsHTTPHandler(SimpleHTTPRequestHandler):
    """OpenHands HTTP处理器。"""
    
    def __init__(self, *args, **kwargs):
        # 确保web目录存在
        web_dir = os.path.join(os.path.dirname(__file__), "web")
        if not os.path.exists(web_dir):
            os.makedirs(web_dir, exist_ok=True)
        
        # 设置目录
        super().__init__(*args, directory=web_dir, **kwargs)
    
    def log_message(self, format, *args):
        """重写日志消息方法，使用logger。"""
        logger.info("%s - - [%s] %s" % (
            self.address_string(),
            self.log_date_time_string(),
            format % args
        ))
        
    def end_headers(self):
        """添加CORS头，允许跨域访问"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

class OpenHandsAPIHandler(SimpleHTTPRequestHandler):
    """OpenHands API处理器。"""
    
    def __init__(self, *args, agent=None, **kwargs):
        self.agent = agent
        # 确保web目录存在
        web_dir = os.path.join(os.path.dirname(__file__), "web")
        if not os.path.exists(web_dir):
            os.makedirs(web_dir, exist_ok=True)
        
        # 设置目录
        super().__init__(*args, directory=web_dir, **kwargs)
    
    def log_message(self, format, *args):
        """重写日志消息方法，使用logger。"""
        logger.info("%s - - [%s] %s" % (
            self.address_string(),
            self.log_date_time_string(),
            format % args
        ))
    
    def end_headers(self):
        """添加CORS头，允许跨域访问"""
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        """处理OPTIONS请求"""
        self.send_response(200)
        self.end_headers()
    
    def do_POST(self):
        """处理POST请求"""
        if self.path == '/api/chat':
            # 获取请求内容长度
            content_length = int(self.headers['Content-Length'])
            # 读取请求内容
            post_data = self.rfile.read(content_length)
            # 解析JSON数据
            import json
            try:
                data = json.loads(post_data.decode('utf-8'))
                message = data.get('message', '')
                
                # 处理消息
                if self.agent:
                    response = self.agent.process_input(message)
                    if hasattr(response, 'content'):
                        result = {
                            'status': 'success',
                            'message': response.content
                        }
                    else:
                        result = {
                            'status': 'error',
                            'message': '无法获取回复'
                        }
                else:
                    # 如果没有agent，返回模拟回复
                    result = {
                        'status': 'success',
                        'message': f'您好！我收到了您的消息：{message}'
                    }
                
                # 发送响应
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode('utf-8'))
            except Exception as e:
                # 发送错误响应
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    'status': 'error',
                    'message': f'处理请求时出错: {str(e)}'
                }).encode('utf-8'))
        else:
            # 对于其他路径，返回404
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'status': 'error',
                'message': '未找到请求的资源'
            }).encode('utf-8'))

def create_handler_class(agent):
    """创建处理器类，并传入agent"""
    return lambda *args, **kwargs: OpenHandsAPIHandler(*args, agent=agent, **kwargs)

def start_http_server(port=8000, agent=None):
    """
    启动HTTP服务器。
    
    参数:
        port: 端口号
        agent: 智能体实例
    """
    server_address = ('', port)
    handler_class = create_handler_class(agent)
    httpd = HTTPServer(server_address, handler_class)
    logger.info(f"启动HTTP服务器在端口 {port}")
    httpd.serve_forever()

def open_browser(port=8000, delay=1.0):
    """
    打开浏览器。
    
    参数:
        port: 端口号
        delay: 延迟时间（秒）
    """
    time.sleep(delay)
    webbrowser.open(f"http://localhost:{port}")

def check_for_updates(auto_update=False):
    """
    检查更新。
    
    参数:
        auto_update: 是否自动更新
    """
    try:
        # 获取模型更新器
        updater = get_model_updater()
        
        # 检查更新
        logger.info("检查模型更新...")
        updates = updater.check_for_updates()
        
        if updates:
            logger.info(f"找到{len(updates)}个可更新模型")
            
            # 自动更新
            if auto_update:
                for update in updates:
                    logger.info(f"正在更新模型: {update['name']}")
                    updater.update_model(update)
        else:
            logger.info("没有可更新的模型")
    
    except Exception as e:
        logger.error(f"检查更新失败: {e}")

def main():
    """主函数。"""
    parser = argparse.ArgumentParser(description="启动OpenHands智能健康管家")
    parser.add_argument("--port", help="HTTP服务器端口", type=int, default=8000)
    parser.add_argument("--config", help="配置文件路径", default="agent_config.json")
    parser.add_argument("--local", help="使用本地模型", action="store_true")
    parser.add_argument("--local-model", help="本地模型路径", default=None)
    parser.add_argument("--no-browser", help="不自动打开浏览器", action="store_true")
    parser.add_argument("--check-updates", help="检查更新", action="store_true")
    parser.add_argument("--auto-update", help="自动更新", action="store_true")
    
    args = parser.parse_args()
    
    # 设置本地模型环境变量
    if args.local:
        os.environ["USE_LOCAL_MODEL"] = "true"
    if args.local_model:
        os.environ["LOCAL_MODEL_PATH"] = args.local_model
    
    try:
        # 检查更新
        if args.check_updates or args.auto_update:
            update_thread = threading.Thread(target=check_for_updates, args=(args.auto_update,))
            update_thread.daemon = True
            update_thread.start()
        
        # 初始化环境
        env = initialize_environment(
            config_file=args.config,
            with_examples=True
        )
        
        # 初始化中文增强的Claude Agent
        agent = get_chinese_claude_agent(config_file=args.config)
        
        # 获取模型库
        library = get_model_library()
        
        # 暂时不使用上下文管理器
        # context_manager = get_context_manager()
        
        # 检查web目录是否存在
        web_dir = os.path.join(os.path.dirname(__file__), "web")
        if not os.path.exists(web_dir):
            os.makedirs(web_dir, exist_ok=True)
            
            # 创建简单的index.html
            with open(os.path.join(web_dir, "index.html"), "w", encoding="utf-8") as f:
                f.write("""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OpenHands - 智能健康管家</title>
    <style>
        body {
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
            color: #333;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            background-color: #4CAF50;
            color: white;
            padding: 20px;
            text-align: center;
            border-radius: 5px 5px 0 0;
        }
        .chat-container {
            background-color: white;
            border-radius: 0 0 5px 5px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
        }
        .messages {
            height: 400px;
            overflow-y: auto;
            margin-bottom: 20px;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        .message {
            margin-bottom: 10px;
            padding: 10px;
            border-radius: 5px;
        }
        .user-message {
            background-color: #e3f2fd;
            margin-left: 20%;
            text-align: right;
        }
        .bot-message {
            background-color: #f1f1f1;
            margin-right: 20%;
        }
        .input-container {
            display: flex;
        }
        #user-input {
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px 0 0 5px;
            font-size: 16px;
        }
        #send-button {
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 0 5px 5px 0;
            cursor: pointer;
            font-size: 16px;
        }
        #send-button:hover {
            background-color: #45a049;
        }
        .features {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-top: 20px;
        }
        .feature {
            flex: 1;
            min-width: 200px;
            background-color: white;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .feature h3 {
            margin-top: 0;
            color: #4CAF50;
        }
        footer {
            text-align: center;
            margin-top: 20px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>OpenHands - 智能健康管家</h1>
            <p>您的健康助手，随时为您服务</p>
        </header>
        
        <div class="chat-container">
            <div class="messages" id="messages">
                <div class="message bot-message">
                    您好！我是您的智能健康管家，有什么可以帮助您的吗？
                </div>
            </div>
            <div class="input-container">
                <input type="text" id="user-input" placeholder="请输入您的问题..." autofocus>
                <button id="send-button">发送</button>
            </div>
        </div>
        
        <div class="features">
            <div class="feature">
                <h3>健康咨询</h3>
                <p>提供健康知识、症状分析和生活建议</p>
            </div>
            <div class="feature">
                <h3>饮食指导</h3>
                <p>根据您的健康状况提供个性化饮食建议</p>
            </div>
            <div class="feature">
                <h3>运动计划</h3>
                <p>制定适合您的运动计划，帮助您保持健康</p>
            </div>
        </div>
        
        <footer>
            <p>© 2023 OpenHands - 智能健康管家</p>
        </footer>
    </div>

    <script>
        document.addEventListener('DOMContentLoaded', function() {
            const messagesContainer = document.getElementById('messages');
            const userInput = document.getElementById('user-input');
            const sendButton = document.getElementById('send-button');
            
            // 禁用按钮函数
            function disableButton() {
                sendButton.disabled = true;
                sendButton.textContent = '处理中...';
                sendButton.style.backgroundColor = '#999';
            }
            
            // 启用按钮函数
            function enableButton() {
                sendButton.disabled = false;
                sendButton.textContent = '发送';
                sendButton.style.backgroundColor = '#4CAF50';
            }
            
            function addMessage(text, isUser) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${isUser ? 'user-message' : 'bot-message'}`;
                
                // 支持换行和简单的Markdown格式
                const formattedText = text
                    .replace(/\n/g, '<br>')
                    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                    .replace(/\*(.*?)\*/g, '<em>$1</em>')
                    .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
                
                messageDiv.innerHTML = formattedText;
                messagesContainer.appendChild(messageDiv);
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            }
            
            async function sendMessage() {
                const message = userInput.value.trim();
                if (message) {
                    addMessage(message, true);
                    userInput.value = '';
                    disableButton();
                    
                    try {
                        // 发送到后端API
                        const response = await fetch('/api/chat', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({ message: message })
                        });
                        
                        if (response.ok) {
                            const data = await response.json();
                            if (data.status === 'success') {
                                addMessage(data.message, false);
                            } else {
                                addMessage(`错误: ${data.message}`, false);
                            }
                        } else {
                            addMessage('服务器错误，请稍后再试。', false);
                        }
                    } catch (error) {
                        console.error('Error:', error);
                        addMessage('抱歉，发生了网络错误，请检查您的连接。', false);
                    } finally {
                        enableButton();
                    }
                }
            }
            
            sendButton.addEventListener('click', sendMessage);
            
            userInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault(); // 防止换行
                    sendMessage();
                }
            });
            
            // 添加Shift+Enter支持多行输入
            userInput.addEventListener('keydown', function(e) {
                if (e.key === 'Enter' && e.shiftKey) {
                    // 允许Shift+Enter换行
                }
            });
            
            // 自动聚焦输入框
            userInput.focus();
        });
    </script>
</body>
</html>
""")
        
        # 启动HTTP服务器
        server_thread = threading.Thread(target=start_http_server, args=(args.port, agent), daemon=True)
        server_thread.start()
        
        # 打开浏览器
        if not args.no_browser:
            browser_thread = threading.Thread(target=open_browser, args=(args.port,))
            browser_thread.start()
        
        logger.info(f"OpenHands智能健康管家已启动，请访问 http://localhost:{args.port}")
        
        # 打印系统信息
        import datetime
        now = datetime.datetime.now()
        hour = now.hour
        if 5 <= hour < 12:
            greeting = "早上好"
        elif 12 <= hour < 17:
            greeting = "下午好"
        elif 17 <= hour < 22:
            greeting = "晚上好"
        else:
            greeting = "您好"
        logger.info(f"{greeting}！现在是{now.hour}:{now.minute}")
        
        # 保持主线程运行
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("服务器已停止")
        
    except Exception as e:
        logger.error(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()