import os
import sys
import subprocess
import threading
import webbrowser
import time
import signal
import psutil
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socket
from pathlib import Path
import json

def get_resource_path(relative_path):
    """获取资源文件的路径"""
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def find_free_port():
    """找到一个可用的端口"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port

def create_runtime_config(backend_port, frontend_port):
    """创建运行时配置文件"""
    try:
        config = {
            'backendUrl': f'http://localhost:{backend_port}',
            'frontendPort': frontend_port
        }
        frontend_path = get_resource_path('frontend/build')
        config_path = os.path.join(frontend_path, 'config.json')
        
        # 确保目录存在
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f)
            
        print(f"Runtime config created at: {config_path}")
        
    except Exception as e:
        print(f"Error creating runtime config: {str(e)}")
        raise

def run_backend():
    """运行后端服务"""
    env_path = get_resource_path('.env')
    if not os.path.exists(env_path):
        print(f"Error: Cannot find .env file at {env_path}")
        sys.exit(1)
    
    os.environ['ENV_FILE'] = env_path
    from backend.app import app
    port = find_free_port()
    os.environ['FLASK_PORT'] = str(port)
    print(f"Backend running on port {port}")
    return port, app

def run_frontend(port):
    """运行前端服务"""
    try:
        frontend_path = get_resource_path('frontend/build')
        if not os.path.exists(frontend_path):
            raise Exception(f"Frontend build directory not found at: {frontend_path}")
            
        print(f"Serving frontend from: {frontend_path}")
        os.chdir(frontend_path)
        
        class Handler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=frontend_path, **kwargs)
            
            def do_GET(self):
                if self.path == '/':
                    self.path = '/index.html'
                return super().do_GET()

        httpd = HTTPServer(('', port), Handler)
        print(f"Frontend running on port {port}")
        httpd.serve_forever()
        
    except Exception as e:
        print(f"Error starting frontend server: {str(e)}")
        raise

def main():
    try:
        # 启动后端
        backend_port, app = run_backend()
        backend_thread = threading.Thread(
            target=app.run,
            kwargs={'port': backend_port},
            daemon=True
        )
        backend_thread.start()
        
        # 等待后端启动
        time.sleep(2)
        
        # 启动前端
        frontend_port = find_free_port()
        
        # 创建运行时配置
        create_runtime_config(backend_port, frontend_port)
        
        frontend_thread = threading.Thread(
            target=run_frontend,
            args=(frontend_port,),
            daemon=True
        )
        frontend_thread.start()
        
        # 打开浏览器
        webbrowser.open(f'http://localhost:{frontend_port}')
        
        print(f"Application running at http://localhost:{frontend_port}")
        print(f"Backend API available at http://localhost:{backend_port}")
        
        while True:
            time.sleep(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        input("Press Enter to exit...")
        sys.exit(1)
    except KeyboardInterrupt:
        print("正在关闭服务...")
        sys.exit(0)

if __name__ == '__main__':
    main() 