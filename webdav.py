import os
from wsgidav.wsgidav_app import WsgiDAVApp
from wsgidav.fs_dav_provider import FilesystemProvider
from wsgidav.dc.simple_dc import SimpleDomainController
from pathlib import Path
import threading
import logging

# 配置 WebDAV 服务器
class WebDAVServer:
    def __init__(self, storage_path, host="0.0.0.0", port=5889, auth_enabled=True, username="admin", password="admin"):
        self.storage_path = storage_path
        self.host = host
        self.port = port
        self.auth_enabled = auth_enabled
        self.username = username
        self.password = password
        self.server_thread = None
        self.server = None
        self.logger = logging.getLogger("webdav")
        
    def get_config(self):
        config = {
            "provider_mapping": {
                "/": FilesystemProvider(str(self.storage_path))
            },
            "http_authenticator": {
                "domain_controller": None,
            },
            "simple_dc": {
                "user_mapping": {}
            },
            "verbose": 3,
            # 修改日志配置
            "logging": {
                "enable_loggers": []
            },
            "property_manager": True,
            # 修改锁配置
            "lock_storage": True,
        }
        
        # 配置身份验证
        if self.auth_enabled:
            config["http_authenticator"]["domain_controller"] = SimpleDomainController
            config["simple_dc"]["user_mapping"] = {
                "*": {
                    self.username: {
                        "password": self.password,
                        "description": "WebDAV Admin",
                        "roles": ["admin"]
                    }
                }
            }
            
        return config
    
    def start(self):
        """启动 WebDAV 服务器"""
        if self.server_thread and self.server_thread.is_alive():
            self.logger.info("WebDAV 服务器已经在运行")
            return
            
        config = self.get_config()
        app = WsgiDAVApp(config)
        
        def run_server():
            from cheroot import wsgi
            server = wsgi.Server((self.host, self.port), app)
            self.server = server
            self.logger.info(f"WebDAV 服务器运行在 http://{self.host}:{self.port}")
            try:
                server.start()
            except KeyboardInterrupt:
                self.logger.info("WebDAV 服务器关闭")
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self.logger.info(f"WebDAV 服务器线程已启动 (端口: {self.port})")
        
    def stop(self):
        """停止 WebDAV 服务器"""
        if self.server:
            self.server.stop()
            self.logger.info("WebDAV 服务器已停止")
        if self.server_thread and self.server_thread.is_alive():
            self.server_thread.join(timeout=5)
            self.logger.info("WebDAV 服务器线程已停止")

# 配置 WebDAV 服务器函数
def configure_webdav(storage_path, config_path="./config"):
    # 确保配置目录存在
    os.makedirs(config_path, exist_ok=True)
    
    # 从配置文件读取 WebDAV 设置
    config_file = Path(config_path) / "webdav.conf"
    config = {
        "enabled": True,
        "port": 5889,
        "auth_enabled": True,
        "username": "admin",
        "password": "admin"
    }
    
    # 如果配置文件存在，读取配置
    if config_file.exists():
        try:
            import json
            with open(config_file, "r", encoding="utf-8") as f:
                saved_config = json.load(f)
                # 更新配置，保留默认值
                for k, v in saved_config.items():
                    if k in config:
                        config[k] = v
        except Exception as e:
            logging.error(f"读取 WebDAV 配置失败: {str(e)}")
    
    # 保存当前配置
    try:
        import json
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"保存 WebDAV 配置失败: {str(e)}")
    
    # 如果启用了 WebDAV，创建并返回服务器实例
    if config["enabled"]:
        return WebDAVServer(
            storage_path=storage_path,
            port=config["port"],
            auth_enabled=config["auth_enabled"],
            username=config["username"],
            password=config["password"]
        )
    
    return None