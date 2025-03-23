import os
import json
from pathlib import Path
from webdav3.client import Client
import logging
import shutil
import tempfile
import hashlib
import time

logger = logging.getLogger("webdav_client")

# 存储WebDAV连接配置的文件
WEBDAV_CONNECTIONS_FILE = Path("./config/webdav_connections.json")

# 确保配置目录存在
Path("./config").mkdir(parents=True, exist_ok=True)

class WebDAVConnection:
    def __init__(self, name, url, username, password, folder="/", enabled=True):
        self.name = name
        self.url = url
        self.username = username
        self.password = password
        self.folder = folder
        self.enabled = enabled
        self.client = None
        
    def to_dict(self):
        """将连接转换为字典（用于保存）"""
        return {
            "name": self.name,
            "url": self.url,
            "username": self.username,
            "password": self.password,
            "folder": self.folder,
            "enabled": self.enabled
        }
        
    @classmethod
    def from_dict(cls, data):
        """从字典创建连接对象"""
        return cls(
            name=data.get("name", ""),
            url=data.get("url", ""),
            username=data.get("username", ""),
            password=data.get("password", ""),
            folder=data.get("folder", "/"),
            enabled=data.get("enabled", True)
        )
        
    def connect(self):
        """连接到WebDAV服务器"""
        if self.client is not None:
            return self.client
            
        options = {
            'webdav_hostname': self.url,
            'webdav_login': self.username,
            'webdav_password': self.password,
            'disable_check': True  # 避免一些兼容性问题
        }
        
        try:
            self.client = Client(options)
            # 测试连接是否成功
            self.client.list()
            logger.info(f"成功连接到WebDAV服务器: {self.name} ({self.url})")
            return self.client
        except Exception as e:
            logger.error(f"连接WebDAV服务器失败: {self.name} ({self.url}) - {str(e)}")
            self.client = None
            raise
            
    def disconnect(self):
        """断开WebDAV连接"""
        self.client = None
        
    def list_files(self, path=None):
        """列出路径下的文件和文件夹"""
        if path is None:
            path = self.folder
            
        try:
            client = self.connect()
            items = client.list(path)
            
            # 过滤掉当前目录
            items = [item for item in items if item != '.' and item != './']
            
            # 格式化结果
            result = []
            for item in items:
                # 移除结尾的斜杠，确保路径格式一致
                item_name = item.rstrip('/')
                item_path = os.path.join(path, item_name).replace('\\', '/')
                
                # 检查是否是目录
                is_dir = item.endswith('/')
                
                # 获取文件信息
                try:
                    info = client.info(item_path)
                    size = int(info.get("size", 0))
                    modified = info.get("modified", "")
                except:
                    size = 0
                    modified = ""
                
                result.append({
                    "name": item_name,
                    "path": item_path,
                    "is_dir": is_dir,
                    "size": size,
                    "modified": modified,
                    "source": "webdav",
                    "connection": self.name
                })
                
            return result
        except Exception as e:
            logger.error(f"列出WebDAV文件失败: {path} - {str(e)}")
            raise
            
    def download_file(self, remote_path, local_path):
        """下载文件到本地路径"""
        try:
            client = self.connect()
            client.download_sync(remote_path=remote_path, local_path=local_path)
            return True
        except Exception as e:
            logger.error(f"从WebDAV下载文件失败: {remote_path} -> {local_path} - {str(e)}")
            raise
            
    def upload_file(self, local_path, remote_path):
        """上传文件到WebDAV服务器"""
        try:
            client = self.connect()
            client.upload_sync(remote_path=remote_path, local_path=local_path)
            return True
        except Exception as e:
            logger.error(f"上传文件到WebDAV失败: {local_path} -> {remote_path} - {str(e)}")
            raise
            
    def create_directory(self, path):
        """在WebDAV上创建目录"""
        try:
            client = self.connect()
            client.mkdir(path)
            return True
        except Exception as e:
            logger.error(f"在WebDAV上创建目录失败: {path} - {str(e)}")
            raise
            
    def delete_file(self, path):
        """删除WebDAV上的文件"""
        try:
            client = self.connect()
            client.clean(path)
            return True
        except Exception as e:
            logger.error(f"删除WebDAV文件失败: {path} - {str(e)}")
            raise
            
    def get_file_url(self, path):
        """获取WebDAV文件直接访问URL（如果支持）"""
        # 注意：并非所有WebDAV服务器都支持直接URL访问
        return f"{self.url.rstrip('/')}/{path.lstrip('/')}"
            
    def copy_to_local(self, remote_path, local_storage, new_path=None):
        """
        将WebDAV上的文件复制到本地存储
        
        Args:
            remote_path: WebDAV上的文件路径
            local_storage: 本地存储路径
            new_path: 在本地存储中的新路径（可选）
            
        Returns:
            本地文件的路径
        """
        # 确定本地存储路径
        if new_path is None:
            filename = os.path.basename(remote_path)
            local_path = os.path.join(local_storage, filename)
        else:
            local_path = os.path.join(local_storage, new_path)
            
        # 确保目标目录存在
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # 下载文件
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
            
        try:
            self.download_file(remote_path, temp_path)
            # 复制到最终位置
            shutil.move(temp_path, local_path)
            return local_path
        except Exception as e:
            try:
                os.unlink(temp_path)  # 清理临时文件
            except:
                pass
            raise e


# WebDAV连接管理器
class WebDAVConnectionManager:
    def __init__(self):
        self.connections = {}
        self.load_connections()
        
    def load_connections(self):
        """从配置文件加载连接"""
        try:
            if WEBDAV_CONNECTIONS_FILE.exists():
                with open(WEBDAV_CONNECTIONS_FILE, 'r', encoding='utf-8') as f:
                    connections_data = json.load(f)
                    
                self.connections = {}
                for conn_data in connections_data:
                    connection = WebDAVConnection.from_dict(conn_data)
                    self.connections[connection.name] = connection
                    
                logger.info(f"已加载 {len(self.connections)} 个WebDAV连接配置")
            else:
                self.connections = {}
                logger.info("未找到WebDAV连接配置文件，使用空配置")
        except Exception as e:
            logger.error(f"加载WebDAV连接配置失败: {str(e)}")
            self.connections = {}
            
    def save_connections(self):
        """保存连接到配置文件"""
        try:
            connections_data = [conn.to_dict() for conn in self.connections.values()]
            with open(WEBDAV_CONNECTIONS_FILE, 'w', encoding='utf-8') as f:
                json.dump(connections_data, f, indent=2, ensure_ascii=False)
            logger.info(f"已保存 {len(self.connections)} 个WebDAV连接配置")
        except Exception as e:
            logger.error(f"保存WebDAV连接配置失败: {str(e)}")
            
    def add_connection(self, name, url, username, password, folder="/", enabled=True):
        """添加新的WebDAV连接"""
        connection = WebDAVConnection(name, url, username, password, folder, enabled)
        self.connections[name] = connection
        self.save_connections()
        return connection
        
    def update_connection(self, name, url=None, username=None, password=None, folder=None, enabled=None):
        """更新现有的WebDAV连接"""
        if name not in self.connections:
            raise ValueError(f"连接不存在: {name}")
            
        connection = self.connections[name]
        
        if url is not None:
            connection.url = url
        if username is not None:
            connection.username = username
        if password is not None and password != "":  # 空密码不更新
            connection.password = password
        if folder is not None:
            connection.folder = folder
        if enabled is not None:
            connection.enabled = enabled
            
        # 重置客户端连接
        connection.disconnect()
        
        self.save_connections()
        return connection
        
    def delete_connection(self, name):
        """删除WebDAV连接"""
        if name in self.connections:
            connection = self.connections[name]
            connection.disconnect()
            del self.connections[name]
            self.save_connections()
            return True
        return False
        
    def get_connection(self, name):
        """获取指定名称的连接"""
        return self.connections.get(name)
        
    def get_all_connections(self):
        """获取所有连接"""
        return list(self.connections.values())
        
    def test_connection(self, name=None, url=None, username=None, password=None):
        """
        测试WebDAV连接
        
        如果提供了名称，测试已有连接；否则测试传入的参数
        """
        try:
            if name is not None:
                connection = self.get_connection(name)
                if connection is None:
                    return {"success": False, "message": f"连接不存在: {name}"}
            else:
                connection = WebDAVConnection("test", url, username, password)
                
            client = connection.connect()
            if client is None:
                return {"success": False, "message": "连接失败"}
                
            # 尝试列出文件以测试连接
            files = connection.list_files()
            
            return {
                "success": True, 
                "message": f"连接成功，列出了 {len(files)} 个项目",
                "items": files[:10]  # 仅返回前10个项目作为预览
            }
        except Exception as e:
            return {"success": False, "message": f"连接测试失败: {str(e)}"}