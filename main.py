import os
import socket
import webbrowser
from fastapi import FastAPI, Request, Response, UploadFile, File, Form, HTTPException, Depends, Query
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import mimetypes
from pathlib import Path
import logging
import uvicorn
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import asyncio
from contextlib import asynccontextmanager
import shutil
import uuid
from datetime import datetime, timedelta
import hashlib
from pydantic import BaseModel
from typing import Optional, Dict, Any
import json
import platform
from typing import List
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
import os 
from view import get_media_thumbnail,start_cleanup_service
import shutil
import asyncio
from functools import lru_cache
import hashlib
from webdav import configure_webdav
from webdav_client import WebDAVConnectionManager, WebDAVConnection



# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# 定义本地映射检查请求模型
class LocalFileCheckRequest(BaseModel):
    fileName: str
    directory: str
    fileSize: int
    lastModified: Optional[int] = None

# 定义本地映射请求模型
class LocalFileMapRequest(BaseModel):
    fileName: str
    directory: str
    localPath: str

class MappingSourceRequest(BaseModel):
    sources: List[str]



# 配置文件路径常量
MAPPING_SOURCES_CONFIG = Path("./config/mapping_sources.json")
# 确保配置目录存在
Path("./config").mkdir(parents=True, exist_ok=True)
# 从配置文件加载映射源列表
def load_mapping_sources():
    try:
        if MAPPING_SOURCES_CONFIG.exists():
            with open(MAPPING_SOURCES_CONFIG, "r", encoding="utf-8") as f:
                sources = json.load(f)
                return sources if isinstance(sources, list) else []
        return []
    except Exception as e:
        logger.error(f"加载映射源失败: {str(e)}")
        return []
    

# 保存映射源列表到配置文件
def save_mapping_sources(sources):
    try:
        with open(MAPPING_SOURCES_CONFIG, "w", encoding="utf-8") as f:
            json.dump(sources, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"保存映射源失败: {str(e)}")
        return False

# 初始化 LOCAL_FILE_SOURCES
LOCAL_FILE_SOURCES = load_mapping_sources()





# 创建线程池
executor = ThreadPoolExecutor(max_workers=8)

# 获取局域网IP地址
def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        logger.error(f"获取局域网IP失败: {str(e)}")
        return "127.0.0.1"

PORTA = 5888

# 定义生命周期管理器
@asynccontextmanager
async def lifespan(app: FastAPI):
# 启动时执行的代码
    start_cleanup_service(interval_minutes=60, max_age_days=7)
    cleanup_task = asyncio.create_task(cleanup_expired_links())
    local_ip = get_local_ip()
    port = PORTA
    url = f"http://{local_ip}:{port}"
    logger.info(f"服务启动，自动打开网页: {url}")
    
    # 启动 WebDAV 服务器
    if webdav_server:
        webdav_server.start()
        webdav_port = webdav_server.port
        webdav_url = f"http://{local_ip}:{webdav_port}"
        logger.info(f"WebDAV 服务启动在 {webdav_url}")
        logger.info(f"WebDAV 凭据 - 用户名: {webdav_server.username}, 密码: {webdav_server.password}")
    
    webbrowser.open(url)  # 自启动浏览器
    yield
    # 关闭时执行的代码
    logger.info("Shutting down the application")
    
    # 停止 WebDAV 服务器
    if webdav_server:
        webdav_server.stop()
    
    cleanup_task.cancel()
    executor.shutdown()

# 初始化 FastAPI 应用
app = FastAPI(lifespan=lifespan, title="ZAY-Cloud")
router = APIRouter()

# 指定文件存储路径
FILE_STORAGE_PATH = Path("./storage")
# 确保路径存在
FILE_STORAGE_PATH.mkdir(parents=True, exist_ok=True)


# 初始化 WebDAV 服务器
webdav_server = configure_webdav(FILE_STORAGE_PATH)


# 在文件开头的常量部分添加
# 背景图片存储路径
BACKGROUND_DIR = FILE_STORAGE_PATH / "backgrounds"
# 确保背景图片目录存在
BACKGROUND_DIR.mkdir(parents=True, exist_ok=True)

# 背景设置存储路径
BACKGROUND_CONFIG = Path("./config/background.json")

# 设置模板目录（用于渲染 HTML 页面）
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
app.mount("/storage", StaticFiles(directory=FILE_STORAGE_PATH), name="storage")

# 存储直链信息的字典 {token: (file_path, expiry_time)}
direct_links = {}

# 媒体类型扩展
class MediaType:
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"
    DOCUMENT = "document"
    AI = "document"  # 暂留
    OTHER = "other"

# 媒体类型映射
MEDIA_TYPES = {
    # 视频文件
    ".mp4": MediaType.VIDEO,
    ".webm": MediaType.VIDEO,
    ".ogv": MediaType.VIDEO, 
    ".mkv": MediaType.VIDEO,  # 确保MKV被识别为视频
    ".avi": MediaType.VIDEO,
    ".mov": MediaType.VIDEO,
    ".m4v": MediaType.VIDEO,  # 添加M4V支持
    ".ts": MediaType.VIDEO,   # 添加TS支持
    ".3gp": MediaType.VIDEO,  # 添加3GP支持
    ".flv": MediaType.VIDEO,  # 添加FLV支持
    # 音频文件
    ".mp3": MediaType.AUDIO,
    ".wav": MediaType.AUDIO,
    ".ogg": MediaType.AUDIO,
    ".flac": MediaType.AUDIO,
    ".aac": MediaType.AUDIO,
    ".m4a": MediaType.AUDIO,  # 添加M4A支持
    # 图片文件
    ".jpg": MediaType.IMAGE, 
    ".jpeg": MediaType.IMAGE,
    ".png": MediaType.IMAGE,
    ".gif": MediaType.IMAGE,
    ".webp": MediaType.IMAGE,
    # 文档文件
    ".pdf": MediaType.DOCUMENT,
    ".doc": MediaType.DOCUMENT,
    ".docx": MediaType.DOCUMENT,
    ".txt": MediaType.DOCUMENT,
    ".md": MediaType.DOCUMENT,
}

# 定义批量映射请求模型
class MapAllContentsRequest(BaseModel):
    sourcePath: str
    targetPath: str
    includeSubfolders: bool = True


# 辅助函数：获取文件类型
def get_media_type(filename: str) -> str:
    extension = Path(filename).suffix.lower()
    return MEDIA_TYPES.get(extension, MediaType.OTHER)

# 辅助函数：检查文件是否是视频
def is_video_file(filename: str) -> bool:
    return get_media_type(filename) == MediaType.VIDEO

# 辅助函数：检查文件是否是音频
def is_audio_file(filename: str) -> bool:
    return get_media_type(filename) == MediaType.AUDIO

# 辅助函数：检查文件是否是图片
def is_image_file(filename: str) -> bool:
    return get_media_type(filename) == MediaType.IMAGE

# 辅助函数：获取正确的 Content-Type
def get_content_type(filename: str) -> str:
    mime_type, _ = mimetypes.guess_type(filename)
    if not mime_type:
        extension = Path(filename).suffix.lower()
        mime_types = {
            ".mp4": "video/mp4",
            ".webm": "video/webm",
            ".ogv": "video/ogg",
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".ogg": "audio/ogg",
            ".flac": "audio/flac",
            ".mkv": "video/x-matroska",  # 添加MKV支持
            ".avi": "video/x-msvideo",   # 明确AVI支持
            ".mov": "video/quicktime",   # 明确MOV支持
            ".m4v": "video/x-m4v",       # 添加M4V支持
            ".3gp": "video/3gpp",        # 添加3GP支持
            ".ts": "video/mp2t",         # 添加TS支持
        }
        mime_type = mime_types.get(extension, "application/octet-stream")
    return mime_type

# 辅助函数：流式传输文件内容（支持 Range 请求，用于视频播放）
def stream_file(file_path: Path, request: Request):
    file_size = file_path.stat().st_size
    content_type = get_content_type(file_path.name)
    logger.info(f"Streaming file: {file_path.name}, Content-Type: {content_type}")

    range_header = request.headers.get("Range")
    
    # 定义文件流生成器
    def file_iterator(start, end):
        with open(file_path, "rb") as f:
            f.seek(start)
            chunk_size = 512 * 1024  # 512KB 的块大小
            position = start
            
            while position <= end:
                bytes_to_read = min(chunk_size, end - position + 1)
                data = f.read(bytes_to_read)
                if not data:
                    break
                yield data
                position += bytes_to_read

    if not range_header:
        # 如果是媒体文件但没有Range请求，使用206响应并流式传输前几MB
        if is_video_file(file_path.name) or is_audio_file(file_path.name):
            initial_chunk = 2 * 1024 * 1024  # 2MB 初始块
            end = min(initial_chunk - 1, file_size - 1)
            
            headers = {
                "Content-Length": str(end + 1),
                "Content-Type": content_type,
                "Accept-Ranges": "bytes",
                "Content-Range": f"bytes 0-{end}/{file_size}"
            }
            return StreamingResponse(
                file_iterator(0, end),
                status_code=206,
                headers=headers
            )
        # 非媒体文件使用普通流式响应
        else:
            headers = {
                "Content-Length": str(file_size),
                "Content-Type": content_type,
                "Accept-Ranges": "bytes",
            }
            return StreamingResponse(
                file_iterator(0, file_size - 1),
                status_code=200,
                headers=headers
            )

    # 处理Range请求
    start, end = 0, file_size - 1
    range_str = range_header.replace("bytes=", "")
    if "-" in range_str:
        start_str, end_str = range_str.split("-")
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else file_size - 1
        end = min(end, file_size - 1)
    
    # 合理优化块大小 - 现在由生成器控制
    chunk_size = end - start + 1
    
    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(chunk_size),
        "Content-Type": content_type,
    }
    logger.info(f"Range: {start}-{end}, Total size: {file_size}")
    
    return StreamingResponse(
        file_iterator(start, end),
        status_code=206,
        headers=headers
    )
# 同步保存文件的函数（在线程池中运行）
def save_file_sync(file_path: Path, content: bytes):
    # 确保父目录存在
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(content)
    logger.info(f"File saved: {file_path}")

# 同步删除文件的函数（在线程池中运行）
def delete_file_sync(file_path: Path):
    file_path.unlink()
    logger.info(f"File deleted: {file_path}")

# 同步创建文件夹的函数（在线程池中运行）
def create_folder_sync(folder_path: Path):
    folder_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Folder created: {folder_path}")

# 同步删除文件夹的函数（在线程池中运行）
def delete_folder_sync(folder_path: Path):
    shutil.rmtree(folder_path)
    logger.info(f"Folder deleted: {folder_path}")

# 辅助函数: 获取所有文件和文件夹
def get_files_and_folders(directory_path: Path, relative_to: Path = None):
    if relative_to is None:
        relative_to = directory_path

    items = []
    for item_path in directory_path.iterdir():
        relative_path = item_path.relative_to(relative_to)
        if item_path.is_dir():
            items.append({
                "name": item_path.name,
                "path": str(relative_path),
                "type": "folder",
                "children": get_files_and_folders(item_path, relative_to)
            })
        else:
            media_type = get_media_type(item_path.name)
            items.append({
                "name": item_path.name,
                "path": str(relative_path),
                "type": "file",
                "media_type": media_type,
                "size": item_path.stat().st_size,
                "is_video": media_type == MediaType.VIDEO,
                "is_audio": media_type == MediaType.AUDIO,
                "is_image": media_type == MediaType.IMAGE
            })
    return items

# 路由：主页，显示文件列表
@app.get("/", response_class=HTMLResponse)
async def list_files(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

# API路由: 获取文件和文件夹列表
@app.get("/api/files", response_class=JSONResponse)
async def api_list_files(path: str = ""):
    target_path = FILE_STORAGE_PATH / path if path else FILE_STORAGE_PATH
    if not target_path.exists() or not target_path.is_dir():
        raise HTTPException(status_code=404, detail=f"目录 '{path}' 不存在")
    
    # 确保目标路径是FILE_STORAGE_PATH的子目录
    if not str(target_path).startswith(str(FILE_STORAGE_PATH)):
        raise HTTPException(status_code=403, detail="无权访问该路径")
        
    files_and_folders = get_files_and_folders(target_path, FILE_STORAGE_PATH)
    
    # 分类信息
    categories = {
        "videos": [],
        "audios": [],
        "images": [],
        "documents": [],
        "others": []
    }
    
    # 填充分类信息
    for item in files_and_folders:
        if item["type"] == "file":
            media_type = item.get("media_type", "other")
            if media_type == MediaType.VIDEO:
                categories["videos"].append(item)
            elif media_type == MediaType.AUDIO:
                categories["audios"].append(item)
            elif media_type == MediaType.IMAGE:
                categories["images"].append(item)
            elif media_type == MediaType.DOCUMENT:
                categories["documents"].append(item)
            else:
                categories["others"].append(item)
                
    return {
        "items": files_and_folders,
        "categories": categories,
        "current_path": path
    }

# API路由: 创建文件夹
@app.post("/api/folders")
async def create_folder(folder_path: str = Form(...)):
    try:
        folder_path = folder_path.replace('/+', '/').strip('/')
        target_path = FILE_STORAGE_PATH / folder_path
        
        logger.info(f"收到创建文件夹请求: {folder_path}")
        
        if not str(target_path).startswith(str(FILE_STORAGE_PATH)):
            raise HTTPException(status_code=403, detail="无权在此位置创建文件夹")
        
        if target_path.exists():
            if target_path.is_dir():
                logger.info(f"文件夹已存在: {folder_path}")
                return {"message": f"文件夹 {folder_path} 已存在"}
            else:
                raise HTTPException(status_code=400, detail=f"{folder_path} 是一个文件，无法创建同名文件夹")
        
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            executor,
            partial(create_folder_sync, target_path)
        )
        logger.info(f"文件夹创建成功: {folder_path}")
        return {"message": f"文件夹 {folder_path} 创建成功"}
    except Exception as e:
        logger.error(f"创建文件夹失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建文件夹失败: {str(e)}")


# API路由: 删除文件夹
@app.delete("/api/folders/{folder_path:path}")
async def delete_folder(folder_path: str):
    try:
        # 确保路径安全
        target_path = FILE_STORAGE_PATH / folder_path
        if not str(target_path).startswith(str(FILE_STORAGE_PATH)):
            raise HTTPException(status_code=403, detail="无权删除此文件夹")
            
        if not target_path.exists() or not target_path.is_dir():
            raise HTTPException(status_code=404, detail=f"文件夹 '{folder_path}' 不存在")
            
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            executor,
            partial(delete_folder_sync, target_path)
        )
        return {"message": f"文件夹 {folder_path} 删除成功"}
    except Exception as e:
        logger.error(f"删除文件夹失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除文件夹失败: {str(e)}")

# 路由：下载文件
@app.get("/download/{file_path:path}")
async def download_file(file_path: str):
    target_path = FILE_STORAGE_PATH / file_path
    if not target_path.exists() or not target_path.is_file():
        return {"error": "文件不存在"}
    return FileResponse(
        target_path,
        media_type=mimetypes.guess_type(target_path)[0] or "application/octet-stream",
        filename=target_path.name
    )

# 路由：流式播放媒体
@app.get("/stream/{file_path:path}")
async def stream_media(file_path: str, request: Request):
    target_path = FILE_STORAGE_PATH / file_path
    if not target_path.exists() or not target_path.is_file():
        return {"error": "文件不存在"}
    return stream_file(target_path, request)

# 路由：播放页面（显示媒体播放器）
@app.get("/play/{file_path:path}", response_class=HTMLResponse)
async def play_media(request: Request, file_path: str):
    target_path = FILE_STORAGE_PATH / file_path
    if not target_path.exists() or not target_path.is_file():
        return HTMLResponse(content="<h1>文件不存在</h1>", status_code=404)
    
    media_type = get_media_type(target_path.name)
    if (media_type not in [MediaType.VIDEO, MediaType.AUDIO]):
        return HTMLResponse(content="<h1>不是可播放的媒体文件</h1>", status_code=400)
        # 获取MIME类型
    content_type, _ = mimetypes.guess_type(target_path)
    if not content_type:
        content_type = "application/octet-stream"
    # 是否为音频文件
    is_audio = media_type
    
    # 对于音频文件使用专用模板
    if is_audio in [MediaType.AUDIO]:
        return templates.TemplateResponse("music.html", {
            "request": request,
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "content_type": content_type,
            "is_audio": is_audio
        })

    # 计算 Content-Type 并传递给模板
    content_type = get_content_type(target_path.name)
    return templates.TemplateResponse(
        "play.html",
        {
            "request": request, 
            "file_path": file_path, 
            "file_name": target_path.name,
            "content_type": content_type,
            "is_audio": media_type == MediaType.AUDIO
        }
    )

# 路由：上传文件
@app.post("/upload")
async def upload_file(
    file: UploadFile, 
    directory: str = Form(default="")
):
    try:
        # 确保上传目标路径是安全的
        target_dir = FILE_STORAGE_PATH / directory
        if not str(target_dir).startswith(str(FILE_STORAGE_PATH)):
            return {"error": "无权在此位置上传文件"}
            
        # 确保目录存在
        target_dir.mkdir(parents=True, exist_ok=True)
            
        file_path = target_dir / file.filename
        # 检查文件是否已存在
        if file_path.exists():
            return {"error": f"文件 {file.filename} 已存在"}
        
        # 异步读取文件内容
        content = await file.read()
        
        # 将文件保存操作提交到线程池
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            executor,
            partial(save_file_sync, file_path, content)
        )
        
        # 确定媒体类型
        media_type = get_media_type(file.filename)
        
        return {
            "message": f"文件 {file.filename} 上传成功",
            "file": {
                "name": file.filename,
                "path": f"{directory}/{file.filename}" if directory else file.filename,
                "media_type": media_type,
                "size": len(content),
                "is_video": media_type == MediaType.VIDEO,
                "is_audio": media_type == MediaType.AUDIO,
                "is_image": media_type == MediaType.IMAGE
            }
        }
    except Exception as e:
        logger.error(f"上传失败: {str(e)}")
        return {"error": f"上传失败: {str(e)}"}

# 路由：删除文件
@app.delete("/delete/{file_path:path}")
async def delete_file(file_path: str):
    try:
        target_path = FILE_STORAGE_PATH / file_path
        # 安全检查
        if not str(target_path).startswith(str(FILE_STORAGE_PATH)):
            return {"error": "无权删除此文件"}
            
        if not target_path.exists() or not target_path.is_file():
            return {"error": "文件不存在"}
        
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            executor,
            partial(delete_file_sync, target_path)
        )
        return {"message": f"文件 {target_path.name} 删除成功"}
    except Exception as e:
        logger.error(f"删除失败: {str(e)}")
        return {"error": f"删除失败: {str(e)}"}

# API路由：生成直链下载
@app.get("/api/direct-link/{file_path:path}")
async def generate_direct_link(request: Request, file_path: str, expires_in: int = 86400):
    try:
        target_path = FILE_STORAGE_PATH / file_path
        if not target_path.exists() or not target_path.is_file():
            raise HTTPException(status_code=404, detail="文件不存在")
            
        # 生成唯一令牌
        token = str(uuid.uuid4())
        # 设置过期时间
        expiry_time = datetime.now() + timedelta(seconds=expires_in)
        # 存储直链信息
        direct_links[token] = (file_path, expiry_time)
        
        # 生成直链URL
        direct_link = f"/dl/{token}"
        
        return {
            "direct_link": direct_link, 
            "full_link": f"{request.base_url}{direct_link[1:]}",
            "expires_at": expiry_time.isoformat()
        }
    except Exception as e:
        logger.error(f"生成直链失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"生成直链失败: {str(e)}")

# 直链下载
@app.get("/dl/{token}")
async def direct_link_download(token: str):
    # 检查token是否存在
    if token not in direct_links:
        raise HTTPException(status_code=404, detail="链接不存在或已过期")
        
    file_path, expiry_time = direct_links[token]
    
    # 检查链接是否过期
    if datetime.now() > expiry_time:
        # 移除过期的链接
        direct_links.pop(token, None)
        raise HTTPException(status_code=410, detail="链接已过期")
    
    target_path = FILE_STORAGE_PATH / file_path
    if not target_path.exists() or not target_path.is_file():
        direct_links.pop(token, None)
        raise HTTPException(status_code=404, detail="文件不存在")
        
    return FileResponse(
        target_path,
        media_type=mimetypes.guess_type(target_path)[0] or "application/octet-stream",
        filename=target_path.name
    )

# 定时清理过期的直链
async def cleanup_expired_links():
    while True:
        try:
            now = datetime.now()
            expired_tokens = [
                token for token, (_, expiry) in direct_links.items() 
                if now > expiry
            ]
            
            for token in expired_tokens:
                direct_links.pop(token, None)
                logger.info(f"已清理过期直链: {token}")
                
            # 每小时检查一次
            await asyncio.sleep(3600)
        except Exception as e:
            logger.error(f"清理过期直链时出错: {str(e)}")
            await asyncio.sleep(3600)

async def cleanup_expired_links():
    while True:
        try:
            now = datetime.now()
            expired_tokens = [
                token for token, (_, expiry) in direct_links.items() 
                if now > expiry
            ]
            
            for token in expired_tokens:
                direct_links.pop(token, None)
                logger.info(f"已清理过期直链: {token}")
                
            # 每小时检查一次
            await asyncio.sleep(3600)
        except Exception as e:
            logger.error(f"清理过期直链时出错: {str(e)}")
            await asyncio.sleep(3600)

# 添加缓存装饰器
@lru_cache(maxsize=32)  # 缓存最多32个文件的元数据
def get_file_metadata(file_path: str):
    """获取文件元数据并缓存结果"""
    path = Path(file_path)
    if not path.exists():
        return None
    
    stat = path.stat()
    return {
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "content_type": get_content_type(path.name),
        "media_type": get_media_type(path.name)
    }

# 优化view_file路由
@app.get("/view/{file_path:path}")
async def view_file(request: Request, file_path: str):
    """
    提供文件的直接预览，适用于图片、音频和视频等媒体文件
    """
    target_path = FILE_STORAGE_PATH / file_path
    full_path = str(target_path)
    
    # 使用缓存检查文件
    metadata = get_file_metadata(full_path)
    if not metadata:
        raise HTTPException(status_code=404, detail="文件不存在")
    
    # 确保目标路径在存储目录内
    if not str(target_path).startswith(str(FILE_STORAGE_PATH)):
        raise HTTPException(status_code=403, detail="无权访问此文件")
    
    # 获取媒体类型
    media_type = metadata["media_type"]
    
    # 使用ETag和304缓存机制
    etag = f"\"{hashlib.md5((full_path + str(metadata['mtime'])).encode()).hexdigest()}\""
    if_none_match = request.headers.get("if-none-match")
    
    if if_none_match and if_none_match == etag:
        return Response(status_code=304)
    
    # 根据媒体类型处理
    if media_type == MediaType.VIDEO or media_type == MediaType.AUDIO:
        # 流式传输音频/视频文件
        response = stream_file(target_path, request)
        response.headers["ETag"] = etag
        return response
    else:
        # 直接返回文件（适用于图片和其他可直接预览的文件）
        response = FileResponse(
            target_path,
            media_type=metadata["content_type"]
        )
        response.headers["ETag"] = etag
        return response


# 存储临时分片的目录
CHUNK_TEMP_DIR = Path("./temp_chunks")
CHUNK_TEMP_DIR.mkdir(parents=True, exist_ok=True)

# API路由：处理文件分片上传
@app.post("/api/upload/chunk")
async def upload_chunk(
    file: UploadFile = File(...),
    directory: str = Form(...),
    fileName: str = Form(...),
    chunkIndex: int = Form(...),
    totalChunks: int = Form(...),
    fileSize: int = Form(...),
    chunkSize: int = Form(...),
):
    try:
        # 规范化目录路径
        directory = directory.strip().replace('\\', '/')
        
        # 创建文件唯一标识，用于关联所有分片
        file_id = f"{fileName}_{fileSize}_{directory}"
        file_id_hash = str(hash(file_id))
        
        # 创建临时存储分片的目录
        chunk_dir = CHUNK_TEMP_DIR / file_id_hash
        chunk_dir.mkdir(parents=True, exist_ok=True)
        
        # 分片文件路径
        chunk_path = chunk_dir / f"chunk_{chunkIndex}"
        
        # 读取分片内容并保存
        content = await file.read()
        
        # 使用线程池异步保存分片
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            executor,
            partial(save_file_sync, chunk_path, content)
        )
        
        logger.info(f"已接收分片 {chunkIndex + 1}/{totalChunks} - 文件: {fileName}")
        
        return {
            "message": f"分片 {chunkIndex + 1}/{totalChunks} 上传成功",
            "chunk_index": chunkIndex,
            "file_id": file_id_hash,
        }
    except Exception as e:
        logger.error(f"分片上传失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"分片上传失败: {str(e)}")

# API路由：完成分片上传，合并所有分片
@app.post("/api/upload/complete")
async def complete_upload(request: Request):
    try:
        # 解析请求体
        data = await request.json()
        fileName = data.get("fileName")
        directory = data.get("directory", "").strip().replace('\\', '/')
        totalChunks = data.get("totalChunks")
        fileSize = data.get("fileSize")
        
        if not all([fileName, totalChunks, fileSize]):
            raise HTTPException(status_code=400, detail="缺少必要参数")
            
        # 创建文件唯一标识
        file_id = f"{fileName}_{fileSize}_{directory}"
        file_id_hash = str(hash(file_id))
        
        # 临时分片目录
        chunk_dir = CHUNK_TEMP_DIR / file_id_hash
        
        if not chunk_dir.exists():
            raise HTTPException(status_code=404, detail="找不到相关分片数据")
            
        # 目标文件路径
        target_dir = FILE_STORAGE_PATH / directory
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / fileName
        
        # 检查文件是否已存在
        if target_path.exists():
            # 清理临时分片
            shutil.rmtree(chunk_dir)
            raise HTTPException(status_code=409, detail=f"文件 {fileName} 已存在")
        
        # 异步合并分片
        try:
            # 在单独的线程中执行合并操作
            def merge_chunks():
                with open(target_path, 'wb') as target_file:
                    # 按顺序合并所有分片
                    for i in range(totalChunks):
                        chunk_path = chunk_dir / f"chunk_{i}"
                        if not chunk_path.exists():
                            raise FileNotFoundError(f"分片 {i+1}/{totalChunks} 不存在")
                        
                        with open(chunk_path, 'rb') as chunk_file:
                            target_file.write(chunk_file.read())
                
                # 验证文件大小
                actual_size = target_path.stat().st_size
                if actual_size != fileSize:
                    raise ValueError(f"文件大小不匹配: 预期 {fileSize} 字节，实际 {actual_size} 字节")
                
                # 清理临时分片
                shutil.rmtree(chunk_dir)
                
                logger.info(f"文件 {fileName} 合并完成，总大小: {actual_size} 字节")

            # 在线程池中执行合并操作
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(executor, merge_chunks)
            
            # 确定媒体类型
            media_type = get_media_type(fileName)
            
            return {
                "message": f"文件 {fileName} 上传成功",
                "file": {
                    "name": fileName,
                    "path": f"{directory}/{fileName}" if directory else fileName,
                    "media_type": media_type,
                    "size": fileSize,
                    "is_video": media_type == MediaType.VIDEO,
                    "is_audio": media_type == MediaType.AUDIO,
                    "is_image": media_type == MediaType.IMAGE
                }
            }
        except Exception as e:
            # 出错时尝试清理
            try:
                if target_path.exists():
                    target_path.unlink()
            except:
                pass
            
            logger.error(f"合并文件失败: {str(e)}")
            raise HTTPException(status_code=500, detail=f"合并文件失败: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"完成上传请求处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"完成上传请求处理失败: {str(e)}")






# 添加获取系统默认路径的API
@app.get("/api/system-default-paths")
async def get_system_default_paths():
    """获取系统默认路径建议"""
    default_paths = []
    system = platform.system()
    
    try:
        # Windows系统默认路径
        if system == "Windows":
            # 添加下载文件夹
            try:
                import winreg
                with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 
                                 r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders") as key:
                    downloads_dir = winreg.QueryValueEx(key, "{374DE290-123F-4565-9164-39C4925E467B}")[0]
                    default_paths.append(downloads_dir)
            except:
                pass
                
            # 添加桌面和文档
            user_home = str(Path.home())
            default_paths.extend([
                user_home + "\\Desktop",
                user_home + "\\Documents",
                user_home + "\\Pictures",
                user_home + "\\Videos",
                "C:\\Users\\Public\\Downloads"
            ])
        
        # Linux系统默认路径
        elif system == "Linux":
            user_home = str(Path.home())
            default_paths.extend([
                user_home + "/Downloads",
                user_home + "/Desktop",
                user_home + "/Documents",
                user_home + "/Pictures",
                user_home + "/Videos",
                "/mnt",
                "/media/" + os.getlogin()
            ])
        
        # macOS系统默认路径
        elif system == "Darwin":
            user_home = str(Path.home())
            default_paths.extend([
                user_home + "/Downloads",
                user_home + "/Desktop",
                user_home + "/Documents",
                user_home + "/Pictures",
                user_home + "/Movies"
            ])
            
        # 过滤掉不存在的路径
        valid_paths = [path for path in default_paths if Path(path).exists() and Path(path).is_dir()]
        
        return {
            "system": system,
            "defaultPaths": valid_paths
        }
    except Exception as e:
        logger.error(f"获取系统默认路径失败: {str(e)}")
        return {
            "system": system,
            "defaultPaths": [],
            "error": str(e)
        }


# 添加 API 路由用于获取映射源列表
@app.get("/api/mapping-sources")
async def get_mapping_sources():
    """获取当前配置的映射源列表"""
    return {"sources": LOCAL_FILE_SOURCES}

# 添加 API 路由用于更新映射源列表
@app.post("/api/mapping-sources")
async def update_mapping_sources(request: MappingSourceRequest):
    """更新映射源列表配置"""
    global LOCAL_FILE_SOURCES
    
    try:
        # 检查路径有效性
        valid_sources = []
        invalid_sources = []
        
        for source in request.sources:
            # 规范化路径格式
            source_path = Path(source.strip())
            
            # 检查路径是否存在
            if source_path.exists() and source_path.is_dir():
                valid_sources.append(str(source_path))
            else:
                invalid_sources.append(source)
        
        # 更新全局变量
        LOCAL_FILE_SOURCES = valid_sources
        
        # 保存到配置文件
        save_mapping_sources(valid_sources)
        
        return {
            "success": True,
            "message": "映射源已更新",
            "validSources": valid_sources,
            "invalidSources": invalid_sources
        }
    except Exception as e:
        logger.error(f"更新映射源失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"更新映射源失败: {str(e)}")




# API路由：检查文件是否可以从本地映射
@app.post("/api/check-local-file")
async def check_local_file(request: LocalFileCheckRequest):
    try:
        # 规范化目录路径
        directory = request.directory.strip().replace('\\', '/')
        file_name = request.fileName
        file_size = request.fileSize
        last_modified = request.lastModified  # 可选的最后修改时间
        
        logger.info(f"检查文件是否可以本地映射: {file_name}, 大小: {file_size} 字节")
        
        # 首先检查目标位置是否已存在该文件
        target_path = FILE_STORAGE_PATH / directory / file_name
        if target_path.exists() and target_path.is_file():
            if target_path.stat().st_size == file_size:
                logger.info(f"文件已存在于目标位置: {target_path}")
                return {"canMap": False, "reason": "文件已存在于目标位置"}
        
        # 如果用户提供了本地路径，直接检查该路径
        if hasattr(request, 'localPath') and request.localPath:
            local_path = Path(request.localPath)
            if local_path.exists() and local_path.is_file() and local_path.stat().st_size == file_size:
                logger.info(f"找到匹配的本地文件: {local_path}")
                return {
                    "canMap": True, 
                    "localPath": str(local_path), 
                    "sourceDirectory": str(local_path.parent)
                }
        
        # 扩展搜索范围 - 包括常见下载目录
        common_paths = []
        
        # 添加系统下载文件夹
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 
                              r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders") as key:
                downloads_dir = winreg.QueryValueEx(key, "{374DE290-123F-4565-9164-39C4925E467B}")[0]
                common_paths.append(Path(downloads_dir))
        except:
            # 回退到常见下载路径
            user_home = Path.home()
            common_paths.append(user_home / "Downloads")
            common_paths.append(user_home / "Desktop")
        LOCAL_FILE_SOURCES = load_mapping_sources()
   
        # 如果设置了LOCAL_FILE_SOURCES，加入搜索范围 - 这里需要修改
        if LOCAL_FILE_SOURCES:
            # 将字符串路径转换为Path对象
            common_paths.extend([Path(path) for path in LOCAL_FILE_SOURCES])
            
        # 在常见路径中搜索
        for search_dir in common_paths:
            if not isinstance(search_dir, Path):
                search_dir = Path(search_dir)  # 确保是Path对象
                
            if not search_dir.exists() or not search_dir.is_dir():
                continue
                
            # 首先直接在父目录中查找同名文件
            direct_match = search_dir / file_name
            if direct_match.exists() and direct_match.is_file() and direct_match.stat().st_size == file_size:
                logger.info(f"找到匹配的本地文件: {direct_match}")
                return {
                    "canMap": True, 
                    "localPath": str(direct_match), 
                    "sourceDirectory": str(search_dir)
                }
            
            # 递归搜索 (限制深度为2，避免搜索时间过长)
            for path in search_dir.glob('*/*'):
                if path.is_file() and path.name == file_name and path.stat().st_size == file_size:
                    # 如果提供了lastModified，进一步验证
                    if last_modified and abs(path.stat().st_mtime * 1000 - last_modified) > 60000:  # 允许1分钟误差
                        continue
                        
                    logger.info(f"找到匹配的本地文件: {path}")
                    return {
                        "canMap": True, 
                        "localPath": str(path), 
                        "sourceDirectory": str(path.parent)
                    }
        
        logger.info(f"未找到匹配的本地文件: {file_name}")
        return {"canMap": False, "reason": "未找到匹配的本地文件"}
    except Exception as e:
        logger.error(f"检查本地文件映射失败: {str(e)}")
        return {"canMap": False, "reason": str(e)}

# API路由：映射本地文件到存储
@app.post("/api/map-local-file")
async def map_local_file(request: LocalFileMapRequest):
    try:
        # 规范化路径
        directory = request.directory.strip().replace('\\', '/')
        file_name = request.fileName  # 保持原始文件名，包括空格
        local_path = Path(request.localPath)
        
        # 安全检查
        if not local_path.exists() or not local_path.is_file():
            raise HTTPException(status_code=404, detail="源文件不存在")
        
        # 确保目标目录存在
        target_dir = FILE_STORAGE_PATH / directory
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / file_name
        
        # 检查目标文件是否已存在
        if target_path.exists():
            raise HTTPException(status_code=409, detail="目标文件已存在")
        
        # 执行映射操作（有两种选择：复制或创建硬链接）
        # 在线程池中异步执行复制
        loop = asyncio.get_running_loop()
        
        # 决定使用硬链接还是复制
        try:
            # 尝试创建硬链接
            def create_hard_link():
                try:
                    # 确保使用完全一致的文件名（包括空格）
                    import os
                    os.link(str(local_path), str(target_path))
                    return True
                except Exception as e:
                    logger.warning(f"创建硬链接失败: {str(e)}，将改用复制")
                    return False
            
            link_created = await loop.run_in_executor(executor, lambda: os.path.exists(str(target_path)) or create_hard_link())
            
            # 如果硬链接失败，则复制文件
            if not link_created:
                def copy_file():
                    shutil.copy2(local_path, target_path)
                
                await loop.run_in_executor(executor, copy_file)
                logger.info(f"已复制文件: {local_path} -> {target_path}")
            else:
                logger.info(f"已创建硬链接: {local_path} -> {target_path}") 
        except Exception as e:
            # 出错时回退到复制
            def copy_file():
                shutil.copy2(local_path, target_path)
            
            await loop.run_in_executor(executor, copy_file)
            logger.info(f"已复制文件: {local_path} -> {target_path}")
        
        # 确定媒体类型
        media_type = get_media_type(file_name)
        file_size = local_path.stat().st_size
        
        return {
            "message": f"文件 {file_name} 已成功映射",
            "file": {
                "name": file_name,
                "path": f"{directory}/{file_name}" if directory else file_name,
                "media_type": media_type,
                "size": file_size,
                "is_video": media_type == MediaType.VIDEO,
                "is_audio": media_type == MediaType.AUDIO,
                "is_image": media_type == MediaType.IMAGE,
                "mapped_from": str(local_path)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"映射本地文件失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"映射本地文件失败: {str(e)}")


# 添加在文件的合适位置（其他API路由之后）

# 背景图片模型
class BackgroundSettings(BaseModel):
    current_background: Optional[str] = None
    user_backgrounds: List[Dict[str, str]] = []

# 加载背景设置
def load_background_settings():
    try:
        if BACKGROUND_CONFIG.exists():
            with open(BACKGROUND_CONFIG, "r", encoding="utf-8") as f:
                data = json.load(f)
                return BackgroundSettings(**data)
        return BackgroundSettings()
    except Exception as e:
        logger.error(f"加载背景设置失败: {str(e)}")
        return BackgroundSettings()

# 保存背景设置
def save_background_settings(settings: BackgroundSettings):
    try:
        with open(BACKGROUND_CONFIG, "w", encoding="utf-8") as f:
            json.dump(settings.dict(), f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logger.error(f"保存背景设置失败: {str(e)}")
        return False

# API: 获取背景设置
@app.get("/api/background")
async def get_background_settings(request: Request):
    settings = load_background_settings()
    # 将相对路径转换为完整URL
    base_url = str(request.base_url).rstrip('/')
    
    # 处理用户背景
    formatted_backgrounds = []
    for bg in settings.user_backgrounds:
        bg_copy = bg.copy()
        if bg.get("url") and not bg["url"].startswith(("http://", "https://")):
            # 相对路径转为完整URL
            bg_copy["url"] = f"{base_url}/{bg['url']}"
        formatted_backgrounds.append(bg_copy)
    
    # 处理当前背景
    current_background = settings.current_background
    if current_background and not current_background.startswith(("http://", "https://")):
        current_background = f"{base_url}/{current_background}"
    
    return {
        "current_background": current_background,
        "user_backgrounds": formatted_backgrounds
    }

# API: 上传背景图片
@app.post("/api/background/upload")
async def upload_background(
    background: UploadFile = File(...),
    name: str = Form(None)
):
    try:
        # 验证是否为图片文件
        content_type = background.content_type
        if not content_type or not content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="只能上传图片文件")
        
        # 生成安全的文件名
        original_name = background.filename
        file_extension = Path(original_name).suffix.lower()
        safe_name = f"{uuid.uuid4()}{file_extension}"
        file_path = BACKGROUND_DIR / safe_name
        
        # 读取内容并保存
        content = await background.read()
        
        # 异步保存文件
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            executor,
            partial(save_file_sync, file_path, content)
        )
        
        # 更新背景设置
        settings = load_background_settings()
        relative_path = f"storage/backgrounds/{safe_name}"
        
        # 使用用户提供的名称或原始文件名
        display_name = name or original_name
        
        # 添加到用户背景列表
        settings.user_backgrounds.append({
            "name": display_name,
            "url": relative_path,
            "id": str(uuid.uuid4())
        })
        
        # 保存设置
        save_background_settings(settings)
        
        return {
            "message": "背景图片上传成功",
            "background": {
                "name": display_name,
                "url": relative_path,
                "id": settings.user_backgrounds[-1]["id"]
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"上传背景图片失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"上传背景图片失败: {str(e)}")

# API: 设置当前背景
@app.post("/api/background/set")
async def set_background(background_url: str = Form(...)):
    try:
        settings = load_background_settings()
        settings.current_background = background_url
        save_background_settings(settings)
        
        return {"message": "背景设置已更新", "current_background": background_url}
    except Exception as e:
        logger.error(f"设置背景失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"设置背景失败: {str(e)}")

# API: 删除背景图片
@app.delete("/api/background/{background_id}")
async def delete_background(background_id: str):
    try:
        settings = load_background_settings()
        
        # 查找要删除的背景
        found = False
        for i, bg in enumerate(settings.user_backgrounds):
            print(bg.get("id"), background_id)
            print(i, background_id)
            if bg.get("id") == background_id:
                print("found")
                found = True
                # 如果是当前背景，则重置当前背景
                if settings.current_background == bg.get("url"):
                    settings.current_background = None
                
                # 如果背景图片在本地存储，则删除文件
                url = bg.get("url", "")
                print(url)
                if url.startswith("storage/backgrounds/"):
                    print("delete")
                    file_name = url.split("/")[-1]
                    file_path = BACKGROUND_DIR / file_name
                    if file_path.exists():
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(
                            executor,
                            partial(delete_file_sync, file_path)
                        )
                
                # 从列表中删除
                settings.user_backgrounds.pop(i)
                break
        
        if not found:
            raise HTTPException(status_code=404, detail="背景不存在")
        
        # 保存设置
        save_background_settings(settings)
        
        return {"message": "背景已删除"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除背景失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除背景失败: {str(e)}")

# API: 重置背景
@app.post("/api/background/reset")
async def reset_background():
    try:
        settings = load_background_settings()
        settings.current_background = None
        save_background_settings(settings)
        
        return {"message": "背景已重置"}
    except Exception as e:
        logger.error(f"重置背景失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"重置背景失败: {str(e)}")


@app.get("/api/thumbnail/{media_type}/{path:path}")
async def get_thumbnail(media_type: str, path: str):
    """获取媒体文件的缩略图"""
    # 将URL路径转换为系统路径
    file_path = os.path.join(FILE_STORAGE_PATH, path)  # 根据您的文件存储路径调整
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="文件不存在")
    
    if media_type not in ["video", "audio"]:
        raise HTTPException(status_code=400, detail="不支持的媒体类型")
    
    thumbnail_path = get_media_thumbnail(file_path, media_type)
    
    if not thumbnail_path or not os.path.exists(thumbnail_path):
        raise HTTPException(status_code=404, detail="无法生成缩略图")
    
    return FileResponse(thumbnail_path)



# API路由：批量映射源文件夹内容到目标目录
@app.post("/api/map-all-contents")
async def map_all_contents(request: MapAllContentsRequest):
    """批量映射源文件夹下的所有内容到目标目录"""
    try:
        source_path = Path(request.sourcePath)
        print(source_path)
        if not source_path.exists() or not source_path.is_dir():
            raise HTTPException(status_code=404, detail="源文件夹不存在")
        
        target_dir = FILE_STORAGE_PATH / request.targetPath
        target_dir.mkdir(parents=True, exist_ok=True)
        
        # 计数器
        mapped_count = 0
        skipped_count = 0
        
        # 定义映射函数（将在线程池中执行）
        def map_files_recursive():
            nonlocal mapped_count, skipped_count
            
            # 递归函数来处理文件和子目录
            def process_directory(src_dir, rel_path=""):
                nonlocal mapped_count, skipped_count
                
                # 处理当前目录中的所有文件
                for item in src_dir.iterdir():
                    # 计算相对路径
                    item_rel_path = f"{rel_path}/{item.name}" if rel_path else item.name
                    
                    if item.is_file():
                        # 创建目标路径
                        dest_path = target_dir / item_rel_path
                        
                        # 如果目标文件已存在，跳过
                        if dest_path.exists():
                            skipped_count += 1
                            continue
                        
                        # 确保父目录存在
                        dest_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        try:
                            # 尝试创建硬链接
                            try:
                                # Windows和Linux的硬链接创建方式
                                os.link(str(item), str(dest_path))
                                mapped_count += 1
                            except Exception:
                                # 如果硬链接失败，回退到复制
                                shutil.copy2(item, dest_path)
                                mapped_count += 1
                        except Exception as e:
                            logger.error(f"映射文件失败: {str(e)}, 源: {item}, 目标: {dest_path}")
                            skipped_count += 1
                    
                    # 处理子目录（如果includeSubfolders为True）
                    elif item.is_dir() and request.includeSubfolders:
                        process_directory(item, item_rel_path)
            
            # 开始处理根目录
            process_directory(source_path)
            return mapped_count, skipped_count
        
        # 在线程池中执行映射操作
        loop = asyncio.get_running_loop()
        mapped_count, skipped_count = await loop.run_in_executor(executor, map_files_recursive)
        
        return {
            "success": True,
            "message": f"批量映射完成，共映射 {mapped_count} 个文件，跳过 {skipped_count} 个已存在文件",
            "mappedCount": mapped_count,
            "skippedCount": skipped_count
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"批量映射文件失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"批量映射文件失败: {str(e)}")




# -------------webDAV-----------------------------
# 添加 WebDAV 配置模型
class WebDAVConfig(BaseModel):
    enabled: bool = True
    port: int = 5889
    auth_enabled: bool = True
    username: str = "admin"
    password: str = "admin"

# 添加 WebDAV 配置 API
@app.get("/api/webdav/config")
async def get_webdav_config():
    """获取 WebDAV 服务配置"""
    config_file = Path("./config/webdav.conf")
    if not config_file.exists():
        return {
            "enabled": webdav_server is not None,
            "port": webdav_server.port if webdav_server else 5889,
            "auth_enabled": webdav_server.auth_enabled if webdav_server else True,
            "username": webdav_server.username if webdav_server else "admin",
            "password": "*****",  # 不返回真实密码
            "status": "running" if (webdav_server and webdav_server.server_thread and 
                                   webdav_server.server_thread.is_alive()) else "stopped",
            "url": f"http://{get_local_ip()}:{webdav_server.port if webdav_server else 5889}" if webdav_server else None
        }
        
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
            # 屏蔽密码
            if "password" in config:
                config["password"] = "*****"
            
            # 添加当前状态信息
            config["status"] = "running" if (webdav_server and webdav_server.server_thread and 
                                          webdav_server.server_thread.is_alive()) else "stopped"
            if webdav_server:
                config["url"] = f"http://{get_local_ip()}:{webdav_server.port}"
            
            return config
    except Exception as e:
        logger.error(f"获取 WebDAV 配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取 WebDAV 配置失败: {str(e)}")

@app.post("/api/webdav/config")
async def update_webdav_config(config: WebDAVConfig):
    """更新 WebDAV 服务配置"""
    global webdav_server
    
    try:
        # 保存配置
        config_file = Path("./config/webdav.conf")
        config_dict = config.dict()
        
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, ensure_ascii=False, indent=2)
        
        # 如果 WebDAV 服务器正在运行，停止它
        if webdav_server:
            webdav_server.stop()
            webdav_server = None
        
        # 如果启用 WebDAV，创建并启动新的服务器
        if config.enabled:
            webdav_server = configure_webdav(FILE_STORAGE_PATH)
            if webdav_server:
                webdav_server.start()
                return {
                    "message": "WebDAV 配置已更新，服务已重启",
                    "status": "running",
                    "url": f"http://{get_local_ip()}:{webdav_server.port}"
                }
            else:
                return {
                    "message": "WebDAV 配置已更新，但服务未能启动",
                    "status": "stopped"
                }
        else:
            return {
                "message": "WebDAV 配置已更新，服务已停止",
                "status": "stopped"
            }
    except Exception as e:
        logger.error(f"更新 WebDAV 配置失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"更新 WebDAV 配置失败: {str(e)}")





# 在应用初始化后添加
# 初始化 WebDAV 客户端管理器
webdav_client_manager = WebDAVConnectionManager()

# 添加 WebDAV 客户端数据模型
class WebDAVConnectionRequest(BaseModel):
    name: str
    url: str
    username: str
    password: str = ""
    folder: str = "/"
    enabled: bool = True

# 添加 WebDAV 客户端 API 路由
@app.get("/api/webdav-client/connections")
async def get_webdav_connections():
    """获取所有 WebDAV 连接"""
    connections = webdav_client_manager.get_all_connections()
    result = []
    for conn in connections:
        conn_data = conn.to_dict()
        conn_data["password"] = "********" if conn_data["password"] else ""  # 隐藏密码
        result.append(conn_data)
    return result

@app.post("/api/webdav-client/connections")
async def add_webdav_connection(connection: WebDAVConnectionRequest):
    """添加新的 WebDAV 连接"""
    try:
        # 检查名称是否已存在
        if webdav_client_manager.get_connection(connection.name):
            raise HTTPException(status_code=400, detail=f"连接名称 '{connection.name}' 已存在")
            
        # 添加连接
        webdav_client_manager.add_connection(
            name=connection.name,
            url=connection.url,
            username=connection.username,
            password=connection.password,
            folder=connection.folder,
            enabled=connection.enabled
        )
        return {"success": True, "message": f"连接 '{connection.name}' 已添加"}
    except Exception as e:
        logger.error(f"添加 WebDAV 连接失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"添加 WebDAV 连接失败: {str(e)}")

@app.put("/api/webdav-client/connections/{name}")
async def update_webdav_connection(name: str, connection: WebDAVConnectionRequest):
    """更新 WebDAV 连接"""
    try:
        # 检查连接是否存在
        if not webdav_client_manager.get_connection(name):
            raise HTTPException(status_code=404, detail=f"连接 '{name}' 不存在")
            
        # 处理重命名情况
        if name != connection.name:
            # 如果新名称已存在，则返回错误
            if webdav_client_manager.get_connection(connection.name):
                raise HTTPException(status_code=400, detail=f"连接名称 '{connection.name}' 已被使用")
                
            # 删除旧连接
            webdav_client_manager.delete_connection(name)
            
            # 创建新连接
            webdav_client_manager.add_connection(
                name=connection.name,
                url=connection.url,
                username=connection.username,
                password=connection.password,
                folder=connection.folder,
                enabled=connection.enabled
            )
        else:
            # 更新现有连接
            webdav_client_manager.update_connection(
                name=name,
                url=connection.url,
                username=connection.username,
                password=connection.password,
                folder=connection.folder,
                enabled=connection.enabled
            )
            
        return {"success": True, "message": f"连接 '{connection.name}' 已更新"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新 WebDAV 连接失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"更新 WebDAV 连接失败: {str(e)}")

@app.delete("/api/webdav-client/connections/{name}")
async def delete_webdav_connection(name: str):
    """删除 WebDAV 连接"""
    try:
        if not webdav_client_manager.delete_connection(name):
            raise HTTPException(status_code=404, detail=f"连接 '{name}' 不存在")
        return {"success": True, "message": f"连接 '{name}' 已删除"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除 WebDAV 连接失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除 WebDAV 连接失败: {str(e)}")

@app.post("/api/webdav-client/test-connection")
async def test_webdav_connection(connection: WebDAVConnectionRequest):
    """测试 WebDAV 连接"""
    try:
        result = webdav_client_manager.test_connection(
            url=connection.url,
            username=connection.username,
            password=connection.password
        )
        return result
    except Exception as e:
        logger.error(f"测试 WebDAV 连接失败: {str(e)}")
        return {"success": False, "message": f"测试连接失败: {str(e)}"}

@app.get("/api/webdav-client/{connection_name}/list")
async def list_webdav_files(connection_name: str, path: str = Query("/")):
    """列出 WebDAV 连接中的文件和文件夹"""
    try:
        connection = webdav_client_manager.get_connection(connection_name)
        if not connection:
            raise HTTPException(status_code=404, detail=f"连接 '{connection_name}' 不存在")
            
        files = connection.list_files(path)
        return files
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"列出 WebDAV 文件失败: {connection_name}/{path} - {str(e)}")
        raise HTTPException(status_code=500, detail=f"列出 WebDAV 文件失败: {str(e)}")

@app.post("/api/webdav-client/{connection_name}/download")
async def download_from_webdav(connection_name: str, path: str = Form(...), target_folder: str = Form("/")):
    """从 WebDAV 下载文件到系统存储"""
    try:
        connection = webdav_client_manager.get_connection(connection_name)
        if not connection:
            raise HTTPException(status_code=404, detail=f"连接 '{connection_name}' 不存在")
            
        # 确定本地存储路径
        filename = os.path.basename(path)
        target_path = os.path.normpath(os.path.join(target_folder, filename))
        target_path = target_path.replace("\\", "/").lstrip("/")  # 规范化路径
        
        local_path = FILE_STORAGE_PATH / target_path
        
        # 确保目标目录存在
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # 下载文件
        connection.download_file(path, local_path)
        
        return {
            "success": True, 
            "message": f"文件已下载",
            "path": target_path
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"从 WebDAV 下载文件失败: {connection_name}/{path} - {str(e)}")
        raise HTTPException(status_code=500, detail=f"下载文件失败: {str(e)}")

@app.post("/api/webdav-client/{connection_name}/mkdir")
async def create_webdav_directory(connection_name: str, path: str = Form(...)):
    """在 WebDAV 上创建目录"""
    try:
        connection = webdav_client_manager.get_connection(connection_name)
        if not connection:
            raise HTTPException(status_code=404, detail=f"连接 '{connection_name}' 不存在")
            
        connection.create_directory(path)
        
        return {
            "success": True, 
            "message": f"目录已创建",
            "path": path
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"在 WebDAV 上创建目录失败: {connection_name}/{path} - {str(e)}")
        raise HTTPException(status_code=500, detail=f"创建目录失败: {str(e)}")

@app.delete("/api/webdav-client/{connection_name}/delete")
async def delete_webdav_file(connection_name: str, path: str = Query(...)):
    """删除 WebDAV 上的文件或目录"""
    try:
        connection = webdav_client_manager.get_connection(connection_name)
        if not connection:
            raise HTTPException(status_code=404, detail=f"连接 '{connection_name}' 不存在")
            
        connection.delete_file(path)
        
        return {
            "success": True, 
            "message": f"已删除",
            "path": path
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除 WebDAV 文件失败: {connection_name}/{path} - {str(e)}")
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")



# 添加转码功能
# @app.get("/transcode/{file_path:path}")
# async def transcode_media(file_path: str, format: str = "mp4", codec: str = "h264"):
#     """将视频转码为浏览器兼容格式"""
#     source_path = FILE_STORAGE_PATH / file_path
#     if not source_path.exists() or not source_path.is_file():
#         raise HTTPException(status_code=404, detail="文件不存在")
    
#     # 缓存路径
#     transcode_cache_dir = Path("./transcode_cache")
#     transcode_cache_dir.mkdir(exist_ok=True)
    
#     # 生成唯一的缓存文件名
#     file_hash = hashlib.md5(f"{file_path}_{format}_{codec}".encode()).hexdigest()
#     output_filename = f"{file_hash}.{format}"
#     output_path = transcode_cache_dir / output_filename
    
#     # 如果已有转码缓存，直接返回
#     if output_path.exists():
#         return FileResponse(
#             output_path,
#             media_type=f"video/{format}",
#             filename=Path(file_path).name
#         )
    
#     # 使用FFmpeg进行转码
#     def do_transcode():
#         try:
#             cmd = [
#                 "ffmpeg", "-i", str(source_path), 
#                 "-c:v", codec if codec != "h264" else "libx264",
#                 "-preset", "fast", "-crf", "23",
#                 "-c:a", "aac", "-b:a", "128k",
#                 "-movflags", "+faststart", 
#                 str(output_path)
#             ]
#             subprocess.run(cmd, check=True)
#             logger.info(f"转码完成: {file_path} -> {output_path}")
#             return True
#         except Exception as e:
#             logger.error(f"转码失败: {e}")
#             return False
    
#     # 异步执行转码
#     loop = asyncio.get_running_loop()
#     success = await loop.run_in_executor(executor, do_transcode)
    
#     if not success:
#         raise HTTPException(status_code=500, detail="视频转码失败")
    
#     return FileResponse(
#         output_path,
#         media_type=f"video/{format}",
#         filename=Path(file_path).name
#     )


# # 添加HLS流生成功能
# @app.get("/hls/{file_path:path}/master.m3u8")
# async def generate_hls_master(file_path: str, request: Request):
#     """生成HLS主播放列表"""
#     source_path = FILE_STORAGE_PATH / file_path
#     if not source_path.exists() or not source_path.is_file():
#         raise HTTPException(status_code=404, detail="文件不存在")
    
#     # 创建HLS缓存目录
#     hls_cache_dir = Path("./hls_cache")
#     hls_cache_dir.mkdir(exist_ok=True)
    
#     # 为每个文件创建唯一的目录
#     file_hash = hashlib.md5(file_path.encode()).hexdigest()
#     hls_dir = hls_cache_dir / file_hash
#     hls_dir.mkdir(exist_ok=True)
    
#     master_playlist_path = hls_dir / "master.m3u8"
    
#     # 如果已存在主播放列表，直接返回
#     if master_playlist_path.exists():
#         return FileResponse(
#             master_playlist_path,
#             media_type="application/vnd.apple.mpegurl"
#         )
    
#     # 生成HLS片段和播放列表
#     def generate_hls():
#         try:
#             # 生成多个质量的视频流
#             resolutions = ["640x360", "960x540", "1280x720"]
#             bitrates = ["800k", "1500k", "3000k"]
            
#             variant_paths = []
            
#             for i, (resolution, bitrate) in enumerate(zip(resolutions, bitrates)):
#                 variant_name = f"variant_{i}.m3u8"
#                 variant_path = hls_dir / variant_name
#                 variant_paths.append((variant_name, resolution, bitrate))
                
#                 # 如果变体播放列表已存在，跳过生成
#                 if variant_path.exists():
#                     continue
                
#                 # 使用FFmpeg生成HLS片段
#                 cmd = [
#                     "ffmpeg", "-i", str(source_path),
#                     "-vf", f"scale={resolution}",
#                     "-c:v", "libx264", "-preset", "fast", "-crf", "23",
#                     "-c:a", "aac", "-b:a", "128k",
#                     "-f", "hls",
#                     "-hls_time", "10",
#                     "-hls_list_size", "0",
#                     "-hls_segment_filename", str(hls_dir / f"segment_{i}_%03d.ts"),
#                     str(variant_path)
#                 ]
                
#                 subprocess.run(cmd, check=True)
            
#             # 创建主播放列表
#             with open(master_playlist_path, 'w') as f:
#                 f.write("#EXTM3U\n")
#                 f.write("#EXT-X-VERSION:3\n")
                
#                 for variant_name, resolution, bitrate in variant_paths:
#                     width, height = resolution.split('x')
#                     f.write(f'#EXT-X-STREAM-INF:BANDWIDTH={bitrate.replace("k", "000")},RESOLUTION={resolution}\n')
#                     f.write(f"{variant_name}\n")
            
#             return True
#         except Exception as e:
#             logger.error(f"生成HLS失败: {e}")
#             return False
    
#     # 异步执行HLS生成
#     loop = asyncio.get_running_loop()
#     success = await loop.run_in_executor(executor, generate_hls)
    
#     if not success:
#         raise HTTPException(status_code=500, detail="生成HLS流失败")
    
#     return FileResponse(
#         master_playlist_path,
#         media_type="application/vnd.apple.mpegurl"
#     )

# # 添加HLS分段处理
# @app.get("/hls/{file_path:path}/{playlist_file}")
# async def get_hls_playlist(file_path: str, playlist_file: str):
#     """获取HLS播放列表或分段"""
#     # 创建HLS缓存目录
#     hls_cache_dir = Path("./hls_cache")
    
#     # 为每个文件创建唯一的目录
#     file_hash = hashlib.md5(file_path.encode()).hexdigest()
#     hls_dir = hls_cache_dir / file_hash
    
#     target_file = hls_dir / playlist_file
    
#     if not target_file.exists():
#         raise HTTPException(status_code=404, detail="HLS文件不存在")
    
#     # 根据文件扩展名确定内容类型
#     content_type = "application/vnd.apple.mpegurl" if playlist_file.endswith(".m3u8") else "video/mp2t"
    
#     return FileResponse(
#         target_file,
#         media_type=content_type
#     )






if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORTA)
