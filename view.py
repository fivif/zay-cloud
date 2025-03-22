import os
import io
import uuid
import cv2
from PIL import Image
from mutagen.mp3 import MP3
from mutagen.id3 import ID3
from mutagen.flac import FLAC
from mutagen.mp4 import MP4
import time
import threading
import datetime
from typing import Optional

# 缩略图缓存目录
THUMBNAIL_CACHE_DIR = os.path.join("static", "thumbnails")
os.makedirs(THUMBNAIL_CACHE_DIR, exist_ok=True)

def get_video_thumbnail(video_path, max_size=(320, 240), timestamp=3.0):
    """
    从视频中提取指定时间点的帧作为缩略图
    
    Args:
        video_path: 视频完整路径
        max_size: 缩略图最大尺寸
        timestamp: 截取视频的时间点(秒)
    
    Returns:
        缩略图存储路径或None
    """
    if not os.path.exists(video_path):
        return None
    
    # 生成缓存文件名
    file_hash = str(uuid.uuid5(uuid.NAMESPACE_URL, video_path))
    thumbnail_path = os.path.join(THUMBNAIL_CACHE_DIR, f"video_{file_hash}.jpg")
    
    # 如果缩略图已存在，直接返回
    if os.path.exists(thumbnail_path):
        return thumbnail_path
    
    try:
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        # 获取视频FPS和总帧数
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # 如果视频太短，取第一帧，否则取指定时间点
        target_frame = 0 if duration < timestamp else int(timestamp * fps)
        
        # 定位到指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        success, frame = cap.read()
        
        if not success:
            # 如果指定帧获取失败，尝试获取第一帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = cap.read()
        
        cap.release()
        
        if not success:
            return None
        
        # 转换为RGB（OpenCV默认是BGR）
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 调整大小
        img = Image.fromarray(frame)
        img.thumbnail(max_size)
        
        # 保存缩略图
        img.save(thumbnail_path, "JPEG", quality=80)
        return thumbnail_path
    
    except Exception as e:
        print(f"视频缩略图生成失败: {e}")
        return None

def get_audio_cover(audio_path, max_size=(320, 240)):
    """
    从音频文件中提取封面图片
    
    Args:
        audio_path: 音频文件路径
        max_size: 缩略图最大尺寸
    
    Returns:
        缩略图存储路径或None
    """
    if not os.path.exists(audio_path):
        return None
    
    # 生成缓存文件名
    file_hash = str(uuid.uuid5(uuid.NAMESPACE_URL, audio_path))
    thumbnail_path = os.path.join(THUMBNAIL_CACHE_DIR, f"audio_{file_hash}.jpg")
    
    # 如果缩略图已存在，直接返回
    if os.path.exists(thumbnail_path):
        return thumbnail_path
    
    try:
        ext = os.path.splitext(audio_path)[1].lower()
        cover_data = None
        
        # MP3文件
        if ext == '.mp3':
            try:
                audio = MP3(audio_path, ID3=ID3)
                if audio.tags:
                    for tag in audio.tags.values():
                        if tag.FrameID == 'APIC':
                            cover_data = tag.data
                            break
            except Exception as e:
                print(f"MP3封面提取错误: {e}")
        
        # FLAC文件
        elif ext == '.flac':
            try:
                audio = FLAC(audio_path)
                if audio.pictures:
                    cover_data = audio.pictures[0].data
            except Exception as e:
                print(f"FLAC封面提取错误: {e}")
        
        # MP4/M4A文件
        elif ext in ['.m4a', '.mp4']:
            try:
                audio = MP4(audio_path)
                if 'covr' in audio:
                    cover_data = audio['covr'][0]
            except Exception as e:
                print(f"MP4/M4A封面提取错误: {e}")
        
        # 如果找到封面数据，处理并保存
        if cover_data:
            img = Image.open(io.BytesIO(cover_data))
            img.thumbnail(max_size)
            img.save(thumbnail_path, "JPEG", quality=80)
            return thumbnail_path
        
        return None
    
    except Exception as e:
        print(f"音频封面提取失败: {e}")
        return None

# 获取任意媒体文件的缩略图
def get_media_thumbnail(file_path, media_type=None):
    """
    获取媒体文件的缩略图，支持视频和音频
    
    Args:
        file_path: 媒体文件路径
        media_type: 媒体类型(video/audio)，如不提供则自动识别
    
    Returns:
        缩略图路径或None
    """
    if not os.path.exists(file_path):
        return None
    
    # 如果未提供媒体类型，根据扩展名判断
    if not media_type:
        ext = os.path.splitext(file_path)[1].lower()
        video_exts = ['.mp4', '.webm', '.avi', '.mov', '.mkv', '.flv']
        audio_exts = ['.mp3', '.flac', '.m4a', '.wav', '.ogg']
        
        if ext in video_exts:
            media_type = 'video'
        elif ext in audio_exts:
            media_type = 'audio'
        else:
            return None
    
    # 根据媒体类型处理
    if media_type == 'video':
        return get_video_thumbnail(file_path)
    elif media_type == 'audio':
        return get_audio_cover(file_path)
    
    return None



def clean_thumbnail_cache(max_age_days: int = 1, batch_size: int = 5):
    """
    清理长时间未访问的缩略图缓存
    
    Args:
        max_age_days: 缩略图最大保留天数
        batch_size: 每次清理的文件数量
    
    Returns:
        清理的文件数量
    """
    if not os.path.exists(THUMBNAIL_CACHE_DIR):
        return 0
    
    now = time.time()
    max_age_seconds = max_age_days * 24 * 3600
    cleaned_count = 0
    
    # 获取所有缩略图文件
    thumbnail_files = [os.path.join(THUMBNAIL_CACHE_DIR, f) 
                      for f in os.listdir(THUMBNAIL_CACHE_DIR) 
                      if os.path.isfile(os.path.join(THUMBNAIL_CACHE_DIR, f))]
    
    # 按照修改时间排序，最旧的在前面
    thumbnail_files.sort(key=lambda x: os.path.getmtime(x))
    
    # 只处理指定批次数量的文件
    for file_path in thumbnail_files[:batch_size]:
        file_age = now - os.path.getmtime(file_path)
        
        # 如果文件超过最大保留时间，删除它
        if file_age > max_age_seconds:
            try:
                os.remove(file_path)
                cleaned_count += 1
                print(f"已清理缩略图: {os.path.basename(file_path)}")
            except Exception as e:
                print(f"清理缩略图失败: {e}")
    
    return cleaned_count

def start_thumbnail_cleanup_task(interval_minutes: int = 30, max_age_days: int = 7):
    """
    启动后台线程，定期清理缩略图缓存
    
    Args:
        interval_minutes: 检查间隔（分钟）
        max_age_days: 缩略图最大保留天数
    """
    def cleanup_worker():
        while True:
            try:
                # 每次只清理一个文件，减少系统负载
                clean_thumbnail_cache(max_age_days=max_age_days, batch_size=1)
                # 等待指定时间
                time.sleep(interval_minutes * 60)
            except Exception as e:
                print(f"缩略图清理任务异常: {e}")
                # 发生异常后短暂等待后继续
                time.sleep(60)
    
    # 创建守护线程，这样主程序退出时线程也会退出
    cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
    cleanup_thread.start()
    print(f"缩略图清理任务已启动，间隔: {interval_minutes}分钟, 最大保留时间: {max_age_days}天")
    
    return cleanup_thread


# 启动缩略图清理服务
def start_cleanup_service(interval_minutes=60, max_age_days=7, initial_cleanup=True):
    """启动缩略图清理服务"""
    if initial_cleanup:
        # 启动时先执行一次批量清理
        cleaned_count = clean_thumbnail_cache(max_age_days=0, batch_size=20)
        print(f"初始清理完成: 已清理 {cleaned_count} 个缩略图")
    
    # 启动后台定时清理任务
    return start_thumbnail_cleanup_task(interval_minutes=interval_minutes, 
                                      max_age_days=max_age_days)

