# ZAY-Cloud 云媒体服务器

ZAY-Cloud 是一个强大的本地云媒体服务器，让您可以轻松管理和访问您的文件、媒体和文档。它提供了直观的Web界面，支持多种文件类型的预览和播放，以及WebDAV网络访问功能。

**在本地会有打包的exe，也上传了Docker**
**- 注意：暂时没有添加登录验证**

![ZAY-Cloud 云媒体服务器](http://zay.204625.xyz/dl/2a925d9d-1c7c-4024-ae41-b28686d3d7c3)

## ✨ 主要特性

- 🎬 **强大的媒体播放功能**：支持各种视频格式播放，具有TikTok风格的滑动切换界面（**人话就是鼠标滚轮切换+上下方向键切换**）
- 🎵 **音频播放器**：内置音频播放器，支持波形显示和播放列表功能（**效果不好我给隐藏了**）
- 📱 **响应式设计**：在各种设备上都能提供良好的使用体验
- 📤 **高级上传功能**：
    - 支持大文件分片上传
    - 文件夹上传
    - 本地文件映射（无需上传即可添加文件）
- 🔗 **文件分享**：生成文件直链和二维码，方便分享
- 📂 **文件管理**：支持创建文件夹、删除、浏览等基本操作
- 📱 **外部播放器支持**：可以使用VLC、PotPlayer等外部播放器打开媒体文件
- ☁️ **WebDAV支持**：
    - 内置WebDAV服务器，可与Windows资源管理器、macOS Finder等客户端集成
    - WebDAV客户端功能，可连接其他WebDAV服务
- 🖼️ **自定义背景**：支持自定义应用背景图片(-**背景可以通过方向键快速切换**)
- 🔄 **多视图模式**：支持卡片视图和列表视图切换

## 🚀 快速开始

### 使用Docker安装

```bash
# 拉取镜像
docker pull fivif/zay-cloud:latest

# 运行容器
docker run -p 5888:5888 fivif/zay-cloud:latest
```

### 手动安装

1.  克隆此仓库

```bash
git clone https://github.com/fivif/zay-cloud.git
cd zay-cloud
```

2.  安装依赖

```bash
pip install -r requirements.txt
```

3.  运行应用

```bash
python main.py
```

4.  访问地址: `http://localhost:5888`

## 🛠️ 技术栈

- **后端**：FastAPI (Python)
- **前端**：HTML, CSS, JavaScript, Bootstrap 5
- **播放器**：Video.js, WaveSurfer.js
- **其他库**：qrcode.js, Font Awesome

## 📚 项目结构

```
zay-cloud/
├── config/                     # 配置文件
├── static/                     # 静态文件
│   ├── css/                    # 样式文件
│   ├── js/                     # JavaScript文件
│   └── logo/                   # 图标和Logo
├── storage/                    # 文件存储目录
├── templates/                  # HTML模板
├── Dockerfile                  # Docker配置
├── main.py                     # 主应用入口
├── webdav.py                   # WebDAV服务器实现
├── webdav_client.py            # WebDAV客户端实现
├── view.py                     # 视图和媒体处理函数
└── requirements.txt            # Python依赖
```

## 📝 使用说明

### 文件管理

- 点击文件夹进入子目录
- 使用面包屑导航返回上级目录
- 点击"新建文件夹"按钮创建目录
- 点击"上传"按钮上传文件或文件夹

### 媒体播放

- 点击视频文件列表区域可直接预览播放，还可通过鼠标和方向键切换同一个文件夹的视频。
- 点击音频文件播放音乐
- 在专门的播放界面可以：
    - 上下滑动或使用方向键切换媒体文件
    - 调整音量、播放速度
    - 切换全屏模式
    - 使用外部播放器打开

### WebDAV功能

- 在侧边栏点击"WebDAV设置"配置WebDAV服务器
- 使用"WebDAV客户端"连接其他WebDAV服务

## 📸 截图
<img width="2541" height="1243" alt="image" src="https://github.com/user-attachments/assets/a83312b4-5bb1-4abb-a1c8-16b8c3dd68ea" />



&nbsp;

## 📄 许可证

本项目采用 MIT 许可证 - 详情请参见 LICENSE 文件

## 💡 未来计划

- [ ] 移动应用更好的支持
- [ ] 多用户支持和权限系统
- [ ] 更多格式的转码支持
- [ ] 更好的媒体播放效果
- [ ] 支持更多设备

## 📧 联系方式

如有任何问题或建议，请通过以下方式联系：

- 邮箱：fivif@outlook.com
- GitHub Issues：[问题报告](https://github.com/fivif/zay-cloud/issues)

* * *

💻 由 ZAY 开发 | 祝您使用愉快！"# zay-cloud"  
