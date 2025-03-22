// 作者: ZAY
let currentPath = "";
let currentCategory = "all";
let fileListCache = null;

// 初始化
document.addEventListener("DOMContentLoaded", () => {
    refreshFileList();
    switchCategory("all");
});

// 显示提示窗口
function showNotification(message, isSuccess) {
    const notification = document.getElementById("notification");
    notification.textContent = message;
    notification.className = `notification ${isSuccess ? "success" : "error"}`;
    notification.style.display = "block";
    setTimeout(() => notification.style.display = "none", 3000);
}

async function refreshFileList() {
try {
const container = document.getElementById("itemsContainer");
if (container) container.innerHTML = '<p class="text-center"><i class="fas fa-spinner fa-spin"></i> 加载中...</p>';

console.log(`请求路径: ${currentPath}`);
const response = await fetch(`/api/files?path=${encodeURIComponent(currentPath)}`);
if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`服务器返回错误 (${response.status}): ${errorText}`);
}

const data = await response.json();
console.log("API返回数据:", data);

if (!Array.isArray(data.items)) data.items = [];
if (!data.categories) data.categories = {videos: [], audios: [], images: [], documents: [], others: []};

// 验证返回的 current_path
if (data.current_path !== currentPath) {

    console.warn(`后端返回路径(${data.current_path})与前端预期(${currentPath})不一致`);
    currentPath = data.current_path || currentPath; // 以后端为准
}

fileListCache = data;
renderItems(data.items, data.categories);

updateBreadcrumb(currentPath);
} catch (error) {
console.error("刷新文件列表失败:", error);
showNotification(`加载文件列表失败: ${error.message}`, false);

const container = document.getElementById("itemsContainer");
if (container) container.innerHTML = `<p class="text-danger">加载失败: ${error.message}</p>`;

if (currentPath && currentPath.includes('/')) {
    const parentPath = currentPath.split('/').slice(0, -1).join('/');
    showNotification(`尝试回到上一级目录: ${parentPath || '根目录'}`, false);
    setTimeout(() => navigateTo(parentPath), 1000);
}
}
}


// 渲染文件和文件夹
function renderItems(items, categories) {
    const isCardView = document.getElementById('fileList').classList.contains('card-view-mode');
    const container = document.getElementById("itemsContainer");

    if (!container) {
        console.error("找不到itemsContainer元素");
        return;
    }

    container.innerHTML = "";

    // 检查items和categories是否有效
    if (!Array.isArray(items)) {
        console.error("items不是数组:", items);
        items = [];
    }

    if (!categories || typeof categories !== 'object') {
        console.error("categories无效:", categories);
        categories = {videos: [], audios: [], images: [], documents: [], others: []};
    }

    // 根据当前类别筛选项目
    let filteredItems = [];
    if (currentCategory === "all") {
        filteredItems = items;
    } else if (categories[currentCategory] && Array.isArray(categories[currentCategory])) {
        filteredItems = categories[currentCategory];
    } else {
        console.warn(`找不到类别 ${currentCategory} 或其不是数组`);
        filteredItems = [];
    }

    // 过滤系统文件夹，如背景图文件夹
    filteredItems = filteredItems.filter(item => {
        
        // 如果是文件夹，检查是否为系统文件夹
        if (item && item.type === "folder") {
            // 隐藏以下文件夹（可根据需要添加）
            const systemFolders = [
                "backgrounds", 
                ".backgrounds", 
                "system_backgrounds", 
                ".system"
            ];
            const folderName = item.name ? item.name.toLowerCase() : "";
            return !systemFolders.includes(folderName);
        }
        return true; // 保留所有非文件夹项目
    });

    // 排序，让文件夹显示在文件前面
    filteredItems.sort((a, b) => {
        // 如果a是文件夹而b不是，a应该排在前面
        if (a.type === "folder" && b.type !== "folder") return -1;
        // 如果b是文件夹而a不是，b应该排在前面
        if (b.type === "folder" && a.type !== "folder") return 1;
        // 其他情况（两者都是文件夹或都是文件）保持原有顺序
        return 0;
    });

    if (filteredItems.length === 0) {
        container.innerHTML = '<p class="text-muted">暂无内容</p>';
        return;
    }

    // 渲染每个项目
    filteredItems.forEach(item => {
        if (!item || typeof item !== 'object') {
            console.warn("跳过无效的项目:", item);
            return;
        }

        const div = document.createElement("div");
        div.className = item.type === "folder" ? "file-item folder-item" : "file-item";
        const icon = item.type === "folder" ? "fas fa-folder" : getFileIcon(item.media_type);

        // 确保path属性存在
        let th = item.path || "";
        let path = th.replace(/\\/g, "/"); 
        console.log(`路径: ${path}`);

        // 创建点击事件处理函数
        let itemClickHandler = "";
        if (item.type === 'folder') {
            itemClickHandler = `navigateTo('${path}')`;
        } else if (item.media_type === 'image') {
            itemClickHandler = `previewImage('${path}', '${item.name}')`;
        } else if (item.media_type === 'audio') {
            itemClickHandler = `playAudio('${path}', '${item.name}')`;
        } else if (item.media_type === 'video') {
            itemClickHandler = `playVideo('${path}', '${item.name}')`;
        }

        // 根据视图模式生成不同的HTML结构
        if (isCardView) {
            // 卡片视图模式
            let previewHtml = '';
            
            if (item.type === 'file') {
                if (item.media_type === 'image') {
                    // 图片文件显示缩略图
                    previewHtml = `
                        <div class="file-preview" onclick="${itemClickHandler}">
                            <img src="/view/${path}" alt="${item.name}" loading="lazy">
                        </div>`;
                } else if (item.media_type === 'video') {
                    // 视频文件显示截图预览
                    previewHtml = `
                        <div class="file-preview" onclick="${itemClickHandler}">
                            <img src="/api/thumbnail/video/${path}" alt="${item.name}" loading="lazy" 
                                 onerror="this.onerror=null; this.src=''; this.parentNode.innerHTML='<i class=\'fas fa-video fa-3x\' style=\'color: rgba(0, 0, 0, 0.4);\'></i>';">
                        </div>`;
                } else if (item.media_type === 'audio') {
                    // 音频文件显示封面预览
                    previewHtml = `
                        <div class="file-preview" onclick="${itemClickHandler}">
                            <img src="/api/thumbnail/audio/${path}" alt="${item.name}" loading="lazy" 
                                 onerror="this.onerror=null; this.src=''; this.parentNode.innerHTML='<i class=\'fas fa-music fa-3x\' style=\'color: rgba(0, 0, 0, 0.4);\'></i>';">
                        </div>`;
                } else {
                    // 其他类型文件显示图标
                    let bigIconClass = `${icon} fa-3x`;
                    previewHtml = `
                        <div class="file-preview" onclick="${itemClickHandler}">
                            <i class="${bigIconClass}" style="color: rgba(0, 0, 0, 0.4);"></i>
                        </div>`;
                }
            } else {
                // 文件夹显示图标
                previewHtml = `
                    <div class="file-preview" onclick="${itemClickHandler}">
                        <i class="fas fa-folder fa-3x" style="color: rgba(0, 0, 0, 0.4);"></i>
                    </div>`;
            }
            div.innerHTML = `
            ${previewHtml}
            <div class="file-name">${item.name || "未命名"}</div>
            <div class="file-info" style="margin-bottom: 30px;">
                
            </div>
            <div class="file-actions" style="bottom: 15px;">
                ${item.type === "file" ? `
                    <a href="/download/${path}" class="btn btn-sm btn-primary me-1" title="下载"><i class="fas fa-download"></i></a>
                    ${item.is_video ? `<a href="/play/${path}" target="_blank" class="btn btn-sm btn-success me-1" title="播放"><i class="fas fa-play"></i></a>` : ""}
                    <button class="btn btn-sm btn-info me-1" onclick="generateDirectLink('${path}')" title="获取链接"><i class="fas fa-link"></i></button>
                    <button class="btn btn-sm btn-secondary me-1" onclick="showFileQRCode('${path}')" title="显示二维码"><i class="fas fa-qrcode"></i></button>
                ` : ""}
                <button class="btn btn-sm btn-danger" onclick="${item.type === 'folder' ? `deleteFolder('${path}')` : `deleteFile('${path}')`}" title="删除"><i class="fas fa-trash"></i></button>
            </div>`;
    } else {
        // 列表视图模式
        div.innerHTML = `
            <div class="d-flex justify-content-between align-items-center">
                <div onclick="${itemClickHandler}" style="${item.type !== 'folder' ? 'cursor: pointer;' : ''}" class="text-enhanced">
                    <i class="${icon} file-icon"></i> ${item.name || "未命名"}
                    ${item.type === "file" && item.size !== undefined ? `<small class="text-muted ms-2">(${formatFileSize(item.size)})</small>` : ""}
                </div>
                <div>
                    ${item.type === "file" ? `
                        <a href="/download/${path}" class="btn btn-sm btn-primary me-1"><i class="fas fa-download"></i></a>
                        ${item.is_video  ? `<a href="/play/${path}" target="_blank" class="btn btn-sm btn-success me-1"><i class="fas fa-play"></i></a>` : ""}
                        <button class="btn btn-sm btn-info me-1" onclick="generateDirectLink('${path}')"><i class="fas fa-link"></i></button>
                        <button class="btn btn-sm btn-secondary me-1" onclick="showFileQRCode('${path}')"><i class="fas fa-qrcode"></i></button>
                    ` : ""}
                    <button class="btn btn-sm btn-danger" onclick="${item.type === 'folder' ? `deleteFolder('${path}')` : `deleteFile('${path}')`}"><i class="fas fa-trash"></i></button>
                </div>
            </div>`;
    }
        
        container.appendChild(div);
    });
}

// 显示文件直链二维码
async function showFileQRCode(filePath) {
    try {
        console.log(`生成文件二维码: ${filePath}`);
        
        // 先获取文件直链
        const response = await fetch(`/api/direct-link/${filePath}`);
        const result = await response.json();
        
        if (!response.ok) {
            throw new Error(result.detail || '获取直链失败');
        }
        
        const linkText = result.full_link;
        const filename = filePath.split('/').pop();
        
        // 尝试使用Bootstrap模态框
        try {
            // 使用新的fileLinkQrcodeModal
            const modal = new bootstrap.Modal(document.getElementById('fileLinkQrcodeModal'));
            const modalTitle = document.getElementById('fileLinkQrcodeModalLabel');
            const qrcodeContainer = document.getElementById('fileLinkQrcodeContainer');
            const linkTextElement = document.getElementById('fileLinkText');
            
            // 修改标题和清理容器
            modalTitle.innerHTML = `<i class="fas fa-qrcode me-2"></i>文件直链：${filename}`;
            qrcodeContainer.innerHTML = '';
            
            // 生成二维码
            new QRCode(qrcodeContainer, {
                text: linkText,
                width: 256,
                height: 256,
                colorDark: "#000000",
                colorLight: "#ffffff",
                correctLevel: QRCode.CorrectLevel.H
            });
            
            // 显示链接文本
            linkTextElement.textContent = linkText;
            
            // 设置复制按钮事件
            document.getElementById('copyLinkBtn').onclick = function() {
                copyToClipboard(linkText);
                showNotification('链接已复制到剪贴板', true);
            };
            
            // 显示模态框
            modal.show();
        } catch (modalError) {
            console.warn('Bootstrap模态框加载失败，使用备选弹窗:', modalError);
            // 使用备选的自定义弹窗
            showCustomQRCodePopup(linkText, filename);
        }
    } catch (error) {
        console.error('生成文件二维码失败:', error);
        showNotification(`生成二维码失败: ${error.message}`, false);
    }
}


// 备选方案：自定义二维码弹窗
function showCustomQRCodePopup(linkText, filename) {
    // 移除之前的弹窗（如果有）
    const oldPopup = document.getElementById('file-link-qrcode-popup');
    if (oldPopup) {
        document.body.removeChild(oldPopup);
    }
    const oldOverlay = document.getElementById('file-link-qrcode-overlay');
    if (oldOverlay) {
        document.body.removeChild(oldOverlay);
    }
    
    // 创建背景遮罩
    const overlay = document.createElement('div');
    overlay.style.position = 'fixed';
    overlay.style.top = '0';
    overlay.style.left = '0';
    overlay.style.width = '100%';
    overlay.style.height = '100%';
    overlay.style.backgroundColor = 'rgba(0,0,0,0.5)';
    overlay.style.zIndex = '9998';
    overlay.id = 'file-link-qrcode-overlay';
    
    // 创建弹窗容器
    const popup = document.createElement('div');
    popup.id = 'file-link-qrcode-popup';
    popup.style.position = 'fixed';
    popup.style.top = '50%';
    popup.style.left = '50%';
    popup.style.transform = 'translate(-50%, -50%)';
    popup.style.backgroundColor = 'white';
    popup.style.padding = '20px';
    popup.style.borderRadius = '8px';
    popup.style.boxShadow = '0 5px 15px rgba(0,0,0,0.3)';
    popup.style.zIndex = '9999';
    popup.style.maxWidth = '90%';
    popup.style.width = '350px';
    popup.style.textAlign = 'center';
    
    // 添加标题和关闭按钮
    const header = document.createElement('div');
    header.style.display = 'flex';
    header.style.justifyContent = 'space-between';
    header.style.alignItems = 'center';
    header.style.marginBottom = '15px';
    
    const title = document.createElement('h5');
    title.innerHTML = `<i class="fas fa-qrcode" style="margin-right:8px;"></i>文件直链：${filename}`;
    title.style.margin = '0';
    
    const closeBtn = document.createElement('button');
    closeBtn.innerHTML = '&times;';
    closeBtn.style.background = 'none';
    closeBtn.style.border = 'none';
    closeBtn.style.fontSize = '22px';
    closeBtn.style.cursor = 'pointer';
    closeBtn.style.padding = '0 5px';
    closeBtn.onclick = closePopup;
    
    header.appendChild(title);
    header.appendChild(closeBtn);
    popup.appendChild(header);
    
    // 添加二维码容器
    const qrContainer = document.createElement('div');
    qrContainer.style.margin = '15px auto';
    popup.appendChild(qrContainer);
    
    // 生成二维码
    new QRCode(qrContainer, {
        text: linkText,
        width: 256,
        height: 256,
        colorDark: "#000000",
        colorLight: "#ffffff",
        correctLevel: QRCode.CorrectLevel.H
    });
    
    // 添加链接文本
    const linkElement = document.createElement('div');
    linkElement.textContent = linkText;
    linkElement.style.margin = '10px 0';
    linkElement.style.wordBreak = 'break-all';
    linkElement.style.fontSize = '12px';
    linkElement.style.color = '#555';
    linkElement.style.padding = '5px';
    linkElement.style.backgroundColor = '#f8f9fa';
    linkElement.style.borderRadius = '4px';
    popup.appendChild(linkElement);
    
    // 添加按钮区域
    const buttonArea = document.createElement('div');
    buttonArea.style.marginTop = '15px';
    
    // 添加复制按钮
    const copyButton = document.createElement('button');
    copyButton.innerHTML = '<i class="fas fa-copy" style="margin-right:5px;"></i>复制链接';
    copyButton.style.padding = '8px 16px';
    copyButton.style.marginRight = '10px';
    copyButton.style.background = '#007bff';
    copyButton.style.color = 'white';
    copyButton.style.border = 'none';
    copyButton.style.borderRadius = '4px';
    copyButton.style.cursor = 'pointer';
    copyButton.onclick = function() {
        copyToClipboard(linkText);
        showNotification('链接已复制到剪贴板', true);
    };
    buttonArea.appendChild(copyButton);
    
    // 添加关闭按钮
    const closeButton = document.createElement('button');
    closeButton.textContent = '关闭';
    closeButton.style.padding = '8px 16px';
    closeButton.style.background = '#6c757d';
    closeButton.style.color = 'white';
    closeButton.style.border = 'none';
    closeButton.style.borderRadius = '4px';
    closeButton.style.cursor = 'pointer';
    closeButton.onclick = closePopup;
    buttonArea.appendChild(closeButton);
    
    popup.appendChild(buttonArea);
    
    // 点击遮罩层关闭弹窗
    overlay.onclick = function(e) {
        if (e.target === overlay) {
            closePopup();
        }
    };
    
    function closePopup() {
        document.body.removeChild(overlay);
        document.body.removeChild(popup);
    }
    
    // 添加到文档
    document.body.appendChild(overlay);
    document.body.appendChild(popup);
}


// 获取文件图标
function getFileIcon(mediaType) {
    switch (mediaType) {
        case "video": return "fas fa-video";
        case "audio": return "fas fa-music";
        case "image": return "fas fa-image";
        case "document": return "fas fa-file-alt";
        default: return "fas fa-file";
    }
}

// 格式化文件大小
function formatFileSize(bytes) {
    if (bytes === 0) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
}

// 更新面包屑导航
function updateBreadcrumb(path) {
    const breadcrumb = document.getElementById("breadcrumb");
    breadcrumb.innerHTML = "";
    
    // 规范化路径 - 去除开头和结尾的斜杠
    path = (path || "").replace(/^\/+|\/+$/g, "");
    const parts = path ? path.split("/").filter(Boolean) : [];
    let cumulativePath = "";

    const homeItem = document.createElement("li");
    homeItem.className = "breadcrumb-item";
    homeItem.innerHTML = `<a href="#" onclick="navigateTo(''); return false;">首页</a>`;
    breadcrumb.appendChild(homeItem);

    parts.forEach((part, index) => {
        // 修改这里 - 直接添加部分到路径而不添加额外斜杠
        console.log(`面包屑路径项: ${part}, 累积路径: ${cumulativePath}`);
        cumulativePath += (cumulativePath ? "/" : "") + part;
        console.log(cumulativePath)
        const item = document.createElement("li");
        item.className = "breadcrumb-item";
        
        console.log(`面包屑路径项: ${part}, 累积路径: ${cumulativePath}`);
        
        if (index === parts.length - 1) {
            item.innerHTML = part;
            item.className = "breadcrumb-item active";
        } else {
            
            item.innerHTML = `<a href="#" onclick="navigateTo('${cumulativePath}'); return false;">${part}</a>`;
        }
        breadcrumb.appendChild(item);
    });
}

// 导航到指定路径
function navigateTo(path) {
console.log(`尝试导航到原始路径: ${path}`);
currentPath = (path || "").replace(/\/+/g, "/").replace(/^\/+|\/+$/g, "");
console.log(`规范化后的路径: ${currentPath}`);
console.log(`即将请求: /api/files?path=${encodeURIComponent(currentPath)}`);
refreshFileList();
}




// 切换分类
function switchCategory(category) {
    currentCategory = category;
    document.querySelectorAll(".category-tab").forEach(tab => tab.classList.remove("active"));
    document.querySelector(`[data-category="${category}"]`).classList.add("active");
    if (fileListCache) renderItems(fileListCache.items, fileListCache.categories); // 使用缓存渲染
    else refreshFileList();
}

// 上传文件
async function uploadFile() {
    const form = document.getElementById("uploadForm");
    const formData = new FormData(form);
    formData.set("directory", currentPath);
    const modal = bootstrap.Modal.getInstance(document.getElementById("uploadModal"));
    modal.hide(); // 立即关闭模态框
    try {
        const response = await fetch("/upload", { method: "POST", body: formData });
        const result = await response.json();
        if (response.ok) {
            showNotification(result.message, true);
            refreshFileList();
        } else {
            showNotification(result.error, false);
        }
    } catch (error) {
        showNotification(`上传失败: ${error}`, false);
    }
}

// 创建文件夹（支持多级）
async function createFolder() {
    let folderName = document.getElementById("folderName").value.trim();
    if (!folderName) {
    showNotification("文件夹名称不能为空", false);
    return;
    }
    
    // 清理输入，允许中文字符和其他常用字符，但过滤掉特殊字符
    folderName = folderName.replace(/[<>:"|?*\\]/g, "").replace(/^\/+|\/+$/g, "");
    if (!folderName) {
    showNotification("文件夹名称包含非法字符，已被清理为空", false);
    return;
    }
    
    let folderPath = currentPath ? `${currentPath}/${folderName}` : folderName;
    folderPath = folderPath.replace(/\/+/g, "/"); // 规范化路径
    
    console.log(`创建文件夹路径: ${folderPath}`);
    
    const modal = bootstrap.Modal.getInstance(document.getElementById("createFolderModal"));
    modal.hide();

try {
const response = await fetch("/api/folders", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: `folder_path=${encodeURIComponent(folderPath)}`
});
const result = await response.json();

if (response.ok) {
    showNotification(result.message, true);
    
    navigateTo(folderPath); // 导航到完整路径
} else {
    showNotification(result.detail || result.error, false);
}
} catch (error) {
showNotification(`创建失败: ${error}`, false);
}

document.getElementById("folderName").value = "";
}



// 删除文件
async function deleteFile(filePath) {
    try {
        const response = await fetch(`/delete/${filePath}`, { method: "DELETE" });
        const result = await response.json();
        if (response.ok) {
            showNotification(result.message, true);
            refreshFileList();
        } else {
            showNotification(result.error, false);
        }
    } catch (error) {
        showNotification(`删除失败: ${error}`, false);
    }
}

// 删除文件夹
async function deleteFolder(folderPath) {
    try {
        const response = await fetch(`/api/folders/${folderPath}`, { method: "DELETE" });
        const result = await response.json();
        if (response.ok) {
            showNotification(result.message, true);
            refreshFileList();
        } else {
            showNotification(result.error, false);
        }
    } catch (error) {
        showNotification(`删除失败: ${error}`, false);
    }
}

// 生成直链并自动复制
// 替换当前的 generateDirectLink 函数

// 生成直链并自动复制
async function generateDirectLink(filePath) {
try {
console.log(`请求直链: ${filePath}`);
const response = await fetch(`/api/direct-link/${filePath}`);
const result = await response.json();
if (response.ok) {
    const linkText = result.full_link;
    // 尝试使用现代 Clipboard API 和备用方法
    if (copyToClipboard(linkText)) {
        showNotification("直链已复制到剪贴板", true);
    } else {
        // 如果复制失败，展示链接让用户可以手动复制
        showNotification(`直链生成成功: ${linkText}`, true);
    }
} else {
    showNotification(result.detail, false);
}
} catch (error) {
showNotification(`生成直链失败: ${error}`, false);
}
}

// 兼容多种浏览器的剪贴板复制函数
function copyToClipboard(text) {
// 先尝试使用现代 Clipboard API
if (navigator.clipboard && navigator.clipboard.writeText) {
navigator.clipboard.writeText(text).catch(err => {
    console.warn('Clipboard API failed:', err);
});
return true;
}

// 备用方法：创建临时元素
try {
const textarea = document.createElement('textarea');
textarea.value = text;

// 使元素不可见
textarea.style.position = 'fixed';
textarea.style.opacity = '0';

document.body.appendChild(textarea);
textarea.select();

// 尝试复制
const successful = document.execCommand('copy');
document.body.removeChild(textarea);
return successful;
} catch (err) {
console.error('Fallback clipboard method failed:', err);
return false;
}
}

// 图片预览功能
function previewImage(path, name) {
const modal = new bootstrap.Modal(document.getElementById('imagePreviewModal'));
const modalTitle = document.getElementById('imagePreviewModalLabel');
const previewImage = document.getElementById('previewImage');

// 设置模态框标题和图片源
modalTitle.textContent = `图片预览: ${name}`;
previewImage.src = `/view/${path}`;

// 显示模态框
modal.show();
}

// 音频播放功能
function playAudio(path, name) {
const modal = new bootstrap.Modal(document.getElementById('audioPlayerModal'));
const modalTitle = document.getElementById('audioPlayerModalLabel');
const audioPlayer = document.getElementById('audioPlayer');

// 设置模态框标题和音频源
modalTitle.textContent = `正在播放: ${name}`;
audioPlayer.querySelector('source').src = `/view/${path}`;
audioPlayer.load(); // 重新加载音频

// 显示模态框
modal.show();

// 模态框显示后自动播放
modal._element.addEventListener('shown.bs.modal', function () {
audioPlayer.play().catch(e => console.warn('自动播放失败:', e));
}, { once: true });
}

// 视频播放功能
function playVideo(path, name) {
const modal = new bootstrap.Modal(document.getElementById('videoPlayerModal'));
const modalTitle = document.getElementById('videoPlayerModalLabel');
const videoPlayer = document.getElementById('videoPlayer');

// 设置模态框标题和视频源

videoPlayer.querySelector('source').src = `/view/${path}`;
videoPlayer.load(); // 重新加载视频

// 显示模态框
modal.show();

// 模态框显示后自动播放
modal._element.addEventListener('shown.bs.modal', function () {
videoPlayer.play().catch(e => console.warn('自动播放失败:', e));
}, { once: true });
}



// 高级上传管理系统
const uploadManager = {
uploads: [],
totalUploads: 0,
completedUploads: 0,

// 添加新上传任务
addUpload: function(file, directory) {
const uploadId = `upload-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
const upload = {
    id: uploadId,
    file: file,
    fileName: file.name,
    fileSize: file.size,
    directory: directory,
    progress: 0,
    uploadedBytes: 0,
    speed: 0,
    lastUpdateTime: Date.now(),
    status: 'pending', // pending, uploading, paused, completed, error
    startTime: Date.now(),
    chunks: [],
    currentChunkIndex: 0,
    totalChunks: 0,
    chunkSize: 10 * 1024 * 1024, // 2MB 分片大小
    enableChunking: file.size > 1000 * 1024 * 1024, // 1000MB以上文件自动启用分片
    error: null,
    controller: new AbortController() // 用于取消上传
};

this.uploads.push(upload);
this.totalUploads++;
this.updateUploadButton();
this.renderUploads();

return upload;
},
// 在 uploadManager 对象中添加本地映射方法
_checkAndMapLocalFile: async function(upload) {
// 如果用户关闭了本地映射选项，则跳过检查
if (!upload.checkLocalMapping) {
console.log('本地映射检查已禁用');
return false;
}

try {
console.log(`检查文件 ${upload.fileName} 是否可映射...`);
// 调用后端 API 检查文件是否可以直接映射
const response = await fetch('/api/check-local-file', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        fileName: upload.fileName,
        directory: upload.directory,
        fileSize: upload.fileSize,
        // 可以传递一些用于识别本地文件的额外信息
        lastModified: upload.file.lastModified
    })
});

if (!response.ok) {
    console.log('检查映射API返回错误，将进行正常上传');
    return false; // 不能映射，需要正常上传
}

const result = await response.json();
console.log('映射检查结果:', result);

if (result.canMap && result.localPath) {
    // 文件可以映射，调用映射 API
    console.log(`文件可映射，正在映射本地文件: ${result.localPath}`);
    const mapResponse = await fetch('/api/map-local-file', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            fileName: upload.fileName,
            directory: upload.directory,
            localPath: result.localPath
        })
    });
    
    if (mapResponse.ok) {
        const mapResult = await mapResponse.json();
        console.log('映射成功:', mapResult);
        upload.status = 'completed';
        upload.progress = 100;
        upload.endTime = Date.now();
        this.completedUploads++;
        
        showNotification(`文件 ${upload.fileName} 已直接映射（无需上传）`, true);
        refreshFileList();
        
        // 5秒后从列表中移除已完成的操作
        setTimeout(() => {
            this.removeUpload(upload.id);
        }, 5000);
        
        this.renderUploads();
        this.updateUploadButton();
        
        return true; // 成功映射
    } else {
        console.warn('映射API调用失败，将进行正常上传');
    }
} else {
    console.log('文件不可映射，将进行正常上传');
}

return false; // 不能映射，需要正常上传
} catch (error) {
console.error('检查本地文件映射失败:', error);
return false; // 出错时默认使用正常上传
}
},
// 开始上传任务
// 在现有 startUpload 方法中添加本地映射检查
startUpload: async function(upload) {
if (upload.status === 'uploading') return;

upload.status = 'uploading';
upload.startTime = Date.now();
upload.lastUpdateTime = Date.now();

this.renderUploads(); // 立即更新UI以显示"上传中"状态

try {
// 首先检查是否可以直接映射本地文件
console.log(`尝试映射本地文件: ${upload.fileName}`);
const mapped = await this._checkAndMapLocalFile(upload);

if (mapped) {
    // 文件已经映射完成，无需上传
    console.log(`文件 ${upload.fileName} 已成功映射，跳过上传步骤`);
    return; // 直接返回，不执行后续上传代码
}

console.log(`文件 ${upload.fileName} 无法映射，开始正常上传流程`);
// 如果不能映射，继续原有的上传逻辑
if (upload.enableChunking) {
    this._prepareChunks(upload);
    this._uploadNextChunk(upload);
} else {
    this._uploadFile(upload);
}
} catch (error) {
console.error(`上传失败: ${error.message}`, error);
upload.status = 'error';
upload.error = error.message || '上传准备过程中出错';
this.renderUploads();
}
},

// 准备文件分片
_prepareChunks: function(upload) {
const file = upload.file;
const chunkSize = upload.chunkSize;
const totalChunks = Math.ceil(file.size / chunkSize);

upload.totalChunks = totalChunks;
upload.chunks = Array(totalChunks).fill().map((_, index) => ({
    index: index,
    start: index * chunkSize,
    end: Math.min((index + 1) * chunkSize, file.size),
    status: 'pending', // pending, uploading, completed, error
    attempts: 0,
    progress: 0
}));

console.log(`准备上传文件: ${file.name}, 大小: ${this._formatFileSize(file.size)}, 分片数: ${totalChunks}`);
},

// 上传下一个分片
_uploadNextChunk: async function(upload) {
if (upload.status !== 'uploading') return;

// 查找下一个待上传的分片
const nextChunk = upload.chunks.find(chunk => chunk.status === 'pending');
if (!nextChunk) {
    // 所有分片已上传，完成文件
    await this._completeMultipartUpload(upload);
    return;
}

nextChunk.status = 'uploading';
upload.currentChunkIndex = nextChunk.index;

try {
    await this._uploadChunk(upload, nextChunk);
    nextChunk.status = 'completed';
    nextChunk.progress = 100;
    
    // 更新整体进度
    const completedChunks = upload.chunks.filter(c => c.status === 'completed').length;
    upload.progress = Math.round((completedChunks / upload.totalChunks) * 100);
    upload.uploadedBytes = upload.chunks.reduce((total, chunk) => {
        return total + (chunk.status === 'completed' ? (chunk.end - chunk.start) : 0);
    }, 0);
    
    // 计算上传速度
    const now = Date.now();
    const timeDiff = (now - upload.lastUpdateTime) / 1000; // 秒
    if (timeDiff > 0) {
        const bytesPerSecond = (upload.uploadedBytes - (upload.lastBytes || 0)) / timeDiff;
        upload.speed = bytesPerSecond;
        upload.lastBytes = upload.uploadedBytes;
        upload.lastUpdateTime = now;
    }
    
    this.renderUploads();
    
    // 继续上传下一个分片
    this._uploadNextChunk(upload);
} catch (error) {
    console.error(`分片 ${nextChunk.index} 上传失败:`, error);
    nextChunk.status = 'error';
    nextChunk.attempts++;
    
    if (nextChunk.attempts < 3) {
        // 重试
        console.log(`重试分片 ${nextChunk.index}, 第 ${nextChunk.attempts} 次尝试`);
        nextChunk.status = 'pending';
        this._uploadNextChunk(upload);
    } else {
        // 放弃该分片，标记整个上传失败
        upload.status = 'error';
        upload.error = error.message || '上传失败，多次重试后仍无法完成';
        this.renderUploads();
    }
}
},

// 上传单个分片
_uploadChunk: async function(upload, chunk) {
const file = upload.file;
const formData = new FormData();
const blob = file.slice(chunk.start, chunk.end);

formData.append('file', blob, file.name);
formData.append('directory', upload.directory);
formData.append('fileName', file.name);
formData.append('chunkIndex', chunk.index);
formData.append('totalChunks', upload.totalChunks);
formData.append('chunkSize', upload.chunkSize);
formData.append('fileSize', file.size);

const response = await fetch('/api/upload/chunk', {
    method: 'POST',
    body: formData,
    signal: upload.controller.signal
});

if (!response.ok) {
    const result = await response.json();
    throw new Error(result.detail || '上传分片失败');
}

return await response.json();
},

// 完成分片上传
_completeMultipartUpload: async function(upload) {
try {
    console.log(`所有分片上传完成，合并文件: ${upload.fileName}`);
    
    const response = await fetch('/api/upload/complete', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            fileName: upload.fileName,
            directory: upload.directory,
            totalChunks: upload.totalChunks,
            fileSize: upload.fileSize
        })
    });
    
    if (!response.ok) {
        const result = await response.json();
        throw new Error(result.detail || '合并文件失败');
    }
    
    const result = await response.json();
    upload.status = 'completed';
    upload.progress = 100;
    upload.endTime = Date.now();
    this.completedUploads++;
    
    showNotification(`文件 ${upload.fileName} 上传成功`, true);
    refreshFileList();
    
    // 5秒后从列表中移除已完成的上传
    setTimeout(() => {
        this.removeUpload(upload.id);
    }, 5000);
} catch (error) {
    console.error('完成分片上传失败:', error);
    upload.status = 'error';
    upload.error = error.message || '合并文件失败';
    showNotification(`文件 ${upload.fileName} 上传失败: ${upload.error}`, false);
}

this.renderUploads();
this.updateUploadButton();
},

// 直接上传整个文件（不分片）
_uploadFile: async function(upload) {
try {
    const formData = new FormData();
    formData.append('file', upload.file);
    formData.append('directory', upload.directory);
    
    // 创建和设置XHR请求
    const xhr = new XMLHttpRequest();
    
    // 设置进度事件
    xhr.upload.addEventListener('progress', (event) => {
        if (event.lengthComputable) {
            upload.progress = Math.round((event.loaded / event.total) * 100);
            upload.uploadedBytes = event.loaded;
            
            // 计算上传速度
            const now = Date.now();
            const timeDiff = (now - upload.lastUpdateTime) / 1000; // 秒
            if (timeDiff > 0) {
                const bytesPerSecond = (event.loaded - (upload.lastBytes || 0)) / timeDiff;
                upload.speed = bytesPerSecond;
                upload.lastBytes = event.loaded;
                upload.lastUpdateTime = now;
            }
            
            this.renderUploads();
        }
    });
    
    // 返回Promise以便处理结果
    await new Promise((resolve, reject) => {
        xhr.addEventListener('load', () => {
            if (xhr.status >= 200 && xhr.status < 300) {
                resolve(JSON.parse(xhr.responseText));
            } else {
                let errorMsg = "上传失败";
                try {
                    const response = JSON.parse(xhr.responseText);
                    errorMsg = response.error || response.detail || errorMsg;
                } catch (e) {
                    errorMsg = `上传失败 (${xhr.status})`;
                }
                reject(new Error(errorMsg));
            }
        });
        
        xhr.addEventListener('error', () => {
            reject(new Error("网络错误，上传失败"));
        });
        
        xhr.addEventListener('abort', () => {
            reject(new Error("上传已取消"));
        });
        
        xhr.open("POST", "/upload");
        xhr.send(formData);
        
        // 保存xhr以便可以取消
        upload.xhr = xhr;
    });
    
    // 上传成功
    upload.status = 'completed';
    upload.progress = 100;
    upload.endTime = Date.now();
    this.completedUploads++;
    
    showNotification(`文件 ${upload.fileName} 上传成功`, true);
    refreshFileList();
    
    // 5秒后从列表中移除已完成的上传
    setTimeout(() => {
        this.removeUpload(upload.id);
    }, 5000);
} catch (error) {
    console.error('文件上传失败:', error);
    upload.status = 'error';
    upload.error = error.message || '上传失败';
    showNotification(`文件 ${upload.fileName} 上传失败: ${upload.error}`, false);
}

this.renderUploads();
this.updateUploadButton();
},

// 暂停上传
pauseUpload: function(uploadId) {
const upload = this.uploads.find(u => u.id === uploadId);
if (upload && upload.status === 'uploading') {
    upload.status = 'paused';
    
    // 取消当前正在进行的上传请求
    if (upload.xhr) {
        upload.xhr.abort();
    }
    if (upload.controller) {
        upload.controller.abort();
        // 创建新的控制器以便恢复上传
        upload.controller = new AbortController();
    }
    
    this.renderUploads();
}
},

// 恢复上传
resumeUpload: function(uploadId) {
const upload = this.uploads.find(u => u.id === uploadId);
if (upload && upload.status === 'paused') {
    this.startUpload(upload);
}
},

// 取消上传
cancelUpload: function(uploadId) {
const upload = this.uploads.find(u => u.id === uploadId);
if (upload && ['uploading', 'paused', 'pending'].includes(upload.status)) {
    upload.status = 'canceled';
    
    // 取消当前正在进行的上传请求
    if (upload.xhr) {
        upload.xhr.abort();
    }
    if (upload.controller) {
        upload.controller.abort();
    }
    
    this.renderUploads();
    
    // 3秒后从列表中移除已取消的上传
    setTimeout(() => {
        this.removeUpload(uploadId);
    }, 3000);
}
},

// 移除上传任务
removeUpload: function(uploadId) {
const index = this.uploads.findIndex(u => u.id === uploadId);
if (index !== -1) {
    this.uploads.splice(index, 1);
    this.renderUploads();
    this.updateUploadButton();
}
},

// 更新上传按钮状态
updateUploadButton: function() {
const button = document.getElementById('uploadProgressButton');
const counter = document.getElementById('uploadCounter');

const activeUploads = this.uploads.filter(u => u.status !== 'completed' && u.status !== 'canceled').length;

if (activeUploads > 0 || this.uploads.length > 0) {
    button.style.display = 'block';
    counter.textContent = activeUploads;
} else {
    button.style.display = 'none';
}
},

// 格式化文件大小
_formatFileSize: function(bytes) {
return formatFileSize(bytes);
},

// 格式化上传速度
_formatSpeed: function(bytesPerSecond) {
if (bytesPerSecond < 1024) {
    return `${bytesPerSecond.toFixed(1)} B/s`;
} else if (bytesPerSecond < 1024 * 1024) {
    return `${(bytesPerSecond / 1024).toFixed(1)} KB/s`;
} else {
    return `${(bytesPerSecond / (1024 * 1024)).toFixed(1)} MB/s`;
}
},

// 计算预计剩余时间
_calculateETA: function(upload) {
if (upload.speed <= 0) return '计算中...';

const remainingBytes = upload.fileSize - upload.uploadedBytes;
const remainingSeconds = Math.ceil(remainingBytes / upload.speed);

if (remainingSeconds < 60) {
    return `${remainingSeconds}秒`;
} else if (remainingSeconds < 3600) {
    return `${Math.floor(remainingSeconds / 60)}分${remainingSeconds % 60}秒`;
} else {
    const hours = Math.floor(remainingSeconds / 3600);
    const minutes = Math.floor((remainingSeconds % 3600) / 60);
    return `${hours}时${minutes}分`;
}
},

// 渲染上传列表
renderUploads: function() {
const container = document.getElementById('uploadProgressContainer');

if (this.uploads.length === 0) {
    container.innerHTML = '<p class="text-center text-muted">暂无上传任务</p>';
    return;
}

let html = '';
this.uploads.forEach(upload => {
    let progressClass, statusText, actionButtons;
    
    switch(upload.status) {
        case 'pending':
            progressClass = 'bg-info';
            statusText = '准备中...';
            actionButtons = `
                <button class="btn btn-sm btn-danger" onclick="uploadManager.cancelUpload('${upload.id}')">
                    <i class="fas fa-times"></i>
                </button>
            `;
            break;
        case 'uploading':
            progressClass = 'bg-primary progress-bar-striped progress-bar-animated';
            statusText = `${upload.progress}% - ${this._formatSpeed(upload.speed)} - 剩余: ${this._calculateETA(upload)}`;
            actionButtons = `
                <button class="btn btn-sm btn-warning me-1" onclick="uploadManager.pauseUpload('${upload.id}')">
                    <i class="fas fa-pause"></i>
                </button>
                <button class="btn btn-sm btn-danger" onclick="uploadManager.cancelUpload('${upload.id}')">
                    <i class="fas fa-times"></i>
                </button>
            `;
            break;
        case 'paused':
            progressClass = 'bg-warning';
            statusText = `已暂停 - ${upload.progress}%`;
            actionButtons = `
                <button class="btn btn-sm btn-success me-1" onclick="uploadManager.resumeUpload('${upload.id}')">
                    <i class="fas fa-play"></i>
                </button>
                <button class="btn btn-sm btn-danger" onclick="uploadManager.cancelUpload('${upload.id}')">
                    <i class="fas fa-times"></i>
                </button>
            `;
            break;
        case 'completed':
            progressClass = 'bg-success';
            statusText = '完成';
            actionButtons = `
                <button class="btn btn-sm btn-secondary" onclick="uploadManager.removeUpload('${upload.id}')">
                    <i class="fas fa-check"></i>
                </button>
            `;
            break;
        case 'error':
            progressClass = 'bg-danger';
            statusText = `失败: ${upload.error || '上传出错'}`;
            actionButtons = `
                <button class="btn btn-sm btn-secondary" onclick="uploadManager.removeUpload('${upload.id}')">
                    <i class="fas fa-times"></i>
                </button>
            `;
            break;
        case 'canceled':
            progressClass = 'bg-secondary';
            statusText = '已取消';
            actionButtons = `
                <button class="btn btn-sm btn-secondary" onclick="uploadManager.removeUpload('${upload.id}')">
                    <i class="fas fa-times"></i>
                </button>
            `;
            break;
    }
    
    html += `
        <div class="mb-3">
            <div class="d-flex justify-content-between mb-1">
                <div class="text-truncate" title="${upload.fileName}" style="max-width: 50%;">
                    <i class="${getFileIcon(get_media_type_from_filename(upload.fileName))} me-1"></i>
                    ${upload.fileName}
                </div>
                <div class="d-flex align-items-center">
                    <span class="me-2">${formatFileSize(upload.fileSize)}</span>
                    <span class="me-2">- ${statusText}</span>
                    <div>${actionButtons}</div>
                </div>
            </div>
            <div class="progress" style="height: 10px;">
                <div class="progress-bar ${progressClass}" role="progressbar" 
                     style="width: ${upload.progress}%" 
                     aria-valuenow="${upload.progress}" aria-valuemin="0" aria-valuemax="100"></div>
            </div>
            ${upload.enableChunking && upload.status === 'uploading' ? `
            <div class="mt-1">
                <small class="text-muted">分片 ${upload.currentChunkIndex + 1}/${upload.totalChunks}</small>
            </div>
            ` : ''}
        </div>
    `;
});

container.innerHTML = html;
}
};

// 从文件名判断媒体类型的辅助函数
function get_media_type_from_filename(filename) {
const extension = filename.split('.').pop().toLowerCase();
const videoExts = ['mp4', 'webm', 'ogv', 'mkv', 'avi', 'mov'];
const audioExts = ['mp3', 'wav', 'ogg', 'flac', 'aac'];
const imageExts = ['jpg', 'jpeg', 'png', 'gif', 'webp'];
const docExts = ['pdf', 'doc', 'docx', 'txt', 'md'];

if (videoExts.includes(extension)) return "video";
if (audioExts.includes(extension)) return "audio";
if (imageExts.includes(extension)) return "image";
if (docExts.includes(extension)) return "document";
return "other";
}

// 开始上传
function startUpload() {
    // 判断当前上传类型
    const isFileUpload = document.getElementById('uploadFile').checked;
    const fileInput = isFileUpload ? document.getElementById("fileInput") : document.getElementById("folderInput");
    const enableChunking = document.getElementById("enableChunking").checked;
    const checkLocalMapping = document.getElementById("checkLocalMapping").checked;

    if (!fileInput.files || fileInput.files.length === 0) {
        showNotification(`请选择要上传的${isFileUpload ? '文件' : '文件夹'}`, false);
        return;
    }

    const directory = currentPath;

    // 关闭模态框
    const modal = bootstrap.Modal.getInstance(document.getElementById("uploadModal"));
    modal.hide();

    // 如果是文件夹上传，创建一个Map来组织文件夹结构
    const folderStructure = new Map();
    if (!isFileUpload) {
        Array.from(fileInput.files).forEach(file => {
            // webkitRelativePath格式: "folder/subfolder/file.ext"
            const path = file.webkitRelativePath;
            const parts = path.split('/');
            
            // 只处理文件，忽略空目录
            if (parts.length > 1) {
                const mainFolder = parts[0]; // 顶级文件夹
                
                // 将文件按照顶级文件夹分组
                if (!folderStructure.has(mainFolder)) {
                    folderStructure.set(mainFolder, []);
                }
                folderStructure.get(mainFolder).push(file);
            }
        });
        
        // 显示将上传的文件夹
        if (folderStructure.size > 0) {
            const folders = Array.from(folderStructure.keys());
            console.log(`将上传 ${folders.length} 个文件夹: ${folders.join(', ')}`);
            showNotification(`开始上传 ${folders.length} 个文件夹，共 ${fileInput.files.length} 个文件`, true);
        }
    }

    // 为每个选中的文件创建上传任务
    if (isFileUpload) {
        // 单文件上传模式
        Array.from(fileInput.files).forEach(file => {
            const upload = uploadManager.addUpload(file, directory);
            upload.enableChunking = enableChunking && file.size > 5 * 1024 * 1024; // 5MB以上且启用分片
            upload.checkLocalMapping = checkLocalMapping; // 添加本地映射检查选项
            uploadManager.startUpload(upload);
        });
    } else {
        // 文件夹上传模式
        Array.from(fileInput.files).forEach(file => {
            // 从webkitRelativePath中获取文件的相对路径
            const relativePath = file.webkitRelativePath;
            const pathParts = relativePath.split('/');
            const fileName = pathParts.pop(); // 最后一部分是文件名
            
            // 构建目标路径：当前目录 + 相对路径中的文件夹部分
            let targetDir = directory;
            
            // 如果有文件夹部分，则添加到目标路径
            if (pathParts.length > 0) {
                targetDir = targetDir ? `${targetDir}/${pathParts.join('/')}` : pathParts.join('/');
            }
            
            const upload = uploadManager.addUpload(file, targetDir);
            upload.fileName = fileName; // 确保使用正确的文件名
            upload.enableChunking = enableChunking && file.size > 5 * 1024 * 1024;
            upload.checkLocalMapping = checkLocalMapping;
            
            // 添加创建文件夹的标记
            upload.createFolders = true;
            upload.relativePath = relativePath;
            
            uploadManager.startUpload(upload);
        });
    }

    // 显示上传进度按钮
    document.getElementById("uploadProgressButton").style.display = "block";

    // 清空文件输入框，以便下次选择相同文件
    fileInput.value = "";
}





// 添加映射源管理相关的 JavaScript 代码
let mappingSources = [];

// 初始化时添加映射源模态框事件
document.getElementById('mappingSourcesModal').addEventListener('show.bs.modal', function () {
loadMappingSources();
});

// 加载映射源列表
async function loadMappingSources() {
try {
    const response = await fetch('/api/mapping-sources');
    if (!response.ok) {
        throw new Error(`服务器返回错误 (${response.status})`);
    }
    
    const data = await response.json();
    mappingSources = data.sources || [];
    
    renderMappingSources();
} catch (error) {
    console.error('加载映射源失败:', error);
    showNotification(`加载映射源失败: ${error.message}`, false);
    
    document.getElementById('mappingSourcesList').innerHTML = `
        <div class="alert alert-danger">
            加载映射源失败: ${error.message}
        </div>
    `;
}
}

// 渲染映射源列表
function renderMappingSources() {
    const container = document.getElementById('mappingSourcesList');
    const countElement = document.getElementById('mappingSourcesCount');

    countElement.textContent = `当前共有 ${mappingSources.length} 个映射源`;

    if (mappingSources.length === 0) {
        container.innerHTML = `
            <div class="list-group-item text-center text-muted py-3">
                <i class="fas fa-folder-open me-2"></i>暂无映射源，请添加
            </div>
        `;
        return;
    }

    let html = '';
    mappingSources.forEach((source, index) => {
        console.log('映射源:', source);
        html += `
            <div class="list-group-item list-group-item-action d-flex justify-content-between align-items-center">
                <div class="text-break" style="max-width: 60%;">
                    <i class="fas fa-folder me-2"></i>${source}
                </div>
                <div>
                    <button class="btn btn-sm btn-primary me-1" data-source-path="${encodeURIComponent(source)}" onclick="mapAllContentsById(this)" title="导入该映射源下所有内容到当前目录">
                        <i class="fas fa-file-import"></i> 批量导入
                    </button>
                    <button class="btn btn-sm btn-danger" onclick="removeMappingSource(${index})">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
        `;
    });

    container.innerHTML = html;
}

// 映射源内容导入处理函数 - 通过元素ID获取路径
function mapAllContentsById(element) {
    const sourcePath = decodeURIComponent(element.getAttribute('data-source-path'));
    mapAllContents(sourcePath);
}

// 映射源下所有内容到当前目录
async function mapAllContents(sourcePath) {
    try {
        // 获取是否包含子文件夹的选项
        const includeSubfolders = document.getElementById('includeSubfolders').checked;
        
        // 添加确认对话框
        if (!confirm(`确定要将 "${sourcePath}" ${includeSubfolders ? '及其所有子文件夹' : '当前级别'} 下的内容映射到当前目录(${currentPath || '根目录'})吗？\n\n这可能会导入大量文件。`)) {
            return;
        }

        // 显示进度提示
        showNotification('正在批量映射文件，请稍候...', true);
        
        const response = await fetch('/api/map-all-contents', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                sourcePath: sourcePath,
                targetPath: currentPath || '',
                includeSubfolders: includeSubfolders
            })
        });
        
        if (!response.ok) {
            throw new Error(`服务器返回错误 (${response.status})`);
        }
        
        const result = await response.json();
        
        // 显示结果通知
        showNotification(`批量映射完成，成功导入 ${result.mappedCount || 0} 个文件`, true);
        
        // 刷新文件列表
        refreshFileList();
        
        // 关闭模态框
        const modal = bootstrap.Modal.getInstance(document.getElementById('mappingSourcesModal'));
        modal.hide();
    } catch (error) {
        console.error('批量映射文件失败:', error);
        showNotification(`批量映射失败: ${error.message}`, false);
    }
}
// 添加新的映射源
function addMappingSource() {
const input = document.getElementById('newMappingSource');
const value = input.value.trim();

if (!value) {
    showNotification('请输入有效的映射源路径', false);
    return;
}

// 检查是否已存在
if (mappingSources.includes(value)) {
    showNotification('该映射源已存在', false);
    return;
}

// 添加到列表
mappingSources.push(value);
renderMappingSources();

// 清空输入框
input.value = '';
}

// 移除映射源
function removeMappingSource(index) {
if (index >= 0 && index < mappingSources.length) {
    mappingSources.splice(index, 1);
    renderMappingSources();
}
}

// 保存映射源更改
async function saveMappingSources() {
try {
    const response = await fetch('/api/mapping-sources', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            sources: mappingSources
        })
    });
    
    if (!response.ok) {
        throw new Error(`服务器返回错误 (${response.status})`);
    }
    
    const result = await response.json();
    
    // 更新列表（使用服务器返回的有效路径）
    mappingSources = result.validSources || [];
    renderMappingSources();
    
    // 显示结果通知
    if (result.invalidSources && result.invalidSources.length > 0) {
        showNotification(`已保存有效的映射源，但有 ${result.invalidSources.length} 个无效路径被忽略`, false);
    } else {
        showNotification('映射源设置已保存', true);
    }
    
    // 关闭模态框
    const modal = bootstrap.Modal.getInstance(document.getElementById('mappingSourcesModal'));
    modal.hide();
} catch (error) {
    console.error('保存映射源失败:', error);
    showNotification(`保存映射源失败: ${error.message}`, false);
}
}

// 获取系统推荐路径
async function getSuggestedPaths() {
try {
    showNotification('正在获取系统推荐路径...', true);
    
    const response = await fetch('/api/system-default-paths');
    if (!response.ok) {
        throw new Error(`服务器返回错误 (${response.status})`);
    }
    
    const result = await response.json();
    let addedCount = 0;
    
    if (result.defaultPaths && result.defaultPaths.length > 0) {
        // 添加不重复的路径
        result.defaultPaths.forEach(path => {
            if (!mappingSources.includes(path)) {
                mappingSources.push(path);
                addedCount++;
            }
        });
        
        renderMappingSources();
        showNotification(`已添加 ${addedCount} 个系统推荐路径`, true);
    } else {
        showNotification('未找到可添加的系统推荐路径', false);
    }
} catch (error) {
    console.error('获取系统推荐路径失败:', error);
    showNotification(`获取系统推荐路径失败: ${error.message}`, false);
}
}

/**
 * 根据文件类型获取对应的图标
 * @param {string} fileName - 文件名
 * @param {string} type - 类型 (file/folder)
 * @param {string} mediaType - 媒体类型
 * @returns {string} 图标HTML
 */
function getFileIcon(fileName, type, mediaType) {
    if (type === 'folder') {
      return '<i class="fas fa-folder file-icon file-icon-folder zay-icon-md"></i>';
    }
    
    const ext = fileName.split('.').pop().toLowerCase();
    
    // 视频文件
    if (mediaType === 'video' || ['mp4', 'webm', 'mov', 'avi', 'mkv'].includes(ext)) {
      return '<i class="fas fa-film file-icon file-icon-video zay-icon-md"></i>';
    }
    
    // 音频文件
    if (mediaType === 'audio' || ['mp3', 'wav', 'ogg', 'flac'].includes(ext)) {
      return '<i class="fas fa-music file-icon file-icon-audio zay-icon-md"></i>';
    }
    
    // 图像文件
    if (mediaType === 'image' || ['jpg', 'jpeg', 'png', 'gif', 'svg', 'webp'].includes(ext)) {
      return '<i class="fas fa-image file-icon file-icon-image zay-icon-md"></i>';
    }
    
    // 文档文件
    if (['pdf', 'doc', 'docx', 'txt', 'rtf', 'xls', 'xlsx', 'ppt', 'pptx'].includes(ext)) {
      return '<i class="fas fa-file-alt file-icon file-icon-document zay-icon-md"></i>';
    }
    
    // 压缩文件
    if (['zip', 'rar', '7z', 'tar', 'gz'].includes(ext)) {
      return '<i class="fas fa-file-archive file-icon file-icon-archive zay-icon-md"></i>';
    }
    
    // 代码文件
    if (['html', 'css', 'js', 'py', 'java', 'php', 'json', 'xml'].includes(ext)) {
      return '<i class="fas fa-file-code file-icon file-icon-code zay-icon-md"></i>';
    }
    
    // 其他文件
    return '<i class="fas fa-file file-icon zay-icon-md"></i>';
  }
  



