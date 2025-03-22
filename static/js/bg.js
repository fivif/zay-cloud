
    // 背景图片管理功能
    let userBackgrounds = [];
    let currentBackground = null;

    // 初始化页面时加载背景
    document.addEventListener("DOMContentLoaded", () => {
        // 保留原有初始化代码
        refreshFileList();
        switchCategory("all");
        
        // 加载背景图片设置
        loadBackgroundSettings();
    });

   // 加载背景设置
async function loadBackgroundSettings() {
    try {
        // 从服务器加载背景设置
        const response = await fetch('/api/background');
        if (!response.ok) throw new Error('获取背景设置失败');
        
        const data = await response.json();
        
        // 设置当前背景
        if (data.current_background) {
            setBackground(data.current_background);
        }
        
        // 加载用户背景列表
        userBackgrounds = data.user_backgrounds || [];
        renderUserBackgrounds();
    } catch (error) {
        console.error('加载背景设置失败:', error);
    }
}

   // 设置背景图片
async function setBackground(url) {
    document.body.style.backgroundImage = `url('${url}')`;
    currentBackground = url;
    
    try {
        // 保存到服务器
        const formData = new FormData();
        formData.append('background_url', url);
        
        await fetch('/api/background/set', {
            method: 'POST',
            body: formData
        });
    } catch (error) {
        console.error('保存背景设置失败:', error);
    }
    
    // 更新UI，标记当前选中的背景
    const thumbnails = document.querySelectorAll('.bg-thumbnail');
    thumbnails.forEach(thumb => {
        if (thumb.getAttribute('onclick').includes(url)) {
            thumb.classList.add('active');
        } else {
            thumb.classList.remove('active');
        }
    });
}

   // 移除背景
async function resetBackground() {
    document.body.style.backgroundImage = 'none';
    currentBackground = null;
    
    try {
        await fetch('/api/background/reset', {
            method: 'POST'
        });
    } catch (error) {
        console.error('重置背景失败:', error);
    }
    
    // 更新UI，移除所有选中状态
    const thumbnails = document.querySelectorAll('.bg-thumbnail');
    thumbnails.forEach(thumb => thumb.classList.remove('active'));
}

    // 上传背景图片
async function uploadBackgroundImage() {
    const input = document.getElementById('bgImageInput');
    if (!input.files || input.files.length === 0) {
        showNotification("请选择要上传的图片", false);
        return;
    }
    
    const file = input.files[0];
    // 检查文件类型
    if (!file.type.startsWith('image/')) {
        showNotification("请选择有效的图片文件", false);
        return;
    }
    
    try {
        showNotification("正在上传背景图片...", true);
        
        // 创建FormData对象
        const formData = new FormData();
        formData.append('background', file);
        formData.append('name', file.name);
        
        // 发送上传请求
        const response = await fetch('/api/background/upload', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '上传失败');
        }
        
        const result = await response.json();
        
        // 添加到用户背景列表
        userBackgrounds.push({
            url: result.background.url,
            name: result.background.name,
            id: result.background.id
        });
        
        // 设置为当前背景
        setBackground(result.background.url);
        
        // 更新UI
        renderUserBackgrounds();
        
        showNotification("背景图片上传成功", true);
        
        // 清空输入
        input.value = '';
    } catch (error) {
        console.error('上传背景图片失败:', error);
        showNotification(`上传背景图片失败: ${error.message}`, false);
    }
}


function renderUserBackgrounds() {
    const container = document.getElementById('userBackgrounds');
    
    if (!container) {
        console.error('Error: Element with id "userBackgrounds" not found');
        return;
    }
    
    if (userBackgrounds.length === 0) {
        container.innerHTML = '<p>暂无自定义背景</p>'; // 直接在容器中显示提示
        return;
    }
    
    let html = '';
    userBackgrounds.forEach((bg, index) => {
        html += `
            <div class="col-md-4 mb-3">
                <img src="${bg.url}" class="bg-thumbnail ${bg.url === currentBackground ? 'active' : ''}" 
                     onclick="setBackground('${bg.url}')" alt="${bg.name || '自定义背景'}">
                <div class="d-flex justify-content-between align-items-center mt-2">
                    <small class="text-truncate" style="max-width: 70%;">${bg.name || '自定义背景'}</small>
                    <button class="btn btn-sm btn-danger" onclick="deleteUserBackground('${bg.id}')">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
        `;
    });
    
    container.innerHTML = html;
}



    // 删除用户背景
    async function deleteUserBackground(id) {
        console.log(id);
    try {
        const response = await fetch(`/api/background/${id}`, {
            method: 'DELETE'
        }
        
    );
    console.log(id);
    console.log(response);
        
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || '删除失败');
        }
        
        // 从列表中移除
        const index = userBackgrounds.findIndex(bg => bg.id === id);
       
        if (index >= 0) {
            const bg = userBackgrounds[index];
            userBackgrounds.splice(index, 1);
            
            // 如果删除的是当前使用的背景，则切换到另一个背景（如果有）
            if (bg.url === currentBackground) {
                if (userBackgrounds.length > 0) {
                    // 尝试使用下一个背景，如果没有下一个就使用第一个
                    const nextIndex = index < userBackgrounds.length ? index : 0;
                    setBackground(userBackgrounds[nextIndex].url);
                    showNotification("背景已删除，已自动切换到下一个可用背景", true);
                } else {
                    // 没有其他背景可用，重置背景
                    resetBackground();
                    showNotification("背景已删除，没有其他可用背景", true);
                }
            } else {
                showNotification("背景图片已删除", true);
            }
            
            // 确保渲染更新
            renderUserBackgrounds();
        }
    } catch (error) {
        console.error('删除背景图片失败:', error);
        showNotification(`删除背景图片失败: ${error.message}`, false);
    }
}

function adjustStylesForBackground(url) {
    if (!url) {
        // 无背景时恢复默认样式
        document.documentElement.classList.remove('dark-background');
        document.documentElement.classList.remove('light-background');
        return;
    }
    
    // 创建一个临时图像元素来分析背景
    const img = new Image();
    img.crossOrigin = "Anonymous";
    img.onload = function() {
        // 创建Canvas来分析图像
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d');
        canvas.width = this.width;
        canvas.height = this.height;
        ctx.drawImage(this, 0, 0);
        
        // 获取中心区域的图像数据来分析亮度
        const imageData = ctx.getImageData(
            Math.floor(this.width / 4), 
            Math.floor(this.height / 4), 
            Math.floor(this.width / 2), 
            Math.floor(this.height / 2)
        );
        
        // 计算平均亮度
        let totalBrightness = 0;
        for (let i = 0; i < imageData.data.length; i += 4) {
            const r = imageData.data[i];
            const g = imageData.data[i + 1];
            const b = imageData.data[i + 2];
            
            // 加权亮度计算方式
            const brightness = (r * 0.299 + g * 0.587 + b * 0.114) / 255;
            totalBrightness += brightness;
        }
        
        const avgBrightness = totalBrightness / (imageData.data.length / 4);
        console.log('背景平均亮度:', avgBrightness);
        
        // 根据亮度调整样式
        if (avgBrightness < 0.5) {
            // 深色背景
            document.documentElement.classList.add('dark-background');
            document.documentElement.classList.remove('light-background');
        } else {
            // 浅色背景
            document.documentElement.classList.add('light-background');
            document.documentElement.classList.remove('dark-background');
        }
    };
    
    img.onerror = function() {
        console.error('背景图片加载失败，无法分析亮度');
    };
    
    img.src = url;
}

// 修改 setBackground 函数，增加亮度适应
function setBackground(url) {
    document.body.style.backgroundImage = `url('${url}')`;
    currentBackground = url;
    
    // 调用样式适应函数
    adjustStylesForBackground(url);
    
    try {
        // 保存到服务器
        const formData = new FormData();
        formData.append('background_url', url);
        
        fetch('/api/background/set', {
            method: 'POST',
            body: formData
        });
    } catch (error) {
        console.error('保存背景设置失败:', error);
    }
    
    // 更新UI，标记当前选中的背景
    const thumbnails = document.querySelectorAll('.bg-thumbnail');
    thumbnails.forEach(thumb => {
        if (thumb.getAttribute('onclick').includes(url)) {
            thumb.classList.add('active');
        } else {
            thumb.classList.remove('active');
        }
    });
}
// 在文件末尾添加以下代码

// 设置键盘方向键导航功能
function setupBackgroundKeyNavigation() {
    document.addEventListener('keydown', (event) => {
        // 如果当前在输入框中，不处理键盘事件
        if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') {
            return;
        }
        
        // 如果没有背景图片，不处理
        if (userBackgrounds.length <= 1) {
            return;
        }
        
        // 左方向键 - 切换到上一个背景
        if (event.key === 'ArrowLeft') {
            switchToPreviousBackground();
            event.preventDefault(); // 防止页面滚动
        }
        // 右方向键 - 切换到下一个背景
        else if (event.key === 'ArrowRight') {
            switchToNextBackground();
            event.preventDefault(); // 防止页面滚动
        }
        // R键 - 重置背景
        else if (event.key === 'r' || event.key === 'R') {
            resetBackground();
        }
        // 上方向键 - 切换上一个背景
        else if (event.key === 'ArrowUp') {
            switchToPreviousBackground();
            event.preventDefault(); // 防止页面滚动
        }
        // 下方向键 - 切换下一个背景
        else if (event.key === 'ArrowDown') {
            switchToNextBackground();
            event.preventDefault(); // 防止页面滚动
        }
    });
}

// 切换到下一个背景
function switchToNextBackground() {
    if (userBackgrounds.length <= 1) return;
    
    // 找到当前背景的索引
    const currentIndex = userBackgrounds.findIndex(bg => bg.url === currentBackground);
    
    // 计算下一个背景的索引（如果当前是最后一个，则循环到第一个）
    const nextIndex = (currentIndex + 1) % userBackgrounds.length;
    
    // 设置新背景
    setBackground(userBackgrounds[nextIndex].url);
    
    // 显示提示信息
    // showNotification(`背景切换: ${nextIndex + 1}/${userBackgrounds.length}`, true);
}

// 切换到上一个背景
function switchToPreviousBackground() {
    if (userBackgrounds.length <= 1) return;
    
    // 找到当前背景的索引
    const currentIndex = userBackgrounds.findIndex(bg => bg.url === currentBackground);
    
    // 计算上一个背景的索引（如果当前是第一个，则循环到最后一个）
    const prevIndex = (currentIndex - 1 + userBackgrounds.length) % userBackgrounds.length;
    
    // 设置新背景
    setBackground(userBackgrounds[prevIndex].url);
    
    // 显示提示信息
    // showNotification(`背景切换: ${prevIndex + 1}/${userBackgrounds.length}`, true);
}

// 修改DOMContentLoaded事件，添加键盘导航初始化
document.addEventListener("DOMContentLoaded", () => {
    // 保留原有初始化代码
    refreshFileList();
    switchCategory("all");
    
    // 加载背景图片设置
    loadBackgroundSettings();
    
    // 设置键盘导航
    setupBackgroundKeyNavigation();
});