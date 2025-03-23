// WebDAV 设置管理
document.addEventListener('DOMContentLoaded', function() {
    // 获取元素
    const webdavEnabled = document.getElementById('webdavEnabled');
    const webdavSettingsContainer = document.getElementById('webdavSettingsContainer');
    const webdavAuthEnabled = document.getElementById('webdavAuthEnabled');
    const webdavAuthSettings = document.getElementById('webdavAuthSettings');
    const webdavPort = document.getElementById('webdavPort');
    const webdavUsername = document.getElementById('webdavUsername');
    const webdavPassword = document.getElementById('webdavPassword');
    const webdavStatusIcon = document.getElementById('webdavStatusIcon');
    const webdavStatusText = document.getElementById('webdavStatusText');
    const webdavUrlContainer = document.getElementById('webdavUrlContainer');
    const webdavUrl = document.getElementById('webdavUrl');
    const copyWebdavUrlBtn = document.getElementById('copyWebdavUrlBtn');
    const saveWebdavConfigBtn = document.getElementById('saveWebdavConfigBtn');
    
    // 如果找不到 WebDAV 相关元素，直接返回
    if (!webdavEnabled || !saveWebdavConfigBtn) return;
    
    // 切换 WebDAV 启用状态
    webdavEnabled.addEventListener('change', function() {
        webdavSettingsContainer.style.display = this.checked ? 'block' : 'none';
    });
    
    // 切换认证设置
    webdavAuthEnabled.addEventListener('change', function() {
        webdavAuthSettings.style.display = this.checked ? 'block' : 'none';
    });
    
    // 复制 WebDAV URL
    copyWebdavUrlBtn.addEventListener('click', function() {
        webdavUrl.select();
        document.execCommand('copy');
        showNotification('WebDAV URL 已复制到剪贴板', true);
    });
    
    // 加载 WebDAV 配置
    function loadWebDAVConfig() {
        // webdavStatusIcon.innerHTML = '<i class="fas fa-circle-notch fa-spin text-primary"></i>';
        webdavStatusText.textContent = '保存后再检查状态';
        webdavUrlContainer.style.display = 'none';
        
        fetch('/api/webdav/config')
            .then(response => response.json())
            .then(config => {
                // 填充表单
                webdavEnabled.checked = config.enabled;
                webdavSettingsContainer.style.display = config.enabled ? 'block' : 'none';
                
                webdavPort.value = config.port;
                webdavAuthEnabled.checked = config.auth_enabled;
                webdavAuthSettings.style.display = config.auth_enabled ? 'block' : 'none';
                
                webdavUsername.value = config.username;
                // 不设置密码，因为API返回的是掩码
                
                // 更新状态
                if (config.status === 'running') {
                    webdavStatusIcon.innerHTML = '<i class="fas fa-check-circle text-success"></i>';
                    webdavStatusText.textContent = 'WebDAV 服务正在运行';
                    
                    // 显示URL
                    if (config.url) {
                        webdavUrlContainer.style.display = 'block';
                        webdavUrl.value = config.url;
                    }
                } else {
                    webdavStatusIcon.innerHTML = '<i class="fas fa-times-circle text-danger"></i>';
                    webdavStatusText.textContent = 'WebDAV 服务未运行';
                }
            })
            .catch(error => {
                console.error('加载 WebDAV 配置失败:', error);
                webdavStatusIcon.innerHTML = '<i class="fas fa-exclamation-triangle text-warning"></i>';
                webdavStatusText.textContent = '获取状态失败';
            });
    }
    
    // 保存 WebDAV 配置
    saveWebdavConfigBtn.addEventListener('click', function() {
        // 显示保存中状态
        this.disabled = true;
        this.innerHTML = '<i class="fas fa-circle-notch fa-spin"></i> 保存中...';
        
        // 准备配置数据
        const config = {
            enabled: webdavEnabled.checked,
            port: parseInt(webdavPort.value),
            auth_enabled: webdavAuthEnabled.checked,
            username: webdavUsername.value,
            password: webdavPassword.value
        };
        
        // 发送请求
        fetch('/api/webdav/config', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        })
        .then(response => response.json())
        .then(data => {
            showNotification(data.message, true);
            
            // 重置按钮状态
            this.disabled = false;
            this.innerHTML = '保存设置';
            
            // 重新加载配置以更新状态
            loadWebDAVConfig();
        })
        .catch(error => {
            console.error('保存 WebDAV 配置失败:', error);
            showNotification('保存 WebDAV 配置失败: ' + error, false);
            
            // 重置按钮状态
            this.disabled = false;
            this.innerHTML = '保存设置';
        });
    });
    
    // 当模态窗口显示时加载配置
    $('#webdavConfigModal').on('shown.bs.modal', function () {
        loadWebDAVConfig();
    });
});