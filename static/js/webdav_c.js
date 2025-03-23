
// 全局状态管理
const webdavClientState = {
    isLoading: false,
    currentConnection: null,
    connections: []
};
// WebDAV 客户端管理
document.addEventListener('DOMContentLoaded', function() {
    // 获取 WebDAV 连接管理模态窗口
    const webdavClientModal = document.getElementById('webdavClientModal');
    if (!webdavClientModal) return;
    
    // 连接列表
    const webdavConnectionsList = document.getElementById('webdavConnectionsList');
    // 连接表单
    const webdavConnectionForm = document.getElementById('webdavConnectionForm');
    const connectionName = document.getElementById('connectionName');
    const connectionUrl = document.getElementById('connectionUrl');
    const connectionUsername = document.getElementById('connectionUsername');
    const connectionPassword = document.getElementById('connectionPassword');
    const connectionFolder = document.getElementById('connectionFolder');
    const connectionEnabled = document.getElementById('connectionEnabled');
    const testConnectionBtn = document.getElementById('testConnectionBtn');
    const saveConnectionBtn = document.getElementById('saveConnectionBtn');
    const cancelEditBtn = document.getElementById('cancelEditBtn');
    
    // 当前编辑的连接名称
    let currentEditingConnection = '';
    let isEditing = false;
    
    // 加载重试变量
    let connectionLoadRetryCount = 0;
    const MAX_RETRY_ATTEMPTS = 3;
    const RETRY_DELAY = 5000; // 5秒重试间隔
    let retryTimeoutId = null;
    


        // 添加模态窗口事件监听器 - 确保在模态窗口打开时加载连接列表
        webdavClientModal.addEventListener('show.bs.modal', function() {
            console.log('WebDAV客户端模态窗口正在打开，即将加载连接列表...');
            // 重置状态并加载连接列表
            connectionLoadRetryCount = 0;
            if (retryTimeoutId) {
                clearTimeout(retryTimeoutId);
                retryTimeoutId = null;
            }
            loadConnections();
        });

     // 加载连接列表
     function loadConnections(isRetry = false) {
        // 取消任何现有的重试计时器
        if (retryTimeoutId) {
            clearTimeout(retryTimeoutId);
            retryTimeoutId = null;
        }
        
        // 添加明确的加载指示器
        if (!isRetry) {
            // 首次加载时重置重试计数
            connectionLoadRetryCount = 0;
            webdavConnectionsList.innerHTML = `
                <div class="text-center p-4">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">加载中...</span>
                    </div>
                    <p class="mt-2">加载WebDAV连接中...</p>
                </div>
            `;
        } else {
            // 重试时显示重试信息
            webdavConnectionsList.innerHTML = `
                <div class="text-center p-4">
                    <div class="spinner-border text-primary" role="status">
                        <span class="visually-hidden">加载中...</span>
                    </div>
                    <p class="mt-2">正在重试加载WebDAV连接 (${connectionLoadRetryCount}/${MAX_RETRY_ATTEMPTS})...</p>
                </div>
            `;
        }
        console.log('开始加载WebDAV连接列表' + (isRetry ? ` (重试 ${connectionLoadRetryCount})` : '') + '...');
            // 添加调试代码，确认请求正在发送
            const startTime = new Date().getTime();
            console.log(`正在发送请求到 /api/webdav-client/connections，时间: ${startTime}`);
        
        fetch('/api/webdav-client/connections')
            .then(response => {
                console.log('收到API响应, 状态码:', response.status);
                if (!response.ok) {
                    throw new Error(`HTTP错误: ${response.status}`);
                }
                return response.json();
            })
            .then(connections => {
                console.log('解析到连接数据:', connections);
                webdavConnectionsList.innerHTML = '';
                
                if (!connections || !Array.isArray(connections)) {
                    throw new Error('返回的数据格式不正确');
                }
                
                if (connections.length === 0) {
                    webdavConnectionsList.innerHTML = `
                        <div class="text-center p-4 text-muted">
                            <i class="fas fa-info-circle mb-2"></i>
                            <p>暂无 WebDAV 连接</p>
                            <button class="btn btn-sm btn-primary add-connection-btn">
                                <i class="fas fa-plus"></i> 添加连接
                            </button>
                        </div>
                    `;
                    
                    document.querySelector('.add-connection-btn').addEventListener('click', showAddConnectionForm);
                    return;
                }
                
      
      // 创建连接列表
      connections.forEach(conn => {
        const card = document.createElement('div');
        card.className = 'card mb-3 webdav-connection-card';
        card.dataset.connectionName = conn.name;
        
        // 添加连接卡片内容
        card.innerHTML = `
            <div class="card-header d-flex justify-content-between align-items-center ${conn.enabled ? 'bg-light' : 'bg-light-subtle'}">
                <h6 class="mb-0">
                    <i class="fas fa-${conn.enabled ? 'link' : 'unlink'} me-2 ${conn.enabled ? 'text-success' : 'text-muted'}"></i>
                    ${conn.name}
                </h6>
                <div class="btn-group btn-group-sm">
                    <button class="btn btn-outline-primary browse-btn" title="浏览WebDAV内容">
                        <i class="fas fa-folder-open"></i>
                    </button>
                    <button class="btn btn-outline-secondary edit-btn" title="编辑连接">
                        <i class="fas fa-edit"></i>
                    </button>
                    <button class="btn btn-outline-danger delete-btn" title="删除连接">
                        <i class="fas fa-trash"></i>
                    </button>
                </div>
            </div>
            <div class="card-body">
                <div class="small mb-2 d-flex align-items-center">
                    <i class="fas fa-link me-2 text-muted"></i> 
                    <span class="text-truncate" style="max-width: 85%;">
                        ${conn.url}
                    </span>
                </div>
                <div class="small text-muted d-flex flex-wrap">
                    <span class="me-3">
                        <i class="fas fa-user me-1"></i> ${conn.username || '匿名'}
                    </span>
                    ${conn.folder && conn.folder !== '/' ? 
                        `<span><i class="fas fa-folder me-1"></i> ${conn.folder}</span>` : 
                        '<span><i class="fas fa-folder me-1"></i> 根目录</span>'}
                </div>
            </div>
            <div class="card-footer bg-transparent">
                <span class="badge ${conn.enabled ? 'text-bg-success' : 'text-bg-secondary'}">
                    ${conn.enabled ? '已启用' : '已禁用'}
                </span>
                <span class="small text-muted ms-2">最后同步: 从未</span>
            </div>
        `;
        
        webdavConnectionsList.appendChild(card);
        
        // 添加事件监听器
        card.querySelector('.browse-btn').addEventListener('click', () => {
            browseWebDAV(conn.name);
        });
        
        card.querySelector('.edit-btn').addEventListener('click', () => {
            editConnection(conn.name);
        });
        
        card.querySelector('.delete-btn').addEventListener('click', () => {
            deleteConnection(conn.name);
        });
    });
    
    console.log('连接列表渲染完成');
})
            .catch(error => {
                console.error('加载 WebDAV 连接失败:', error);
                webdavConnectionsList.innerHTML = `
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-triangle me-2"></i>
                        加载WebDAV连接失败: ${error.message || error}
                    </div>
                    <div class="text-center mt-3">
                        <button class="btn btn-sm btn-primary add-connection-btn">
                            <i class="fas fa-plus"></i> 添加连接
                        </button>
                    </div>
                `;
                
                document.querySelector('.add-connection-btn')?.addEventListener('click', showAddConnectionForm);
                showNotification('加载 WebDAV 连接失败: ' + error, false);
            });
    }
    
    // 显示添加连接表单
    function showAddConnectionForm() {
        // 重置表单
        webdavConnectionForm.reset();
        currentEditingConnection = '';
        isEditing = false;
        
        // 显示表单区域
        document.getElementById('webdavConnectionFormContainer').style.display = 'block';
        document.getElementById('webdavConnectionListContainer').style.display = 'none';
        
        // 更新标题
        document.getElementById('connectionFormTitle').textContent = '添加 WebDAV 连接';
        
        // 修改保存按钮
        saveConnectionBtn.textContent = '添加连接';
    }
    
    // 编辑连接
    function editConnection(name) {
        currentEditingConnection = name;
        isEditing = true;
        
        // 获取连接详情
        fetch('/api/webdav-client/connections')
            .then(response => response.json())
            .then(connections => {
                const connection = connections.find(c => c.name === name);
                if (!connection) {
                    showNotification(`未找到连接: ${name}`, false);
                    return;
                }
                
                // 填充表单
                connectionName.value = connection.name;
                connectionUrl.value = connection.url;
                connectionUsername.value = connection.username;
                connectionPassword.value = ''; // 出于安全考虑不回显密码
                connectionFolder.value = connection.folder;
                connectionEnabled.checked = connection.enabled;
                
                // 显示表单区域
                document.getElementById('webdavConnectionFormContainer').style.display = 'block';
                document.getElementById('webdavConnectionListContainer').style.display = 'none';
                
                // 更新标题
                document.getElementById('connectionFormTitle').textContent = `编辑连接: ${name}`;
                
                // 修改保存按钮
                saveConnectionBtn.textContent = '更新连接';
            })
            .catch(error => {
                console.error('获取连接详情失败:', error);
                showNotification('获取连接详情失败: ' + error, false);
            });
    }
    
    // 删除连接
    function deleteConnection(name) {
        if (!confirm(`确定要删除连接 "${name}" 吗？`)) {
            return;
        }
        
        fetch(`/api/webdav-client/connections/${name}`, {
            method: 'DELETE'
        })
        .then(response => response.json())
        .then(data => {
            showNotification(data.message, data.success);
            if (data.success) {
                loadConnections();
            }
        })
        .catch(error => {
            console.error('删除连接失败:', error);
            showNotification('删除连接失败: ' + error, false);
        });
    }
    
    // 浏览 WebDAV 内容
    function browseWebDAV(connectionName, path = '/') {
        // 创建或显示 WebDAV 浏览模态窗口
        let browseModal = document.getElementById('webdavBrowseModal');
        
        if (!browseModal) {
            // 创建模态窗口
            browseModal = document.createElement('div');
            browseModal.className = 'modal fade';
            browseModal.id = 'webdavBrowseModal';
            browseModal.setAttribute('tabindex', '-1');
            browseModal.setAttribute('aria-labelledby', 'webdavBrowseModalLabel');
            browseModal.setAttribute('aria-hidden', 'true');
            
            browseModal.innerHTML = `
                <div class="modal-dialog modal-xl">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title" id="webdavBrowseModalLabel">
                                <i class="fas fa-folder-open me-2"></i>浏览 WebDAV
                            </h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body">
                            <div class="d-flex justify-content-between mb-3">
                                <div class="btn-group me-2">
                                    <button id="webdavBackBtn" class="btn btn-sm btn-outline-secondary">
                                        <i class="fas fa-arrow-left"></i> 返回
                                    </button>
                                    <button id="webdavHomeBtn" class="btn btn-sm btn-outline-secondary">
                                        <i class="fas fa-home"></i> 根目录
                                    </button>
                                    <button id="webdavRefreshBtn" class="btn btn-sm btn-outline-secondary">
                                        <i class="fas fa-sync"></i> 刷新
                                    </button>
                                </div>
                                <div class="btn-group">
                                    <button id="webdavCreateFolderBtn" class="btn btn-sm btn-outline-primary">
                                        <i class="fas fa-folder-plus"></i> 新建文件夹
                                    </button>
                                    <button id="webdavDownloadBtn" class="btn btn-sm btn-outline-success">
                                        <i class="fas fa-download"></i> 下载到存储
                                    </button>
                                </div>
                            </div>
                            
                            <div id="webdavPathBreadcrumb" class="mb-3"></div>
                            
                            <div id="webdavFilesList" class="list-group"></div>
                            
                            <div id="webdavLoadingSpinner" class="text-center p-5" style="display: none;">
                                <div class="spinner-border text-primary" role="status">
                                    <span class="visually-hidden">加载中...</span>
                                </div>
                                <p class="mt-2">加载中...</p>
                            </div>
                            
                            <div id="webdavEmptyMessage" class="text-center p-5" style="display: none;">
                                <i class="fas fa-folder-open fa-3x text-muted mb-3"></i>
                                <p>此文件夹为空</p>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
            document.body.appendChild(browseModal);
            
            // 初始化Bootstrap模态窗口
            new bootstrap.Modal(browseModal);
        }
        
        // 保存当前状态
        browseModal.dataset.connectionName = connectionName;
        browseModal.dataset.currentPath = path;
        
        // 显示模态窗口
        const modal = bootstrap.Modal.getInstance(browseModal) || new bootstrap.Modal(browseModal);
        modal.show();
        
        // 加载WebDAV内容
        loadWebDAVContents(connectionName, path);
        
        // 为按钮添加事件监听器
        const backBtn = document.getElementById('webdavBackBtn');
        const homeBtn = document.getElementById('webdavHomeBtn');
        const refreshBtn = document.getElementById('webdavRefreshBtn');
        const createFolderBtn = document.getElementById('webdavCreateFolderBtn');
        const downloadBtn = document.getElementById('webdavDownloadBtn');
        
        backBtn.onclick = () => {
            const currentPath = browseModal.dataset.currentPath;
            if (currentPath === '/' || currentPath === '') {
                return;
            }
            
            // 获取上一级路径
            const pathParts = currentPath.split('/').filter(p => p !== '');
            pathParts.pop();
            const parentPath = '/' + pathParts.join('/');
            
            browseModal.dataset.currentPath = parentPath;
            loadWebDAVContents(connectionName, parentPath);
        };
        
        homeBtn.onclick = () => {
            browseModal.dataset.currentPath = '/';
            loadWebDAVContents(connectionName, '/');
        };
        
        refreshBtn.onclick = () => {
            loadWebDAVContents(connectionName, browseModal.dataset.currentPath);
        };
        
        createFolderBtn.onclick = () => {
            const folderName = prompt('请输入新文件夹名称:');
            if (!folderName) return;
            
            const currentPath = browseModal.dataset.currentPath;
            const newPath = currentPath === '/' ? 
                '/' + folderName : 
                currentPath + '/' + folderName;
            
            // 创建文件夹
            const formData = new FormData();
            formData.append('path', newPath);
            
            fetch(`/api/webdav-client/${connectionName}/mkdir`, {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                showNotification(data.message, data.success);
                if (data.success) {
                    loadWebDAVContents(connectionName, currentPath);
                }
            })
            .catch(error => {
                console.error('创建文件夹失败:', error);
                showNotification('创建文件夹失败: ' + error, false);
            });
        };
        
        downloadBtn.onclick = () => {
            // 打开目标文件夹选择对话框
            const currentPath = getCurrentPath();
            if (!currentPath) return;
            
            showFolderSelector('选择下载目标文件夹', '/', selected => {
                if (!selected) return;
                
                // 下载选中文件
                const selectedItems = getSelectedWebDAVItems();
                if (selectedItems.length === 0) {
                    showNotification('请先选择要下载的文件', false);
                    return;
                }
                
                downloadSelectedItems(connectionName, selectedItems, selected);
            });
        };
    }
    
    // 加载WebDAV内容
    function loadWebDAVContents(connectionName, path) {
        const filesList = document.getElementById('webdavFilesList');
        const pathBreadcrumb = document.getElementById('webdavPathBreadcrumb');
        const loadingSpinner = document.getElementById('webdavLoadingSpinner');
        const emptyMessage = document.getElementById('webdavEmptyMessage');
        
        // 显示加载中
        filesList.innerHTML = '';
        loadingSpinner.style.display = 'block';
        emptyMessage.style.display = 'none';
        
        // 更新面包屑
        updatePathBreadcrumb(connectionName, path);
        
        fetch(`/api/webdav-client/${connectionName}/list?path=${encodeURIComponent(path)}`)
            .then(response => response.json())
            .then(files => {
                loadingSpinner.style.display = 'none';
                filesList.innerHTML = '';
                
                if (files.length === 0) {
                    emptyMessage.style.display = 'block';
                    return;
                }
                
                // 排序：文件夹在前，文件在后
                files.sort((a, b) => {
                    if (a.is_dir && !b.is_dir) return -1;
                    if (!a.is_dir && b.is_dir) return 1;
                    return a.name.localeCompare(b.name);
                });
                
                // 创建文件列表
                files.forEach(file => {
                    const item = document.createElement('div');
                    item.className = 'list-group-item list-group-item-action d-flex justify-content-between align-items-center';
                    
                    // 文件大小格式化
                    const fileSize = file.is_dir ? '' : formatFileSize(file.size);
                    
                    // 图标
                    let icon;
                    if (file.is_dir) {
                        icon = '<i class="fas fa-folder text-warning"></i>';
                    } else {
                        const ext = file.name.split('.').pop().toLowerCase();
                        if (['jpg', 'jpeg', 'png', 'gif', 'webp'].includes(ext)) {
                            icon = '<i class="fas fa-image text-info"></i>';
                        } else if (['mp4', 'webm', 'avi', 'mov'].includes(ext)) {
                            icon = '<i class="fas fa-video text-danger"></i>';
                        } else if (['mp3', 'wav', 'ogg', 'flac'].includes(ext)) {
                            icon = '<i class="fas fa-music text-success"></i>';
                        } else if (['pdf', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx'].includes(ext)) {
                            icon = '<i class="fas fa-file-alt text-primary"></i>';
                        } else {
                            icon = '<i class="fas fa-file text-secondary"></i>';
                        }
                    }
                    
                    item.innerHTML = `
                        <div class="form-check d-flex align-items-center">
                            <input class="form-check-input me-2" type="checkbox" value="" id="check_${file.name}">
                            <label class="form-check-label" for="check_${file.name}">
                                <span class="me-2">${icon}</span>
                                ${file.name}
                            </label>
                        </div>
                        <div>
                            <small class="text-muted me-2">${fileSize}</small>
                        </div>
                    `;
                    
                    filesList.appendChild(item);
                    
                    // 存储文件信息
                    item.dataset.file = JSON.stringify(file);
                    
                    // 为目录添加点击事件
                    if (file.is_dir) {
                        item.addEventListener('dblclick', () => {
                            const modal = document.getElementById('webdavBrowseModal');
                            modal.dataset.currentPath = file.path;
                            loadWebDAVContents(connectionName, file.path);
                        });
                    }
                });
            })
            .catch(error => {
                console.error('加载 WebDAV 内容失败:', error);
                loadingSpinner.style.display = 'none';
                filesList.innerHTML = `
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-triangle me-2"></i>
                        加载失败: ${error}
                    </div>
                `;
            });
    }
    
    // 更新路径面包屑
    function updatePathBreadcrumb(connectionName, path) {
        const breadcrumb = document.getElementById('webdavPathBreadcrumb');
        const parts = path.split('/').filter(p => p !== '');
        
        let html = `
            <nav aria-label="breadcrumb">
                <ol class="breadcrumb mb-0">
                    <li class="breadcrumb-item">
                        <a href="#" data-path="/" class="text-decoration-none">
                            <i class="fas fa-network-wired me-1"></i>${connectionName}
                        </a>
                    </li>
        `;
        
        if (parts.length === 0) {
            html += `<li class="breadcrumb-item active" aria-current="page">根目录</li>`;
        } else {
            let currentPath = '';
            parts.forEach((part, index) => {
                currentPath += '/' + part;
                if (index === parts.length - 1) {
                    html += `<li class="breadcrumb-item active" aria-current="page">${part}</li>`;
                } else {
                    html += `
                        <li class="breadcrumb-item">
                            <a href="#" data-path="${currentPath}" class="text-decoration-none">${part}</a>
                        </li>
                    `;
                }
            });
        }
        
        html += `</ol></nav>`;
        breadcrumb.innerHTML = html;
        
        // 添加面包屑点击事件
        breadcrumb.querySelectorAll('a[data-path]').forEach(a => {
            a.addEventListener('click', (e) => {
                e.preventDefault();
                const clickedPath = a.dataset.path;
                const modal = document.getElementById('webdavBrowseModal');
                modal.dataset.currentPath = clickedPath;
                loadWebDAVContents(connectionName, clickedPath);
            });
        });
    }
    
    // 文件大小格式化
    function formatFileSize(size) {
        if (size < 1024) return size + ' B';
        if (size < 1024 * 1024) return (size / 1024).toFixed(1) + ' KB';
        if (size < 1024 * 1024 * 1024) return (size / (1024 * 1024)).toFixed(1) + ' MB';
        return (size / (1024 * 1024 * 1024)).toFixed(1) + ' GB';
    }
    
    // 获取当前路径
    function getCurrentPath() {
        const modal = document.getElementById('webdavBrowseModal');
        return modal ? modal.dataset.currentPath : null;
    }
    
    // 获取选中的WebDAV项目
    function getSelectedWebDAVItems() {
        const filesList = document.getElementById('webdavFilesList');
        const selectedItems = [];
        
        filesList.querySelectorAll('.form-check-input:checked').forEach(checkbox => {
            const item = checkbox.closest('.list-group-item');
            if (item && item.dataset.file) {
                const file = JSON.parse(item.dataset.file);
                selectedItems.push(file);
            }
        });
        
        return selectedItems;
    }
    
    // 下载选中的项目
    function downloadSelectedItems(connectionName, items, targetFolder) {
        // 显示进度
        showNotification(`开始下载 ${items.length} 个文件...`, true);
        
        // 逐个下载文件
        let completed = 0;
        let errors = 0;
        
        items.forEach(item => {
            if (item.is_dir) {
                // 目录暂不处理
                completed++;
                return;
            }
            
            // 创建表单数据
            const formData = new FormData();
            formData.append('path', item.path);
            formData.append('target_folder', targetFolder);
            
            fetch(`/api/webdav-client/${connectionName}/download`, {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                completed++;
                if (!data.success) errors++;
                
                if (completed === items.length) {
                    // 所有下载完成
                    const errorMsg = errors > 0 ? `，${errors} 个失败` : '';
                    showNotification(`下载完成${errorMsg}，刷新查看文件`, true);
                }
            })
            .catch(error => {
                console.error('下载文件失败:', error);
                completed++;
                errors++;
                
                if (completed === items.length) {
                    // 所有下载完成
                    const errorMsg = errors > 0 ? `，${errors} 个失败` : '';
                    showNotification(`下载完成${errorMsg}，刷新查看文件`, true);
                }
            });
        });
    }
    
    // 显示文件夹选择器
    function showFolderSelector(title, initialPath, callback) {
        let folderSelector = document.getElementById('folderSelectorModal');
        
        if (!folderSelector) {
            folderSelector = document.createElement('div');
            folderSelector.className = 'modal fade';
            folderSelector.id = 'folderSelectorModal';
            folderSelector.setAttribute('tabindex', '-1');
            folderSelector.setAttribute('aria-hidden', 'true');
            
            folderSelector.innerHTML = `
                <div class="modal-dialog">
                    <div class="modal-content">
                        <div class="modal-header">
                            <h5 class="modal-title" id="folderSelectorTitle">选择文件夹</h5>
                            <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                        </div>
                        <div class="modal-body">
                            <div id="folderSelectorBreadcrumb" class="mb-3"></div>
                            <div id="folderSelectorList" class="list-group" style="max-height: 300px; overflow-y: auto;"></div>
                        </div>
                        <div class="modal-footer">
                            <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">取消</button>
                            <button type="button" class="btn btn-primary" id="selectFolderBtn">选择此文件夹</button>
                        </div>
                    </div>
                </div>
            `;
            
            document.body.appendChild(folderSelector);
        }
        
        const modal = new bootstrap.Modal(folderSelector);
        document.getElementById('folderSelectorTitle').textContent = title;
        
        // 存储当前路径和回调
        folderSelector.dataset.currentPath = initialPath;
        folderSelector.dataset.callback = String(callback);
        
        // 加载文件夹列表
        loadFolderList(initialPath);
        
        // 选择按钮事件
        document.getElementById('selectFolderBtn').onclick = function() {
            const selectedPath = folderSelector.dataset.currentPath;
            modal.hide();
            callback(selectedPath);
        };
        
        // 显示模态窗口
        modal.show();
    }
    
    // 加载文件夹列表
 // 加载文件夹列表
// 加载文件夹列表
function loadFolderList(path) {
    const folderList = document.getElementById('folderSelectorList');
    const breadcrumb = document.getElementById('folderSelectorBreadcrumb');
    
    // 显示加载中
    folderList.innerHTML = '<div class="text-center p-3"><div class="spinner-border spinner-border-sm" role="status"></div> 加载中...</div>';
    console.log('加载文件夹列表:', path);
    
    // 检查路径，如果是根目录，改用空字符串
    const requestPath = path.startsWith('/') ? path.slice(1) : path;
    
    // 获取文件夹列表
    fetch(`/api/files?path=${encodeURIComponent(requestPath)}`)
        .then(response => {
            if (!response.ok) {
                if (response.status === 403) {
                    // 创建可选择的顶级目录
                    createTopLevelDirectories(folderList);
                    // 更新面包屑
                    updateFolderBreadcrumb(path);
                    return null; // 不继续处理
                }
                throw new Error(`HTTP 错误 ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data === null) return; // 如果是403错误并已处理，则返回
            
            folderList.innerHTML = '';
            
            // 更新面包屑
            updateFolderBreadcrumb(path);
                
            // 添加上级目录选项 (只有当不在根目录时才显示)
            if (path !== '/' && path !== '') {
                const parentItem = document.createElement('a');
                parentItem.className = 'list-group-item list-group-item-action d-flex align-items-center';
                parentItem.innerHTML = '<i class="fas fa-level-up-alt me-2"></i> 上级目录';
                
                parentItem.addEventListener('click', () => {
                    const pathParts = path.split('/').filter(p => p !== '');
                    pathParts.pop();
                    const parentPath = pathParts.length > 0 ? '/' + pathParts.join('/') : '/';
                    
                    document.getElementById('folderSelectorModal').dataset.currentPath = parentPath;
                    loadFolderList(parentPath);
                });
                
                folderList.appendChild(parentItem);
            }
            
            // 添加安全检查，防止undefined错误
            if (!data || !data.items || !Array.isArray(data.items)) {
                folderList.innerHTML += '<div class="text-center p-3 text-muted">无法获取文件夹信息</div>';
                return;
            }
        
            // 仅显示文件夹
            const folders = data.items.filter(item => item.type === "folder");
            
            if (folders.length === 0) {
                folderList.innerHTML += '<div class="text-center p-3 text-muted">没有子文件夹</div>';
            }
            
            // 添加文件夹列表
            folders.forEach(folder => {
                const item = document.createElement('a');
                item.className = 'list-group-item list-group-item-action d-flex align-items-center';
                item.innerHTML = `<i class="fas fa-folder text-warning me-2"></i> ${folder.name}`;
                
                item.addEventListener('click', () => {
                    const newPath = path === '/' ? 
                        '/' + folder.path : 
                        folder.path;
                    
                    document.getElementById('folderSelectorModal').dataset.currentPath = newPath;
                    loadFolderList(newPath);
                });
                
                folderList.appendChild(item);
            });
            
            // 添加"选择当前文件夹"按钮
            document.getElementById('selectFolderBtn').disabled = false;
        })
        .catch(error => {
            console.error('加载文件夹列表失败:', error);
            folderList.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    加载失败: ${error.message || error}
                </div>
                <div class="text-center mt-3">
                    <button class="btn btn-sm btn-secondary retry-folder-btn">
                        <i class="fas fa-sync me-1"></i> 重试
                    </button>
                </div>
            `;
            
            // 添加重试按钮事件监听
            folderList.querySelector('.retry-folder-btn')?.addEventListener('click', () => {
                loadFolderList(path);
            });
        });
}

// 创建顶级目录列表(当无法访问根目录时使用)
function createTopLevelDirectories(container) {
    container.innerHTML = '';
    
    // 添加一个创建新的根级文件夹的选项
    const newFolderItem = document.createElement('div');
    newFolderItem.className = 'alert alert-info mb-3';
    newFolderItem.innerHTML = `
        <p><i class="fas fa-info-circle me-2"></i> 无法直接访问根目录。请选择以下选项：</p>
        <div class="mt-2">
            <button class="btn btn-outline-primary create-root-folder-btn">
                <i class="fas fa-folder-plus me-1"></i> 创建顶级文件夹
            </button>
            <button class="btn btn-outline-secondary ms-2 use-current-folder-btn">
                <i class="fas fa-check me-1"></i> 使用当前路径
            </button>
        </div>
    `;
    container.appendChild(newFolderItem);
    
    // 添加创建顶级文件夹的事件
    container.querySelector('.create-root-folder-btn').addEventListener('click', () => {
        const folderName = prompt('请输入顶级文件夹名称：');
        if (folderName) {
            // 创建新文件夹
            fetch('/api/folders', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `folder_path=${encodeURIComponent(folderName)}`
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    // 设置当前路径为新创建的文件夹
                    let newPath = '/' + folderName;
                    document.getElementById('folderSelectorModal').dataset.currentPath = newPath;
                    newPath = newPath.startsWith('/') ? newPath.slice(1) : newPath;

                    loadFolderList(newPath);
                    showNotification(`文件夹 "${folderName}" 创建成功`, true);
                } else {
                    showNotification(`创建文件夹失败: ${data.error || '未知错误'}`, false);
                }
            })
            .catch(error => {
                showNotification(`创建文件夹请求失败: ${error.message}`, false);
            });
        }
    });
    
    // 使用当前路径按钮事件
    container.querySelector('.use-current-folder-btn').addEventListener('click', () => {
        // 启用选择按钮
        document.getElementById('selectFolderBtn').disabled = false;
    });
    
    // 添加一些常用目录的快捷方式
    // const commonDirs = [
    //     { name: '下载', path: '/downloads' },
    //     { name: '文档', path: '/documents' },
    //     { name: '媒体', path: '/media' },
    //     { name: '图片', path: '/images' }
    // ];
    
    const quickAccessDiv = document.createElement('div');
    quickAccessDiv.className = 'list-group mt-3';
    quickAccessDiv.innerHTML = '<div class="list-group-item list-group-item-info">快速访问</div>';
    
    commonDirs.forEach(dir => {
        const item = document.createElement('a');
        item.className = 'list-group-item list-group-item-action d-flex align-items-center';
        item.innerHTML = `<i class="fas fa-folder text-warning me-2"></i> ${dir.name}`;
        
        item.addEventListener('click', () => {
            // 尝试加载这个目录
            document.getElementById('folderSelectorModal').dataset.currentPath = dir.path;
            loadFolderList(dir.path);
        });
        
        quickAccessDiv.appendChild(item);
    });
    
    container.appendChild(quickAccessDiv);
}
    // 更新文件夹选择器面包屑
    function updateFolderBreadcrumb(path) {
        const breadcrumb = document.getElementById('folderSelectorBreadcrumb');
        const parts = path.split('/').filter(p => p !== '');
        
        let html = `
            <nav aria-label="breadcrumb">
                <ol class="breadcrumb mb-0">
                    <li class="breadcrumb-item">
                        <a href="#" data-path="/" class="text-decoration-none">
                            <i class="fas fa-home me-1"></i>根目录
                        </a>
                    </li>
        `;
        
        if (parts.length > 0) {
            let currentPath = '';
            parts.forEach((part, index) => {
                currentPath += '/' + part;
                if (index === parts.length - 1) {
                    html += `<li class="breadcrumb-item active" aria-current="page">${part}</li>`;
                } else {
                    html += `
                        <li class="breadcrumb-item">
                            <a href="#" data-path="${currentPath}" class="text-decoration-none">${part}</a>
                        </li>
                    `;
                }
            });
        }
        
        html += `</ol></nav>`;
        breadcrumb.innerHTML = html;
        
        // 添加面包屑点击事件
        breadcrumb.querySelectorAll('a[data-path]').forEach(a => {
            a.addEventListener('click', (e) => {
                e.preventDefault();
                const clickedPath = a.dataset.path;
                document.getElementById('folderSelectorModal').dataset.currentPath = clickedPath;
                loadFolderList(clickedPath);
            });
        });
    }
    
    // 测试连接
    testConnectionBtn.addEventListener('click', function() {
        const btn = this;
        const originalText = btn.textContent;
        
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-circle-notch fa-spin"></i> 测试中...';
        
        const connection = {
            name: connectionName.value || "测试连接",
            url: connectionUrl.value,
            username: connectionUsername.value,
            password: connectionPassword.value,
            folder: connectionFolder.value,
            enabled: connectionEnabled.checked
        };
        
        fetch('/api/webdav-client/test-connection', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(connection)
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showNotification(data.message, true);
            } else {
                showNotification(data.message, false);
            }
            
            btn.disabled = false;
            btn.textContent = originalText;
        })
        .catch(error => {
            console.error('测试连接失败:', error);
            showNotification('测试连接失败: ' + error, false);
            btn.disabled = false;
            btn.textContent = originalText;
        });
    });
    
    // 保存连接
    saveConnectionBtn.addEventListener('click', function() {
        const btn = this;
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-circle-notch fa-spin"></i> 保存中...';
        
        const connection = {
            name: connectionName.value,
            url: connectionUrl.value,
            username: connectionUsername.value,
            password: connectionPassword.value,
            folder: connectionFolder.value || "/",
            enabled: connectionEnabled.checked
        };
        
        // 验证表单
        if (!connection.name) {
            showNotification('请输入连接名称', false);
            btn.disabled = false;
            btn.textContent = isEditing ? '更新连接' : '添加连接';
            return;
        }
        
        if (!connection.url) {
            showNotification('请输入WebDAV服务器URL', false);
            btn.disabled = false;
            btn.textContent = isEditing ? '更新连接' : '添加连接';
            return;
        }
        
        // 根据是否编辑状态选择API
        let url, method;
        
        if (isEditing) {
            url = `/api/webdav-client/connections/${currentEditingConnection}`;
            method = 'PUT';
        } else {
            url = '/api/webdav-client/connections';
            method = 'POST';
        }
        
        fetch(url, {
            method: method,
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(connection)
        })
        .then(response => response.json())
        .then(data => {
            showNotification(data.message, data.success);
            
            if (data.success) {
                // 返回到连接列表
                document.getElementById('webdavConnectionFormContainer').style.display = 'none';
                document.getElementById('webdavConnectionListContainer').style.display = 'block';
                
                // 重新加载连接列表
                loadConnections();
            }
            
            btn.disabled = false;
            btn.textContent = isEditing ? '更新连接' : '添加连接';
        })
        .catch(error => {
            console.error('保存连接失败:', error);
            showNotification('保存连接失败: ' + error, false);
            btn.disabled = false;
            btn.textContent = isEditing ? '更新连接' : '添加连接';
        });
    });
    
    // 取消编辑
    cancelEditBtn.addEventListener('click', function() {
        document.getElementById('webdavConnectionFormContainer').style.display = 'none';
        document.getElementById('webdavConnectionListContainer').style.display = 'block';
    });
    
    // 显示添加按钮点击事件
    document.getElementById('showAddWebdavBtn').addEventListener('click', showAddConnectionForm);
    
    // 当模态窗口显示时加载连接列表
    $('#webdavClientModal').on('shown.bs.modal', function () {
        loadConnections();
    });
});

// 模态窗口管理
function setupModalManager() {
    // 防止模态窗口堆叠
    $(document).on('show.bs.modal', '.modal', function() {
        // 获取当前可见模态窗口数量
        const visibleModals = $('.modal:visible').length;
        
        // 如果已经有模态窗口打开，先关闭它们
        if (visibleModals > 0) {
            $('.modal').modal('hide');
        }
        
        // 设置z-index以确保正确显示
        setTimeout(() => {
            const zIndex = 1050 + (10 * $('.modal:visible').length);
            $(this).css('z-index', zIndex);
            $('.modal-backdrop').not('.modal-stack').css('z-index', zIndex - 1).addClass('modal-stack');
        }, 10);
    });
}

// DOM加载完成后初始化模态窗口管理
document.addEventListener('DOMContentLoaded', function() {
    setupModalManager();
});


