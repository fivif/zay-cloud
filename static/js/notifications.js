// 全局通知系统
function showGlobalNotification(message, isSuccess = true, duration = 3000) {
    const container = document.getElementById('globalNotificationContainer');
    
    // 创建通知元素
    const notification = document.createElement('div');
    notification.className = `alert ${isSuccess ? 'alert-success' : 'alert-danger'} alert-dismissible fade show mb-2`;
    notification.role = 'alert';
    notification.style.boxShadow = '0 4px 8px rgba(0,0,0,0.1)';
    
    // 设置通知内容
    notification.innerHTML = `
        ${isSuccess ? '<i class="fas fa-check-circle me-2"></i>' : '<i class="fas fa-exclamation-circle me-2"></i>'}
        <span>${message}</span>
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // 添加到容器
    container.appendChild(notification);
    
    // 创建Bootstrap警告对象以便后续关闭
    const alertInstance = new bootstrap.Alert(notification);
    
    // 设置定时关闭
    setTimeout(() => {
        alertInstance.close();
        // 删除DOM元素
        setTimeout(() => notification.remove(), 500);
    }, duration);
    
    return notification;
}

// 覆盖现有的showNotification函数，使用新的全局通知系统
window.showNotification = function(message, isSuccess = true) {
    return showGlobalNotification(message, isSuccess);
};

// 文档加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    console.log('通知系统已初始化');
});