    // 二维码生成功能
    document.getElementById('qrcodeModal').addEventListener('shown.bs.modal', function () {
        // 清空容器
        const container = document.getElementById('qrcodeContainer');
        container.innerHTML = '';
        
        // 获取当前URL
        const currentUrl = window.location.href;
        
        // 创建二维码
        new QRCode(container, {
            text: currentUrl,
            width: 200,
            height: 200,
            colorDark: "#000000",
            colorLight: "#ffffff",
            correctLevel: QRCode.CorrectLevel.H
        });
    });