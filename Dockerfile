FROM python:3.11-slim

WORKDIR /ZAY-CLOUD

# 复制并安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 安装OpenCV所需的所有依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 复制应用代码
COPY . .
RUN mkdir -p /ZAY-CLOUD/storage

# 暴露端口
EXPOSE 5888

# 运行 Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "5888"]