# 构建上下文须为仓库根目录（docker-compose: context: ., dockerfile: backend/Dockerfile）。
# /app 为项目根，与 database.py 中 PROJECT_ROOT 一致；SQLite 默认为 /app/video_rating.db。

FROM python:3.9-slim-bookworm

WORKDIR /app
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# ffmpeg：moviepy；libsndfile：soundfile/librosa；glib/gomp：opencv-python-headless 常见依赖
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /tmp/requirements.txt

# 先装 CPU 版 torch/torchaudio（避免默认 PyPI 拉 CUDA 轮子）；其余依赖单独安装（requirements 中仍保留版本约束便于本地开发一致）
RUN pip install --upgrade pip setuptools wheel \
    && pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
    && sed '/^torch==/d; /^torchaudio==/d' /tmp/requirements.txt > /tmp/requirements-notorch.txt \
    && pip install -r /tmp/requirements-notorch.txt

COPY backend /app/backend

EXPOSE 8000

# 首次拉模型/导入较慢，start-period 给足余量
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
