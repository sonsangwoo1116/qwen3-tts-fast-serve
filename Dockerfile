# qwen3-tts-fast-serve
# Optimized Qwen3-TTS serving engine (paged KV cache, CUDA graphs, audio post-processing)

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Build flash-attention from source to match PyTorch 2.6.0 ABI exactly
RUN pip install --no-cache-dir ninja packaging psutil && \
    git clone --depth 1 --branch v2.7.4.post1 https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attn && \
    cd /tmp/flash-attn && \
    MAX_JOBS=4 pip install --no-cache-dir --no-build-isolation . && \
    rm -rf /tmp/flash-attn

# Install main dependencies
RUN pip install --no-cache-dir \
    "qwen-tts>=0.0.5" \
    "transformers>=4.57.3" \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.32.0" \
    "soundfile>=0.12.0" \
    librosa \
    "pyzmq>=27.1.0" \
    "xxhash>=3.6.0" \
    python-multipart \
    numpy \
    scipy

# Copy application code
COPY . /app

# Install the package
RUN pip install --no-cache-dir -e .

ENV HOST=0.0.0.0
ENV PORT=8880

EXPOSE 8880

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8880/health || exit 1

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8880"]
