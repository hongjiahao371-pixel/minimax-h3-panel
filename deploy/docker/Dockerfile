# MiniMax-H3 面板 + ComfyUI 一体镜像（UGOS Pro UPK / 自托管 Docker 通用）
# 单容器同时运行 ComfyUI(8188) 与面板(8189)，共享同一 python/torch 环境（后处理直接可用）
# 构建示例（在有 Docker 的机器上，构建上下文 = 本目录）：
#   docker build -t jvsheng/minimax-h3-panel:upk-v1 .
#   docker save jvsheng/minimax-h3-panel:upk-v1 | gzip > h3panel-amd64.tar.gz

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai
ENV PYTHONUNBUFFERED=1

# ubuntu 源换阿里云（国内构建稳定性）
RUN sed -i -e 's|http://archive.ubuntu.com|http://mirrors.aliyun.com|g' \
           -e 's|http://security.ubuntu.com|http://mirrors.aliyun.com|g' \
           -e 's|http://cn.archive.ubuntu.com|http://mirrors.aliyun.com|g' \
           /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null || true
RUN sed -i -e 's|http://archive.ubuntu.com|http://mirrors.aliyun.com|g' \
           -e 's|http://security.ubuntu.com|http://mirrors.aliyun.com|g' \
           /etc/apt/sources.list 2>/dev/null || true

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg python3 python3-pip python3-venv ca-certificates tzdata curl \
    && rm -rf /var/lib/apt/lists/*

# torch cu128（pip wheel 自带 CUDA 运行库，无需 cuda 基底镜像）
RUN python3 -m venv /app/venv \
    && /app/venv/bin/pip install --no-cache-dir \
       torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# ComfyUI 依赖 + GGUF 支持 + 面板依赖
COPY ComfyUI/requirements.txt /tmp/req-comfy.txt
RUN /app/venv/bin/pip install --no-cache-dir -r /tmp/req-comfy.txt \
    && /app/venv/bin/pip install --no-cache-dir gguf flask requests pillow

# 应用代码（构建上下文提供）
COPY ComfyUI /app/ComfyUI
COPY panel /app/panel
COPY postproc /app/postproc
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh \
    && mkdir -p /models/diffusion_models /models/text_encoders /models/vae /models/loras /models/unet \
       /output /data/comfyui-input /data/comfyui-user /data/home

ENV HOME=/data/home
EXPOSE 8188 8189
ENTRYPOINT ["/entrypoint.sh"]
