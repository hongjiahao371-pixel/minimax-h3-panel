#!/bin/bash
# MiniMax-H3 面板容器入口：环境探测 → 目录接线 → 配置生成 → 双服务启动
set -e

# ---------- 1. 主机 NVIDIA 驱动库发现（devices: 直通的经典配套） ----------
# /hostlibs 只读挂载宿主机驱动库目录。注意：绝不能把整个宿主 lib 目录塞进
# LD_LIBRARY_PATH（宿主 glibc 与容器不匹配会导致 mkdir 等基础工具
# "stack smashing detected" 崩溃）——只挑 NVIDIA 相关库复制到容器内目录。
if [ -d /hostlibs ]; then
  LIBDIR=$(find /hostlibs -maxdepth 2 -name "libnvidia-ml.so*" 2>/dev/null | head -1 | xargs -r dirname)
  if [ -n "$LIBDIR" ]; then
    mkdir -p /driver-libs
    # shellcheck disable=SC2044
    for f in $(find /hostlibs -maxdepth 2 \( -name "libnvidia-ml.so*" -o -name "libcuda.so*" \
        -o -name "libnvidia-ptxjitcompiler.so*" -o -name "libnvidia-nvvm.so*" \
        -o -name "libnvidia-opencl.so*" -o -name "libnvidia-allocator.so*" \) 2>/dev/null); do
      cp -Lf "$f" /driver-libs/ 2>/dev/null || true
    done
    export LD_LIBRARY_PATH="/driver-libs${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    echo "[entrypoint] NVIDIA 驱动库已隔离注入: $(ls /driver-libs | wc -l) 个文件"
  else
    echo "[entrypoint] WARN: /hostlibs 挂载了但没找到 NVIDIA 驱动库——容器将以 CPU 降级模式运行"
  fi
fi

# ---------- 2. 持久卷接线（符号链接到 ComfyUI 标准目录） ----------
mkdir -p /app/ComfyUI/models
for d in diffusion_models text_encoders vae loras unet; do
  mkdir -p "/models/$d"
  ln -sfn "/models/$d" "/app/ComfyUI/models/$d"
done
mkdir -p /output /data/comfyui-input /data/comfyui-user /data/home
ln -sfn /output /app/ComfyUI/output
ln -sfn /data/comfyui-input /app/ComfyUI/input
ln -sfn /data/comfyui-user /app/ComfyUI/user

# ---------- 3. 面板配置（缺失时生成，已有则保留用户改动） ----------
if [ ! -f /data/config.json ]; then
  cat > /data/config.json <<'EOF'
{
  "comfy_url": "http://127.0.0.1:8188",
  "port": 8189,
  "output_dir": "/output",
  "data_dir": "/data",
  "postproc_dir": "/app/postproc",
  "ffmpeg": "/usr/bin/ffmpeg",
  "ffprobe": "/usr/bin/ffprobe",
  "python": "/app/venv/bin/python3",
  "postproc_home": "",
  "comfy_models_dir": "/models/diffusion_models",
  "default_model": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
  "vram_budget_tokens": 115000000,
  "vram_warn_tokens": 95000000,
  "access_password": ""
}
EOF
  echo "[entrypoint] 已生成默认配置 /data/config.json"
fi

# ---------- 4. ComfyUI 后台启动 ----------
# 监听 0.0.0.0：UPK/单容器部署时 8188 会发布到宿主机，前端工作流可视化
# 需要浏览器直连 ComfyUI 的 WS 事件通道（面板自身走 127.0.0.1 同样可用）
cd /app/ComfyUI
/app/venv/bin/python main.py --listen 0.0.0.0 --port 8188 --disable-auto-launch \
  > /data/comfyui.log 2>&1 &
echo "[entrypoint] ComfyUI 启动中（日志 /data/comfyui.log）..."
/app/venv/bin/python - <<'PY'
import time, urllib.request
for _ in range(150):
    try:
        urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=2)
        print("[entrypoint] ComfyUI 就绪")
        break
    except Exception:
        time.sleep(2)
PY

# ---------- 5. 面板前台 ----------
cd /app/panel
exec /app/venv/bin/python panel.py
