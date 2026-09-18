#!/usr/bin/env bash
# 给 ComfyUI 安装 GGUF 模型加载支持（面板模型库的 GGUF 条目依赖它）
# 用法：在 ComfyUI 所在机器上执行  bash install_gguf_node.sh [comfyui_dir] [venv_python]
#   comfyui_dir 默认 /volume2/comfyui/ComfyUI（作者 NAS 布局）
#   venv_python 默认 /usr/bin/python3（配合 HOME 用户目录装依赖的布局）
set -e
COMFY_DIR="${1:-/volume2/comfyui/ComfyUI}"
PY="${2:-/usr/bin/python3}"
GH_PROXY="${GH_PROXY:-https://ghproxy.net}"   # github 直连不通时自动走代理前缀

cd "$COMFY_DIR"

if [ -d custom_nodes/ComfyUI-GGUF ]; then
  echo "[gguf] ComfyUI-GGUF 已存在，跳过 clone"
else
  URL="https://github.com/city96/ComfyUI-GGUF.git"
  echo "[gguf] clone $URL"
  git clone "$URL" custom_nodes/ComfyUI-GGUF 2>/dev/null || \
    git clone "$GH_PROXY/$URL" custom_nodes/ComfyUI-GGUF
fi

echo "[gguf] 安装 python 依赖 gguf"
"$PY" -m pip install --user --break-system-packages -i https://pypi.org/simple gguf 2>/dev/null \
  || "$PY" -m pip install gguf \
  || "$PY" -m pip install --user gguf

cat <<'NOTE'

[gguf] 安装完成。请重启 ComfyUI 后验证：
  curl -s http://127.0.0.1:8188/object_info/UnetLoaderGGUF   # 返回 JSON 即成功
GGUF 权重文件放到 models/unet 或 models/diffusion_models 目录，
面板模型库「下载部署」会自动放到正确位置。
NOTE
