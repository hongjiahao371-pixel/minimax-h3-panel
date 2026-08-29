#!/usr/bin/env bash
# AI 增强引擎安装脚本：下载 RIFE 插帧 / Real-ESRGAN 超分所需的代码与权重
# 用法: bash install.sh   （在本目录执行；需要 python3 + torch(CUDA) 已可用）
# 国内网络自动走镜像（ghproxy.net / hf-mirror.com），失败可手动重跑
set -e
cd "$(dirname "$0")"

echo "==> [1/3] ComfyUI-Frame-Interpolation（RIFE 网络定义）"
if [ ! -d Frame-Interpolation ]; then
  try_dl() { curl -fsSL --insecure "$1" -o fi.zip 2>/dev/null; }
  try_dl "https://ghproxy.net/https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/archive/refs/heads/main.zip" \
    || try_dl "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation/archive/refs/heads/main.zip" \
    || { echo "下载失败，请手动下载 main.zip 解压为 Frame-Interpolation/"; exit 1; }
  python3 -c "import zipfile; zipfile.ZipFile('fi.zip').extractall()"
  mv ComfyUI-Frame-Interpolation-main Frame-Interpolation && rm -f fi.zip
fi
echo "    Frame-Interpolation 就绪"

echo "==> [2/3] RIFE 权重 rife47.pth（约 21MB）"
mkdir -p ckpts
if [ ! -s ckpts/rife47.pth ] || [ "$(stat -c%s ckpts/rife47.pth 2>/dev/null || echo 0)" -lt 1000000 ]; then
  curl -fsSL --insecure "https://hf-mirror.com/marduk191/rife/resolve/main/rife47.pth" -o ckpts/rife47.pth \
    || curl -fsSL --insecure "https://huggingface.co/marduk191/rife/resolve/main/rife47.pth" -o ckpts/rife47.pth \
    || { echo "下载失败，请手动下载 rife47.pth 放入 ckpts/"; exit 1; }
fi
echo "    rife47.pth 就绪 ($(stat -c%s ckpts/rife47.pth) bytes)"

echo "==> [3/3] Real-ESRGAN x2plus 权重（约 67MB）"
if [ ! -s ckpts/x2plus.pth ] || [ "$(stat -c%s ckpts/x2plus.pth 2>/dev/null || echo 0)" -lt 1000000 ]; then
  curl -fsSL --insecure "https://ghproxy.net/https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth" -o ckpts/x2plus.pth \
    || curl -fsSL --insecure "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth" -o ckpts/x2plus.pth \
    || { echo "下载失败，请手动下载 RealESRGAN_x2plus.pth 放入 ckpts/"; exit 1; }
fi
echo "    x2plus.pth 就绪 ($(stat -c%s ckpts/x2plus.pth) bytes)"

mkdir -p tmp
echo ""
echo "全部就绪。冒烟测试（可选，需要一段测试视频）："
echo "  python3 postproc_cli.py rife /path/to/test.mp4 /tmp/rife_out.mp4"
echo ""
echo "注意：postproc_cli.py 需要 torch(CUDA) 环境。若面板与 torch 不在同一 python 环境，"
echo "请在 panel/config.json 里设置 \"python\"（torch 所在的解释器）和 \"postproc_home\"（其用户包 HOME）。"
