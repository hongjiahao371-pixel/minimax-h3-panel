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

# 给 ComfyUI-GGUF 的架构识别表补 MiniMax-H3（社区 H3 GGUF 均为无元数据裸张量格式，
# 靠特征键探测架构；截至 2026-09 主干识别表尚无 H3，不打补丁会报 Unknown model architecture）
CONVERT="$COMFY_DIR/custom_nodes/ComfyUI-GGUF/tools/convert.py"
if [ -f "$CONVERT" ] && ! grep -q "ModelMiniMaxH3" "$CONVERT"; then
  python3 - "$CONVERT" <<'PYEOF'
import sys
p = sys.argv[1]
src = open(p, encoding="utf-8").read()
cls = '''class ModelMiniMaxH3(ModelTemplate):
    arch = "minimax_h3"
    keys_detect = [
        ("adaln_t_table",),
        ("audio_patch_proj.weight",),
    ]

arch_list = [ModelFlux, ModelSD3, ModelAura, ModelHiDream, CosmosPredict2,
             ModelLTXV, ModelHyVid, ModelWan, ModelSDXL, ModelSD1, ModelLumina2, ModelMiniMaxH3]'''
import re
m = re.search(r"arch_list = \[ModelFlux[^\]]*\]", src)
if m:
    open(p, "w", encoding="utf-8").write(src.replace(m.group(0), cls))
    print("[gguf] 已给 convert.py 打 MiniMax-H3 识别补丁")
else:
    print("[gguf] 警告：未找到 arch_list，跳过补丁（新版可能已原生支持 H3）")
PYEOF
else
  echo "[gguf] convert.py 已含 H3 补丁或文件不存在，跳过"
fi

cat <<'NOTE'

[gguf] 安装完成。请重启 ComfyUI 后验证：
  curl -s http://127.0.0.1:8188/object_info/UnetLoaderGGUF   # 返回 JSON 即成功
GGUF 权重文件放到 models/unet 或 models/diffusion_models 目录，
面板模型库「下载部署」会自动放到正确位置。
NOTE
