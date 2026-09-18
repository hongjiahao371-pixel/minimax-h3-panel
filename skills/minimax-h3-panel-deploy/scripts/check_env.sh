#!/usr/bin/env bash
# MiniMax-H3 面板部署前环境体检（只读，不改任何东西）
# 可用环境变量覆盖检查目标：
#   COMFY_URL   ComfyUI 地址        （默认 http://127.0.0.1:8188）
#   MODELS_DIR  ComfyUI 扩散模型目录（可选，提供则检查）
#   PANEL_DIR   已 clone 的面板仓库 （可选，默认当前目录）
set -u
COMFY_URL="${COMFY_URL:-http://127.0.0.1:8188}"
MODELS_DIR="${MODELS_DIR:-}"
PANEL_DIR="${PANEL_DIR:-$(pwd)}"
PASS=0; FAIL=0

ok()   { echo "  [PASS] $1"; PASS=$((PASS+1)); }
bad()  { echo "  [FAIL] $1"; FAIL=$((FAIL+1)); }
warn() { echo "  [warn] $1"; }

echo "== 1. 基础工具 =="
command -v python3 >/dev/null 2>&1 && python3 --version | sed 's/^/         /' && ok "python3" || bad "python3 未安装（需要 3.10+）"
command -v ffmpeg >/dev/null 2>&1 && ok "ffmpeg" || bad "ffmpeg 未安装（面板拼接/尾帧/备份都依赖）"
command -v ffprobe >/dev/null 2>&1 && ok "ffprobe" || bad "ffprobe 未安装"
command -v git >/dev/null 2>&1 && ok "git" || bad "git 未安装"

echo "== 2. 面板 python 依赖 =="
for mod in flask requests PIL; do
  python3 -c "import $mod" >/dev/null 2>&1 && ok "python 模块 $mod" || bad "python 模块 $mod 缺失（pip install -r requirements.txt）"
done

echo "== 3. GPU =="
if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/         GPU: /'
    ok "NVIDIA 驱动可用"
  else
    bad "nvidia-smi 存在但无法通信——驱动挂了或显卡不在位（ComfyUI 会起不来）"
  fi
else
  warn "未找到 nvidia-smi——本机没有 NVIDIA 驱动；若 ComfyUI 在别的机器上可忽略"
fi

echo "== 4. ComfyUI (${COMFY_URL}) =="
STATS=$(curl -s --max-time 5 "$COMFY_URL/system_stats" 2>/dev/null)
if [ -n "$STATS" ] && echo "$STATS" | python3 -c "import json,sys;json.load(sys.stdin)" >/dev/null 2>&1; then
  ok "ComfyUI 可达"
  echo "$STATS" | python3 -c "
import json,sys
d=json.load(sys.stdin)
dev=(d.get('devices') or [{}])[0]
print(f'         GPU: {dev.get(\"name\",\"?\")}  显存 {round((dev.get(\"vram_total\") or 0)/2**30,1)}GB（空闲 {round((dev.get(\"vram_free\") or 0)/2**30,1)}GB）')
print(f'         内存 {round((d.get(\"system\",{}).get(\"ram_total\") or 0)/2**30,1)}GB')" 2>/dev/null
  VRAM=$(echo "$STATS" | python3 -c "import json,sys;print(round(((json.load(sys.stdin).get('devices') or [{}])[0].get('vram_total') or 0)/2**30,1))" 2>/dev/null)
  python3 -c "exit(0 if $VRAM and $VRAM >= 8 else 1)" 2>/dev/null \
    && ok "显存 ${VRAM}GB（≥8GB 可跑小量化，12GB+ 建议）" \
    || warn "显存 ${VRAM}GB 偏小——建议在模型库选 ≤10GB 的量化模型"
  # H3 节点包是否在（新版 ComfyUI 内置）
  curl -s --max-time 8 "$COMFY_URL/object_info/MiniMaxH3ImageToVideo" 2>/dev/null | grep -q '"MiniMaxH3ImageToVideo"' \
    && ok "H3 节点包已加载（MiniMaxH3ImageToVideo）" \
    || bad "ComfyUI 里没有 H3 节点——请升级 ComfyUI 到内置 MiniMax-H3 支持的版本"
else
  bad "ComfyUI 不可达（${COMFY_URL}）——先装好并启动 ComfyUI，面板无法替代它"
fi

echo "== 5. 模型目录与 H3 权重 =="
if [ -n "$MODELS_DIR" ]; then
  [ -d "$MODELS_DIR" ] && ok "MODELS_DIR 存在: $MODELS_DIR" || bad "MODELS_DIR 不存在: $MODELS_DIR"
  if [ -d "$MODELS_DIR" ]; then
    FOUND=$(find "$MODELS_DIR" -maxdepth 1 \( -name "*.safetensors" -o -name "*.gguf" \) -size +100M 2>/dev/null | wc -l | tr -d ' ')
    [ "$FOUND" -ge 1 ] && ok "发现 $FOUND 个权重文件" || bad "目录里没有 >100M 的权重文件——H3 DiT 还没下载"
  fi
else
  warn "未设置 MODELS_DIR，跳过模型目录检查（部署 config.json 时确认 comfy_models_dir）"
fi

echo "== 6. 磁盘 =="
FREE_KB=$(df -k "${PANEL_DIR:-.}" 2>/dev/null | awk 'NR==2{print $4}' | tr -dc '0-9')
FREE_GB=$([ -n "$FREE_KB" ] && echo $((FREE_KB / 1024 / 1024)))
[ -n "$FREE_GB" ] && [ "$FREE_GB" -ge 30 ] && ok "磁盘剩余 ${FREE_GB}GB（建议 ≥30GB，单模型 6~20GB）" \
  || warn "磁盘剩余 ${FREE_GB:-?}GB 偏紧——主模型 + 文生视频产出会很快吃满"

echo
echo "== 结论 =="
echo "  通过 $PASS 项，失败 $FAIL 项"
[ "$FAIL" -eq 0 ] && echo "  ✅ 环境就绪，可以按 SKILL.md 第 1 步继续部署" || echo "  ❌ 先解决上面的 FAIL 项再继续"
exit $([ "$FAIL" -eq 0 ] && echo 0 || echo 1)
