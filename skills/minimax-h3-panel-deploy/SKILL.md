---
name: minimax-h3-panel-deploy
description: 部署/更新/排障 MiniMax-H3 视频生成面板（github.com/hongjiahao371-pixel/minimax-h3-panel，一个驱动本机 ComfyUI 生成带原生音频视频的 Flask 面板）。当用户要求安装、部署、搭建、更新、配置或排查「H3 面板 / minimax-h3-panel / H3 视频生成面板 / ComfyUI 的 H3 前端」时使用；也包括为其准备 ComfyUI 与模型环境、安装 GGUF 组件、配置后处理引擎、访问口令或云端 LLM 提示词优化。
---

# MiniMax-H3 面板部署

你要部署的是一个小型 Flask 面板（默认端口 8189），它通过 HTTP 驱动同一台机器上已装好的 ComfyUI（默认 8188）生成视频。面板本身不含模型；生成质量取决于 ComfyUI 侧的 MiniMax-H3 权重。开始前先向用户确认三件事，能省掉大量返工：

1. **装在哪台机器**——面板必须与 ComfyUI 同机（或至少能访问它的 HTTP 端口）；
2. **ComfyUI 是否已经能跑通 H3**——面板不解决 ComfyUI/驱动/权重本身的安装，那部分失败要引导用户先装 ComfyUI（新版自带 H3 节点包）并从 HuggingFace `Comfy-Org/MiniMax-H3` 下载全套权重（DiT + Qwen3VL 文本编码器 + 视频/音频 VAE）；
3. **要哪些可选能力**——AI 增强（插帧/超分）、GGUF 量化支持、云端 LLM 提示词优化、开机自启。默认全不装也能用。

## 第 0 步：环境体检

先跑仓库自带的体检脚本，按 FAIL 项逐个处理，全绿再继续：

```bash
cd <仓库目录>
bash skills/minimax-h3-panel-deploy/scripts/check_env.sh
```

脚本是只读的，可用环境变量覆盖检查目标：`COMFY_URL`（默认 `http://127.0.0.1:8188`）、`MODELS_DIR`、`PANEL_DIR`。它检查 python 版本与依赖、ffmpeg/ffprobe、GPU、ComfyUI 可达性、显存与磁盘。**特别注意**：`nvidia-smi` 不存在或报「couldn't communicate with driver」说明显卡/驱动不在位，此时 ComfyUI 大概率起不来——不要继续装面板，先把 GPU 环境修好。

## 第 1 步：获取代码与依赖

```bash
git clone https://github.com/hongjiahao371-pixel/minimax-h3-panel.git
cd minimax-h3-panel
pip install -r requirements.txt   # 只有 flask / requests / pillow
```

github 直连不通时用代理前缀：`git clone https://ghproxy.net/https://github.com/hongjiahao371-pixel/minimax-h3-panel.git`。pip 镜像选一个稳的（pypi.org 直连慢但稳；清华源偶发 403、阿里云 CDN 可能挂起）。

## 第 2 步：生成配置

```bash
cp panel/config.example.json panel/config.json
```

然后编辑 `panel/config.json`，**必改两项**：

- `output_dir`：ComfyUI 的输出目录（面板从这里读取生成的视频）。
- `comfy_models_dir`：ComfyUI 的 `models/diffusion_models` 目录。**必须指向 ComfyUI 实际扫描的目录本身**。如果该目录是指向别处权重仓的逐文件符号链接（一些 NAS 教程会这么建），把新文件放进权重仓 ComfyUI 是看不见的——这正是模型库下载器的落盘点，指错位置会出现「下载成功但模型列表为空」。拿不准就放一个空文件进去再问一次 ComfyUI 的 `/object_info/UNETLoader` 验证。

其余字段（`comfy_url`、`port`、`postproc_dir`、`vram_budget_tokens` 等）见仓库 README 的配置说明表；12GB 显存参考预算 115000000。每个字段都可用 `PANEL_<大写字段名>` 环境变量覆盖。

## 第 3 步：启动与验收

```bash
cd panel && python3 panel.py
```

按顺序验收，任何一步不过先解决再往下：

1. `curl -s http://127.0.0.1:8189/api/models` 返回 `"ok": true`，且 `installed` 里有 H3 权重文件名；
2. 浏览器打开 `http://<IP>:8189`，模型库页能看到模型卡片；
3. 提交一次短时长生成（864×480、约 5 秒）确认端到端出片。

都通过后再装可选组件。

## 可选组件（按用户需求逐个装）

- **AI 增强（RIFE 插帧 / Real-ESRGAN 超分）**：`cd postproc && bash install.sh`。要求 `config.json` 的 `python` 指向一个有 CUDA 版 torch 的解释器（通常与 ComfyUI 同环境）。
- **GGUF 量化支持**：`bash deploy/install_gguf_node.sh [comfyui目录] [python]`。装完**必须重启 ComfyUI**，用 `curl -s http://127.0.0.1:8188/object_info/UnetLoaderGGUF` 返回 JSON 验证。之后模型库里的 GGUF/LoRA 条目点「下载部署」即可。
- **云端 LLM 提示词优化/拆分**：引导用户在页面「高级选项 → 提示词优化 · 云端 LLM」填 OpenAI 兼容端点（API Base / 模型名 / Key）。不配也能用（本地规则兜底）。注意：MiniMax M3 等推理模型的回复带 `<think>` 思考块，面板已自动剥离，但 token 上限要够（面板已默认调大）。
- **成片自动备份**：`config.json` 设 `auto_backup_dir` 后重启面板。
- **访问口令**（局域网外暴露时务必设置）：`config.json` 的 `access_password`，重启后浏览器首次访问会弹登录框。用 curl 调 API 时需带 `X-Panel-Token: <口令的sha256>` 头或走一次 `/api/login`。
- **开机自启**：复制 `deploy/h3-panel.service` 到 `/etc/systemd/system/`，改掉里面的 `YOUR_USER` 与路径占位符，`sudo systemctl daemon-reload && sudo systemctl enable --now h3-panel`。

## 更新已有部署

```bash
cd <仓库目录> && git pull
sudo systemctl restart h3-panel   # 或重启你启动面板的进程
```

**改了 `panel/templates/index.html` 或任何页面文件后必须重启面板进程**——Flask 生产模式下模板有缓存，不重启浏览器会一直拿旧 JS。跨机部署的（面板代码在别处、通过 scp 同步），同步后同样要重启。

## 排障

遇到「下载成功但模型列表为空」「页面改了不生效」「pip 装不上」「ComfyUI 连接失败但进程还在」这类问题，先读 `references/troubleshooting.md`——里面是作者机器上实际踩过并验证过解法的坑，按症状对号入座，不要从零开始猜。
