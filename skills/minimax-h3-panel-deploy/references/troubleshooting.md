# 排障手册（作者机器实测的坑，按症状对号入座）

## 模型下载成功，但 ComfyUI 的模型列表里看不到

ComfyUI 的模型目录结构可能是指向别处权重仓的**逐文件符号链接**：`ComfyUI/models/diffusion_models/` 本身是真实目录，里面的 `xxx.safetensors` 是指向权重仓的软链。直接往权重仓放新文件，ComfyUI 永远看不见。解法：把文件放进 ComfyUI 扫描的目录本身（或在新位置放好后到扫描目录里补一个软链）。验证手段——放一个空探针文件后请求：

```bash
touch <comfy_models_dir>/_probe.gguf
curl -s http://127.0.0.1:8188/object_info/UNETLoader | python3 -c "import json,sys;print(json.load(sys.stdin)['UNETLoader']['input']['required']['unet_name'][0])"
```

探针出现在列表里即目录指对；GGUF 探针用 `/object_info/UnetLoaderGGUF`。验证完删掉探针。

## 改了页面/模板，浏览器还是旧样子

Flask 非 debug 模式模板有缓存，`templates/*.html` 改动必须重启面板进程（`systemctl restart h3-panel` 或重启启动它的终端进程）。改后端 Python 同理。血泪教训：改三次 scp 三次忘重启，浏览器一直拿旧 JS，排查半天其实是缓存。

## ComfyUI 连接失败 / 面板显示「未响应」

按顺序查：
1. `curl http://127.0.0.1:8188/system_stats` 通不通；
2. `systemctl status comfyui` 是否在崩溃循环（`activating (auto-restart)`）——看 `journalctl -u comfyui -n 40` 的最后一段 traceback。最常见两种：**No CUDA GPUs are available**（显卡被拔/驱动挂了，`nvidia-smi` 会同样报错，装回显卡 `systemctl start comfyui` 即恢复）；端口被占（老进程还没释放 8188，等几秒再启动，进程清显存需要几秒）。
3. 无显卡但要验证 ComfyUI 的节点注册（如 GGUF 装没装对）：`HOME=<用户目录> python3 main.py --listen 127.0.0.1 --port 8186 --cpu` 临时起一个 CPU 实例查 `/object_info`，查完杀掉。注意 HTTP 服务会先于节点注册完成就响应，要轮询目标节点出现再读值。

## pip 安装失败

- Debian 12 报 `externally-managed-environment`：加 `--break-system-packages`（用户目录布局配 `--user`）。
- 镜像选择：清华 403、阿里云 CDN 可能无限挂起，`-i https://pypi.org/simple` 慢但稳。
- 某些 NAS 布局 venv 是空壳、torch 实际装在 `HOME=xxx/.local`：统一用 `HOME=<用户目录> 系统python -m pip install --user ...` 的姿势，别用 venv 的 pip。

## git clone / 下载不通（国内网络）

- github 直连断：`git clone https://ghproxy.net/https://github.com/<repo>`。
- HuggingFace：主站不通时把域名换成 `hf-mirror.com`（路径不变），实测可跑满带宽。
- 面板模型库内置目录已按「hf-mirror 主源 + 官方源回退」写好，一般不需要手动处理。

## GGUF 组件装了但模型列表为空 / 节点不存在

- `/object_info` 里查不到 `UnetLoaderGGUF`：custom_nodes 没装对或 ComfyUI 没重启。
- 节点在但 `unet_name` 选择列表为空：确认 `.gguf` 文件放在 ComfyUI 扫描的 `diffusion_models`（或 `unet`）目录本身，参考上面的软链坑。
- `gguf` 包必须装进 ComfyUI 实际使用的 python 环境；Debian 12 记得 `--break-system-packages`。

## AI 增强（插帧/超分）报引擎未就绪或失败

- 引擎文件缺失按提示在 postproc 目录跑 `install.sh`。
- 报 CUDA 错误：后处理用的 `python` 配置项必须与 ComfyUI 同一套 torch CUDA 环境；纯 CPU 环境跑不了。
- 插帧/超分也要显存，面板会在不足时先让 ComfyUI 卸载闲置模型，仍不足会拒绝任务。

## 面板有访问口令时，脚本怎么调 API

先 `POST /api/login {"password": "..."}` 拿 token，之后请求带 `X-Panel-Token: <token>` 头；或直接带 cookie `h3auth=<token>`。浏览器端首次访问弹登录框，输一次 30 天有效。

## 长片接龙/批量中途失败

- 单任务失败（OOM 等）只影响那一条，批次会继续；失败原因在批次进度面板的失败列表里。
- ComfyUI 掉线批次会在 60s×30 次重试后终止；修好 ComfyUI 重新提交即可，批量状态持久化在 `data/batch.json`，面板重启会自动续跑。
- 接龙某段报「尾帧提取失败」：确认 ffmpeg 可用且成品视频真实存在（`output_dir` 下有对应 mp4）。
