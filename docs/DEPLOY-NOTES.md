# 部署实录与踩坑史（作者机器）

> 本文档是作者在绿联 IDX6011PRO NAS（Debian 12 / Ultra 7 255H / RTX 3080 12GB）上的部署实录，
> 供遇到类似环境问题时参考。通用安装请看主 README。

## 作者实例的环境布局

- ComfyUI 0.32.0 原生安装于 `/volume2/comfyui/ComfyUI`，模型在 `/volume2/comfyui/models`
  （H3 剪枝 INT8 DiT 20G + Qwen3VL-32B 文本编码器 15G + 视频/音频 VAE）
- H3 权重与节点包来自 MiniMax 官方发布（UNET: minimax_h3_fl2va_pruned_int8_convrot；
  文本编码器: qwen3vl_32b_minimax_h3_nvfp4_awq）
- python 依赖全部装在 `HOME=/volume2/comfyui/home` 的用户目录（python3.11，torch 2.11.0+cu128），
  **注意：目录里的 venv/ 是空壳**，正确用法是 `export HOME=... && 系统 python3`
- 面板以 systemd（h3-panel.service）常驻，ComfyUI 以 comfyui.service 常驻
- TeaCache 加速：custom_nodes/ComfyUI-MiniMaxH3-Cache 是作者针对 ComfyUI 0.32.0 重写的整模型级
  TeaCache 实现（官方版本按新内核 API 写、在 0.32.0 会崩），升级 ComfyUI 时注意别覆盖

## v5.0 模型库相关

- **模型目录的真实布局（重要）**：`ComfyUI/models/diffusion_models/` 是真实目录，里面是
  指向 `/volume2/comfyui/models/diffusion_models/` 权重仓的**逐文件符号链接**（8/12 建环境时手工链的）。
  直接往权重仓放新文件 ComfyUI 看不到——必须放进 ComfyUI 扫描目录本身。面板 config 的
  `comfy_models_dir` 配的就是 `/volume2/comfyui/ComfyUI/models/diffusion_models`，
  下载器直接把文件下到这里（.part 临时名 → 完成后同目录原子改名）
- **GGUF 支持**：`bash deploy/install_gguf_node.sh` 一键装（city96/ComfyUI-GGUF 经 ghproxy clone +
  pypi 装 gguf 包到用户目录）。已实测与 ComfyUI 0.32.0 兼容；它的 `unet_gguf` 键映射到
  diffusion_models + unet 两个目录，.gguf 与 .safetensors 同目录混放即可
- **GGUF 与 TeaCache**：MiniMaxH3Cache 会 clone 模型并断言 `MiniMaxH3Model` 类。GGUF 加载后能否
  过断言需实测；不行就在 models_catalog.json 对应条目加 `"turbo_compat": false`（面板会自动跳过缓存节点）
- **加速折叠版权重**（FastH3 4 步 / PDD 8 步等）：步数是固定的，且不能再叠 TeaCache。
  目录条目用 `"steps_fixed": N` + `"turbo_compat": false` 表达，面板自动强制
- 新增目录条目：编辑 panel/models_catalog.json（filename/size_gb/urls 必填，urls 按顺序回退），
  重启面板生效；未登记的本地权重文件也会被扫描出来、以文件名兜底显示
- **v7.0 Turbo LoRA**：LoRA 下载目标目录 = `comfy_models_dir` 的兄弟目录 `loras/`
  （/volume2/comfyui/ComfyUI/models/loras，LoraLoaderModelOnly 从这里列文件）。
  官方蒸馏 LoRA 与 TeaCache 互斥（选 LoRA 时工作流不挂节点 5）、步数锁定 4/8；
  LoRA 对 int8_convrot 量化底模的兼容性待显卡回装实跑验证。批量首帧图用
  `_upload_batch_image` 按批次画幅 cover 裁剪上传（全批同尺寸→合集可无损 copy 拼接）；
  图片在批次受理时一次性上传（base64 不进 batch.json，避免每次落盘放大文件）。
  清理策略硬编码：保留星标 + 最近 7 天（CLEANUP_KEEP_DAYS）
- **v7.1 长文拆分**：`/api/split_prompts` 复用提示词优化的 llm_config（高级选项里那组 API Base/Key）；
  LLM 按分镜拆（system prompt 要求逐行输出），解析时剥编号/引号，<2 条视为失败回落；
  无 LLM 用规则兜底（按 。！？；换行 切句 → 贪心长度均衡分组），默认段数 = 长度/80 夹在 2~8。
  前端拆分结果可编辑后替换/追加到批量列表
- 2026-09-18 时点：**3080 显卡未装回**，comfyui.service 处于 CUDA 缺失崩溃循环（已手动 stop，
  enabled 保留——装回显卡后 `systemctl start comfyui` 即恢复）；生成类端到端验证待显卡回装后补
- **v6.0 通宵批量**：批次状态持久化在 `panel/data/batch.json`（每任务落盘一次）；面板重启时
  `_batch_restore()` 续跑剩余队列，重启前的在途任务由新进程先收尾入库（元数据 prompt 标
  「重启前提交」）——所以批量视频的元数据不依赖浏览器页面。单任务等待上限
  max(1800s, 时长×15)；ComfyUI 提交连接失败按 60s×30 重试后整批终止。批量任务不写 TASKS，
  入库完全由编排器负责（与单发任务的 TASKS watcher 是两条路）。本地验证用 mock ComfyUI
  （Flask 假 /prompt + /history，注入失败/延迟）覆盖 顺序执行/失败跳过/取消/重启续跑 全场景

## 本机 config.json（对应作者环境）

```json
{
  "comfy_url": "http://127.0.0.1:8188",
  "port": 8189,
  "output_dir": "/volume2/comfyui/ComfyUI/output",
  "data_dir": "/volume2/comfyui/panel/data",
  "postproc_dir": "/volume2/comfyui/postproc",
  "ffmpeg": "/usr/bin/ffmpeg",
  "ffprobe": "/usr/bin/ffprobe",
  "python": "/usr/bin/python3",
  "postproc_home": "/volume2/comfyui/home",
  "vram_budget_tokens": 115000000,
  "vram_warn_tokens": 95000000,
  "access_password": ""
}
```

## 踩坑史（最有价值部分）

**ComfyUI / H3 侧**
- H3 无 latent previewer，不发 progress 事件——面板用 executing 节点事件映射阶段进度
- ComfyUI 重启后要等 8188 端口真正释放再启动（大模型内存清理需要几秒）
- 官方 ComfyUI-MiniMaxH3-Cache 节点按新版内核 API 写，在 ComfyUI 0.32.0 必崩，需整模型级重写
- history 的错误详情在 `item["status"]["messages"]` 的 `[event,data]` 二元组里，不在顶层
- 显存实测：864×480×362帧(150M token) OOM；×260帧(108M) 成功约 8 分钟 → 预算定为 115M
- sageattention 在只读 rootfs 上装不了（kernel 编译需要 python-dev）

**面板开发侧**
- Flask 非 debug 模式改 templates 必须重启服务（模板缓存不 auto_reload）
- 模块顶层调用函数必须放定义之后（py_compile 只查语法不查名字解析）
- 拼装页面时 `<style>` 版本必须与功能同步替换（曾两次漏拼导致新功能裸奔）
- stub 外部模块（如 comfy）时万能 `__getattr__` 会吃掉 `__file__`，
  导致 torchvision 的 inspect 崩溃——应按文件路径直接加载目标模块绕过包 __init__ 链
- RealESRGAN x2plus 权重结构 = pixel_unshuffle(2) 进 12 通道 + 两次 2x 上采样 + conv_last 直出 3 通道
  （basicsr 与 Real-ESRGAN 各自 fork 的 RRDBNet 构造不同，以权重 keys 为准）
- venv 可能是空壳：torch 在 HOME 指向的 .local 里，用 `export HOME=... && python3` 而不是 venv/bin/python
- 后台 watcher 收尾任务与页面轮询并发收尾时，用「pop 返回 None 则跳过」的幂等设计避免重复入库

**网络备忘（国内 NAS）**
- pip 官源可用且快；清华镜像 403、部分阿里云 CDN 挂起；pytorch 官源 ~72MB/s 最快
- github 直连不稳，走 ghproxy.net；huggingface 走 hf-mirror.com

**UGREEN NAS 专属**
- sshd 有防爆破：频繁断连重连会临时锁密码 30-60s，务必 SSH ControlMaster 复用连接
- scp 需要 `-O` 传统协议
- /volume1 默认 ACL 会把新文件压成 600，rsync 后需要 `chmod -R a+rX`
