# UGOS Pro UPK 安装包（minimax-h3-panel）

把 H3 视频生成工作台（面板 + ComfyUI 一体容器）打包为绿联 UGOS Pro 可侧载安装的 `.upk` 包。

## 前置条件（重要）

1. **NVIDIA 显卡**：NAS 需已安装 NVIDIA 显卡（建议 8GB+ 显存）并装好驱动
   （UGOS Pro 应用中心 / 官方驱动包提供；30 系已验证可用，新卡以驱动包覆盖范围为准）
2. **模型不在包内**：主模型 + 文本编码器 + VAE 约 40GB 起，安装后首次打开面板，
   在「模型库」页一键下载（国内走 hf-mirror，实测 90MB/s）
3. **侧载需要开发者授权**：测试机需按 UGOS 开发者规范注册设备
   （序列号/MAC/管理员用户名 → 授权文件 `ugdev.sig` 放管理员用户家目录）

## 安装（侧载）

1. UGOS Pro → 应用中心 → 右上角菜单 → 手动安装（离线安装）→ 上传 `.upk` 文件
2. 安装向导选择两个目录：**模型目录**（建议 200GB+ 空间的新建空目录）与**成片输出目录**
3. 安装完成后从应用中心打开（`http://NAS:8189`）→「模型库」下载模型 → 创作

## GPU 说明（重要，实验性）

- UPK schema 官方无 GPU 字段；包内 compose 使用标准 `devices:` 直通 + 宿主驱动库只读挂载
  （`/usr/lib/x86_64-linux-gnu → /hostlibs`，entrypoint 自动发现并注入 LD_LIBRARY_PATH）
- 该声明**未经 UGOS 官方文档化**：在部分固件上可能被安装链路剥除
- 判别方法：安装后 `SSH 到 NAS → docker exec <容器> nvidia-smi`——有输出即直通成功；
  无输出则容器降级为 CPU 模式（面板可用、模型库可用，但生成不可用）
- 降级补救（SSH）：
  ```bash
  # 找到容器后补 GPU 映射（docker 无热插拔，需重建容器，卷与配置保留）
  docker inspect <容器>   # 抄下 Mounts/Env
  docker rm -f <容器>
  # 按根目录 docker-compose.selfhost.yml 重建（devices/runtime 完整无限制）
  ```

## 打包流程（维护者）

前置：镜像已推送到 Docker Hub（`jvsheng/minimax-h3-panel:upk-v1-YYYYMMDD`），
打包机 = 690 NAS（5.40，SSH 有防爆破锁——批量化操作、ControlMaster 复用），
用 [ugos-upk-pack](https://github.com/hongjiahao371-pixel/minimax-h3-panel) skill 的标准流程：

```bash
# 1. Mac 上归档镜像（no_proxy 必须；脚本内置断点续传）
no_proxy='*' python3 ~/.agents/skills/ugos-upk-pack/scripts/docker_archive_from_registry.py \
  --image jvsheng/minimax-h3-panel:upk-v1-YYYYMMDD --arch amd64 \
  --output /tmp/upk-build/h3panel-amd64.tar

# 2. 传输
git archive HEAD | gzip > /tmp/upk-src.tar.gz
sshpass -p '<690密码>' scp -O /tmp/upk-src.tar.gz /tmp/upk-build/h3panel-amd64.tar 690@192.168.5.40:~/

# 3. NAS 上解包 src → 组装树 → append_image_layer（--file 清单覆盖 panel/ 全部文件）→ ugcli check → pack --arch amd64 --build N
# 4. 取回 .upk 到桌面，删除旧 build
```

- `--file` 清单：`panel/panel.py`、`panel/templates/index.html`、`panel/models_catalog.json`
  （新增面板文件必须同步加入清单，否则热补丁层不带新文件）
- compose 内镜像 tag 与 `--tag` 一致且**不用 latest**
- build 号只增不减

## 包体说明

- 镜像约 10~15G（torch cu128 + ComfyUI 0.32 + 面板 + ffmpeg + RIFE/超分引擎），
  `.upk` 体积同级——**只适合侧载，不适合提交应用商店**
- 只支持 amd64（cu128 torch 无 arm64 构建）
- 模型目录卷结构（安装后自动创建）：`diffusion_models/ loras/ text_encoders/ vae/`

## 回退路线

UPK 若因固件差异无法 GPU 直通，改用仓库根 `deploy/docker/docker-compose.selfhost.yml`
直接在 UGOS Pro 的 Docker 界面导入部署——无 UPK 包体/字段限制，GPU 完全可控。
