#!/usr/bin/env python3
"""H3 面板后处理 CLI：RIFE 插帧 ×2 / Real-ESRGAN 超分 ×2
用法: postproc_cli.py <rife|x2> <in.mp4> <out.mp4>
成功时输出一行 JSON: {"ok":true,"out":...,"frames_in":N,"frames_out":M,"seconds":S}
依赖: venv 的 torch(cuda) + /usr/bin/ffmpeg；RIFE 权重 ckpts/rife47.pth；超分权重 ckpts/x2plus.pth
RIFE 网络定义来自 ComfyUI-Frame-Interpolation（stub 掉 comfy 依赖）；
RRDBNet 网络定义内嵌自 basicsr(MIT)，避免安装 basicsr 全家桶。"""
import glob, json, os, shutil, subprocess, sys, tempfile, time, types

BASE = os.path.dirname(os.path.abspath(__file__))
FI_DIR = os.path.join(BASE, "Frame-Interpolation")
CKPT_DIR = os.path.join(BASE, "ckpts")
FFMPEG = os.environ.get("FFMPEG") or shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = os.environ.get("FFPROBE") or shutil.which("ffprobe") or "ffprobe"
DEVICE = "cuda"


def sh(args, timeout=1800):
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"cmd failed: {' '.join(args[:3])}...: {r.stderr[-400:]}")
    return r.stdout


def probe_fps(path):
    out = sh([FFPROBE, "-v", "error", "-select_streams", "v:0",
              "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", path]).strip()
    num, _, den = out.partition("/")
    fps = float(num) / float(den or 1)
    return round(fps, 3)


def probe_dim(path):
    out = sh([FFPROBE, "-v", "error", "-select_streams", "v:0",
              "-show_entries", "stream=width,height", "-of", "csv=p=0", path]).strip()
    w, h = out.split(",")
    return int(w), int(h)


def extract_frames(mp4, d):
    sh([FFMPEG, "-y", "-loglevel", "error", "-i", mp4, "-vsync", "0", os.path.join(d, "%05d.png")])
    frames = sorted(glob.glob(os.path.join(d, "*.png")))
    if not frames:
        raise RuntimeError("抽帧失败：无输出")
    return frames


def assemble(frames_dir, fps, src, out, half=False):
    args = [FFMPEG, "-y", "-loglevel", "error", "-framerate", f"{fps:.4f}",
            "-i", os.path.join(frames_dir, "%05d.png")]
    if not half:
        args += ["-i", src, "-map", "0:v", "-map", "1:a?"]
    args += ["-c:v", "libx264", "-crf", "18", "-preset", "veryfast", "-pix_fmt", "yuv420p"]
    if not half:
        args += ["-c:a", "copy"]
    args += [out]
    sh(args)


# ---------------- RIFE ----------------
def run_rife(mp4, out):
    sys.path.insert(0, FI_DIR)
    torch = __import__("torch")
    fake = types.ModuleType("comfy")
    fake_mm = types.ModuleType("comfy.model_management")
    fake_mm.get_torch_device = lambda: torch.device(DEVICE)
    fake.__file__, fake_mm.__file__ = __file__, __file__

    def _no_attr(name):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *a, **k: None
    setattr(fake_mm, "__getattr__", _no_attr)
    setattr(fake, "__getattr__", lambda name: fake_mm if not name.startswith("__") else (_ for _ in ()).throw(AttributeError(name)))
    fake.model_management = fake_mm
    sys.modules["comfy"] = fake
    sys.modules["comfy.model_management"] = fake_mm
    # 直接按文件加载 rife_arch，绕开会引入 vfi_utils/torchvision 的包 __init__
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "rife_arch", os.path.join(FI_DIR, "vfi_models", "rife", "rife_arch.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rife_arch"] = mod
    spec.loader.exec_module(mod)
    IFNet = mod.IFNet

    model = IFNet(arch_ver="4.7").to(DEVICE).eval()
    sd = torch.load(os.path.join(CKPT_DIR, "rife47.pth"), map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd)

    tmp = tempfile.mkdtemp(prefix="rife_", dir=os.path.join(BASE, "tmp"))
    try:
        frames = extract_frames(mp4, tmp)
        fps = probe_fps(mp4)
        imgs = []
        from PIL import Image
        import numpy as np
        for f in frames:
            a = np.asarray(Image.open(f).convert("RGB")).astype(np.float32) / 255.0
            imgs.append(torch.from_numpy(a).permute(2, 0, 1))
        mids = []
        t0 = time.time()
        B = 12
        pairs_total = len(imgs) - 1
        with torch.no_grad():
            for i in range(0, pairs_total, B):
                b = min(B, pairs_total - i)
                i0 = torch.stack(imgs[i:i + b]).to(DEVICE)
                i1 = torch.stack(imgs[i + 1:i + 1 + b]).to(DEVICE)
                ts = torch.full((b, 1, 1, 1), 0.5, device=DEVICE)
                merged = model(i0, i1, ts, training=False, fastmode=True, ensemble=False)
                mids.append(merged.clamp(0, 1).cpu())
        mid_t = torch.cat(mids)
        out_dir = os.path.join(tmp, "out")
        os.makedirs(out_dir)
        # 交错: f0 m0 f1 m1 ... f(n-1)
        idx = 1
        from PIL import Image as I2
        for i in range(len(imgs)):
            I2.fromarray((imgs[i].permute(1, 2, 0).numpy() * 255).round().astype("uint8")).save(
                os.path.join(out_dir, f"{idx:05d}.png")); idx += 1
            if i < len(mid_t):
                I2.fromarray((mid_t[i].permute(1, 2, 0).numpy() * 255).round().astype("uint8")).save(
                    os.path.join(out_dir, f"{idx:05d}.png")); idx += 1
        assemble(out_dir, fps * 2, mp4, out)
        return {"frames_in": len(imgs), "frames_out": idx - 1, "proc_seconds": round(time.time() - t0)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------- Real-ESRGAN x2（内嵌 RRDBNet，MIT/basicsr） ----------------
def run_x2(mp4, out):
    import torch
    from torch import nn as nn
    import torch.nn.functional as F

    class ResidualDenseBlock(nn.Module):
        def __init__(self, num_feat=64, num_grow_ch=32):
            super().__init__()
            self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
            self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, num_feat, num_grow_ch=32):
            super().__init__()
            self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

        def forward(self, x):
            out = self.rdb1(x)
            out = self.rdb2(out)
            out = self.rdb3(out)
            return out * 0.2 + x

    class RRDBNet(nn.Module):
        """Real-ESRGAN x2plus 权重结构：pixel_unshuffle(2) 进(12ch) → 网络内两次 2x 上采样 → conv_last 直接出 3 通道（净 2x）"""

        def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
            super().__init__()
            self.scale = scale
            if scale == 2:
                num_in_ch = num_in_ch * 4
            elif scale == 1:
                num_in_ch = num_in_ch * 16
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
            self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            if self.scale == 2:
                feat = F.pixel_unshuffle(x, 2)
            elif self.scale == 1:
                feat = F.pixel_unshuffle(x, 4)
            else:
                feat = x
            feat = self.lrelu(self.conv_first(feat))
            feat = feat + self.conv_body(self.body(feat))
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
            return self.conv_last(self.lrelu(self.conv_hr(feat)))

    model = RRDBNet(scale=2).to(DEVICE).eval()
    sd = torch.load(os.path.join(CKPT_DIR, "x2plus.pth"), map_location="cpu", weights_only=False)
    sd = sd.get("params_ema") or sd.get("params") or sd
    model.load_state_dict(sd)

    tmp = tempfile.mkdtemp(prefix="x2_", dir=os.path.join(BASE, "tmp"))
    try:
        frames = extract_frames(mp4, tmp)
        fps = probe_fps(mp4)
        from PIL import Image
        import numpy as np
        t0 = time.time()
        out_dir = os.path.join(tmp, "out")
        os.makedirs(out_dir)
        with torch.no_grad():
            for i, f in enumerate(frames, 1):
                a = np.asarray(Image.open(f).convert("RGB")).astype(np.float32) / 255.0
                t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
                r = model(t).clamp(0, 1)[0].cpu()
                Image.fromarray((r.permute(1, 2, 0).numpy() * 255).round().astype("uint8")).save(
                    os.path.join(out_dir, f"{i:05d}.png"))
        assemble(out_dir, fps, mp4, out)
        return {"frames_in": len(frames), "proc_seconds": round(time.time() - t0)}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _precheck(mode):
    """引擎文件自检，缺失时输出干净 JSON 错误（供面板透传）"""
    problems = []
    if not os.path.isdir(FI_DIR):
        problems.append("Frame-Interpolation 未安装（在 postproc 目录运行 install.sh）")
    if mode in ("rife", "rife_x2") and not os.path.isfile(os.path.join(CKPT_DIR, "rife47.pth")):
        problems.append("rife47.pth 缺失（install.sh 自动下载）")
    if mode in ("x2", "rife_x2") and not os.path.isfile(os.path.join(CKPT_DIR, "x2plus.pth")):
        problems.append("x2plus.pth 缺失（install.sh 自动下载）")
    if problems:
        print(json.dumps({"ok": False, "error": "；".join(problems)}, ensure_ascii=False))
        sys.exit(1)


def main():
    mode, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
    _precheck(mode)
    os.makedirs(os.path.join(BASE, "tmp"), exist_ok=True)
    t0 = time.time()
    if mode == "rife":
        info = run_rife(src, out)
    elif mode == "x2":
        info = run_x2(src, out)
    else:
        raise SystemExit("mode 必须是 rife 或 x2")
    info.update({"ok": True, "out": out, "total_seconds": round(time.time() - t0)})
    print(json.dumps(info, ensure_ascii=False))


if __name__ == "__main__":
    main()
