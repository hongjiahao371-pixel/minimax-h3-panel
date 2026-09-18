#!/usr/bin/env python3
"""MiniMax-H3 生成面板 v2：简易 API + Web UI，代理到本机 ComfyUI(8188)。
支持：文/图生视频、多画幅、批量连发、取消、历史元数据（复用/删除）、加速开关"""
import base64, glob, io, json, os, random, shutil, subprocess, sys, tempfile, threading, time
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, render_template
import requests

import hashlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_panel_config():
    """读取面板配置（panel/config.json，可被 PANEL_* 环境变量覆盖单项）"""
    cfg_path = os.environ.get("PANEL_CONFIG") or os.path.join(BASE_DIR, "config.json")
    cfg = {}
    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        print("[panel] 未找到 config.json，使用默认值（参见 config.example.json）")

    def pick(key, default):
        if key in cfg:
            return cfg[key]
        return os.environ.get("PANEL_" + key.upper(), default)

    def _abs(p):
        if not p:
            return p
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(BASE_DIR, p))

    data_dir = _abs(pick("data_dir", os.path.join(BASE_DIR, "data")))
    os.makedirs(data_dir, exist_ok=True)
    return {
        "comfy_url": pick("comfy_url", "http://127.0.0.1:8188"),
        "port": int(pick("port", 8189)),
        "output_dir": _abs(pick("output_dir", "")) or os.path.join(data_dir, "output"),
        "data_dir": data_dir,
        "postproc_dir": _abs(pick("postproc_dir", os.path.join(BASE_DIR, "..", "postproc"))),
        "ffmpeg": pick("ffmpeg", "ffmpeg"),
        "ffprobe": pick("ffprobe", "ffprobe"),
        "python": pick("python", sys.executable or "python3"),
        "postproc_home": pick("postproc_home", ""),
        "vram_budget_tokens": int(pick("vram_budget_tokens", 115_000_000)),
        "vram_warn_tokens": int(pick("vram_warn_tokens", 95_000_000)),
        "access_password": str(pick("access_password", "") or ""),
        "comfy_models_dir": _abs(pick("comfy_models_dir", "")),
        "default_model": str(pick("default_model", "") or ""),
        "os.makedirs": data_dir,
    }


CONFIG = _load_panel_config()
COMFY = CONFIG["comfy_url"]
OUTPUT = CONFIG["output_dir"]
META_FILE = os.path.join(CONFIG["data_dir"], "meta.json")
FFMPEG = CONFIG["ffmpeg"]
FFPROBE = CONFIG["ffprobe"]
# 显存预算：w*h*length 的 token 积，超过则拒绝（force=True 可强行尝试）
TOKEN_BUDGET = CONFIG["vram_budget_tokens"]
TOKEN_WARN = CONFIG["vram_warn_tokens"]
UNET = "minimax_h3_fl2va_pruned_int8_convrot.safetensors"
CLIP = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
VAE_VID = "minimax_h3_video_vae_fp16.safetensors"
VAE_AUD = "minimax_h3_audio_vae_fp32.safetensors"

# ---------------- 模型库：社区衍生模型注册表 / 下载器 ----------------
MODEL_EXTS = (".safetensors", ".gguf", ".sft")
# ComfyUI 实际扫描的权重目录（UNETLoader / UnetLoaderGGUF 都从这里列文件）
MODELS_DIR = CONFIG.get("comfy_models_dir") or os.path.join(
    os.path.dirname(OUTPUT.rstrip(os.sep)), "models", "diffusion_models")
DEFAULT_MODEL = CONFIG.get("default_model") or UNET
CATALOG_FILE = os.path.join(BASE_DIR, "models_catalog.json")
DL_FILE = os.path.join(CONFIG["data_dir"], "downloads.json")
PANEL_CONFIG_PATH = os.environ.get("PANEL_CONFIG") or os.path.join(BASE_DIR, "config.json")


def _load_catalog():
    try:
        with open(CATALOG_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return {m["filename"]: m for m in data.get("models", []) if m.get("filename")}
    except Exception:
        return {}


CATALOG = _load_catalog()

app = Flask(__name__)

# ---------------- 元数据存储 ----------------
def _load_meta():
    try:
        with open(META_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"videos": {}}

META = _load_meta()
TASKS_FILE = os.path.join(CONFIG["data_dir"], "tasks.json")
LOCK = threading.Lock()

def _load_tasks():
    try:
        with open(TASKS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

TASKS = _load_tasks()  # task_id -> 元数据（持久化到 tasks.json，面板重启后仍可跟踪）

def _save_tasks():
    """写盘未完成任务（调用方需已持有 LOCK）；顺带清理 6 小时前的陈旧条目"""
    cutoff = time.time() - 6 * 3600
    for tid in [k for k, v in TASKS.items() if v.get("t0", 0) < cutoff]:
        TASKS.pop(tid, None)
    tmp = TASKS_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(TASKS, f, ensure_ascii=False)
    os.replace(tmp, TASKS_FILE)

def _meta_from_history(item):
    """面板重启后 TASKS 丢失：从 ComfyUI 历史内嵌的工作流图恢复完整元数据"""
    meta = {"prompt": None, "seed": None, "width": None, "height": None, "length": None,
            "steps": None, "turbo": False, "i2v": False, "end_frame": False,
            "sampler": None, "scheduler": None, "crop": None, "time": int(time.time())}
    try:
        graph = item["prompt"][2]
        n6 = graph.get("6", {}).get("inputs", {})
        n8 = graph.get("8", {}).get("inputs", {})
        n1 = graph.get("1", {}).get("inputs", {})
        unet_fn = n1.get("unet_name") or UNET
        meta.update({"prompt": n6.get("prompt"), "width": n6.get("width"),
                     "height": n6.get("height"), "length": n6.get("length"),
                     "seed": n8.get("seed"), "steps": n8.get("steps"),
                     "sampler": n8.get("sampler_name"), "scheduler": n8.get("scheduler"),
                     "turbo": "5" in graph, "i2v": "14" in graph, "end_frame": "15" in graph,
                     "model": unet_fn, "model_label": (CATALOG.get(unet_fn) or {}).get("label") or unet_fn})
    except Exception:
        pass
    return meta

def _gen_seconds(item, t0=None):
    """真实耗时：优先 ComfyUI 执行事件时间戳，退化为提交时刻差"""
    try:
        ts = {}
        for m in ((item.get("status") or {}).get("messages") or []):
            if isinstance(m, list) and len(m) == 2 and m[0] in ("execution_start", "execution_success", "execution_error"):
                ts[m[0]] = (m[1] or {}).get("timestamp")
        if ts.get("execution_start") and (ts.get("execution_success") or ts.get("execution_error")):
            end = ts.get("execution_success") or ts.get("execution_error")
            return max(1, round((end - ts["execution_start"]) / 1000))
    except Exception:
        pass
    return round(time.time() - t0) if t0 else None

def _record_failure(meta, err):
    """失败样本入库（含 token 规模），供 /api/stats 做 OOM 历史预警"""
    try:
        with LOCK:
            fl = META.setdefault("failures", [])
            fl.append({"w": meta.get("width"), "h": meta.get("height"),
                       "length": meta.get("length"),
                       "tokens": (meta.get("width") or 0) * (meta.get("height") or 0) * (meta.get("length") or 0),
                       "error": (err or "")[:160], "time": int(time.time())})
            del fl[:-200]
            _save_meta()
    except Exception:
        pass

def _save_meta():
    tmp = META_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(META, f, ensure_ascii=False, indent=1)
    os.replace(tmp, META_FILE)

def _probe_duration(path):
    try:
        out = subprocess.run(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", path], capture_output=True, text=True, timeout=10)
        return round(float(out.stdout.strip()))
    except Exception:
        return None


def fit_dims(iw, ih, max_long=864):
    """按图片比例计算视频分辨率（长边 max_long、32 对齐、限制在模型范围）"""
    if iw >= ih:
        w, h = max_long, int(ih * max_long / iw)
    else:
        h, w = max_long, int(iw * max_long / ih)
    w = max(256, min(1344, w // 32 * 32))
    h = max(256, min(1344, h // 32 * 32))
    if w < 256 or h < 256:
        w, h = 864, 480
    return w, h


def cover_resize(img, w, h, crop_pos="center"):
    """等比缩放 + 裁剪到 (w,h)，不拉伸；crop_pos: top/center/bottom"""
    iw, ih = img.size
    scale = max(w / iw, h / ih)
    nw, nh = round(iw * scale), round(ih * scale)
    img = img.resize((nw, nh), Image.LANCZOS)
    l = (nw - w) // 2
    if crop_pos == "top":
        t = 0
    elif crop_pos == "bottom":
        t = nh - h
    else:
        t = (nh - h) // 2
    return img.crop((l, t, l + w, t + h))


def _unet_loader_node(model_profile):
    """模型加载节点：safetensors 走 UNETLoader，GGUF 走 UnetLoaderGGUF（节点 id 恒为 "1"）"""
    mp = model_profile or {}
    fn = mp.get("filename") or UNET
    if mp.get("kind") == "gguf":
        return {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": fn}}
    return {"class_type": "UNETLoader", "inputs": {"unet_name": fn, "weight_dtype": "default"}}


def build_workflow(prompt, width, height, length, seed, steps, image_name=None, end_image_name=None,
                   turbo=False, sampler="euler", scheduler="normal", model_profile=None):
    w = width // 32 * 32
    h = height // 32 * 32
    wf = {
        "1": _unet_loader_node(model_profile),
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP, "type": "minimax", "device": "cpu"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_VID}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": VAE_AUD}},
        "6": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {"clip": ["2", 0], "vae": ["3", 0], "prompt": prompt, "width": w, "height": h, "length": length}},
        "7": {"class_type": "MiniMaxH3SigmaShift", "inputs": {"model": (["5", 0] if turbo else ["1", 0]), "shift_video": 12.0, "shift_audio": 3.0}},
        "8": {"class_type": "KSampler", "inputs": {"model": ["7", 0], "seed": seed, "steps": steps, "cfg": 1.0, "sampler_name": sampler, "scheduler": scheduler, "positive": ["6", 0], "negative": ["6", 0], "latent_image": ["6", 1], "denoise": 1.0}},
        "9": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["8", 0]}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["9", 0], "vae": ["3", 0]}},
        "11": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["9", 1], "vae": ["4", 0]}},
        "12": {"class_type": "CreateVideo", "inputs": {"images": ["10", 0], "fps": 24.0, "audio": ["11", 0]}},
        "13": {"class_type": "SaveVideo", "inputs": {"video": ["12", 0], "filename_prefix": "h3_video", "format": "auto", "codec": "auto"}},
    }
    if turbo:
        wf["5"] = {"class_type": "MiniMaxH3Cache", "inputs": {"model": ["1", 0],
            "resuse_threshold": 0.10, "start_percent": 0.15, "end_percent": 0.9,
            "max_steps": 2, "device": "auto", "verbose": False}}
    if image_name:
        wf["14"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
        wf["6"]["inputs"]["first_frame"] = ["14", 0]
    if end_image_name:
        wf["15"] = {"class_type": "LoadImage", "inputs": {"image": end_image_name}}
        wf["6"]["inputs"]["last_frame"] = ["15", 0]
    return wf


def _find_mp4(outputs):
    for out in outputs.values():
        for k in ("gifs", "videos", "files", "images"):
            for f in out.get(k, []):
                fn = f.get("filename", "")
                if fn.endswith(".mp4"):
                    return fn
    return None


# ---------------- 访问口令（config.access_password 非空时启用） ----------------
AUTH_TOKEN = hashlib.sha256(CONFIG["access_password"].encode()).hexdigest() if CONFIG["access_password"] else ""


@app.before_request
def _auth_guard():
    if not AUTH_TOKEN:
        return None
    p = request.path
    if p == "/" or p == "/api/login":
        return None
    if not (p.startswith("/api/") or p.startswith("/video/")):
        return None
    if (request.headers.get("X-Panel-Token") == AUTH_TOKEN
            or request.cookies.get("h3auth") == AUTH_TOKEN
            or request.args.get("token") == AUTH_TOKEN):
        return None
    return jsonify({"ok": False, "auth": True, "error": "需要访问口令"}), 401


@app.route("/api/login", methods=["POST"])
def login():
    pw = (request.json or {}).get("password") or ""
    if AUTH_TOKEN and hashlib.sha256(pw.encode()).hexdigest() == AUTH_TOKEN:
        resp = jsonify({"ok": True, "token": AUTH_TOKEN})
        resp.set_cookie("h3auth", AUTH_TOKEN, max_age=30 * 86400, samesite="Lax")
        return resp
    return jsonify({"ok": False, "error": "口令错误"}), 401


@app.after_request
def _no_cache_html(response):
    if response.content_type and "text/html" in response.content_type:
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def status():
    try:
        d = requests.get(f"{COMFY}/system_stats", timeout=5).json()
        gpu = d["devices"][0]
        q = requests.get(f"{COMFY}/queue", timeout=5).json()
        vt = gpu.get("vram_total", 0)
        vf = gpu.get("vram_free", vt)
        sysinfo = d.get("system", {})
        rt, rf = sysinfo.get("ram_total", 0), sysinfo.get("ram_free", 0)
        du = shutil.disk_usage(OUTPUT)
        return jsonify({"ok": True, "gpu": gpu.get("name", "GPU"),
                        "vram_gb": round(vt / 1e9, 1),
                        "vram_used_gb": round(max(0, vt - vf) / 1e9, 1),
                        "ram_used_gb": round(max(0, (rt - rf)) / 1e9, 1),
                        "ram_gb": round(rt / 1e9, 1),
                        "disk_free_gb": round(du.free / 1e9, 1),
                        "queued": len(q.get("queue_pending", [])),
                        "running": len(q.get("queue_running", []))})
    except Exception:
        return jsonify({"ok": False, "error": f"ComfyUI 未响应（{COMFY}）——请确认其正在运行且地址配置正确"})


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.json or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"ok": False, "error": "请输入提示词"})
    width = int(data.get("width", 864))
    height = int(data.get("height", 480))
    length = int(data.get("length", 124))
    seed = int(data.get("seed", 0) or 0)
    steps = max(4, min(20, int(data.get("steps", 8) or 8)))
    turbo = bool(data.get("turbo", True))
    count = max(1, min(4, int(data.get("count", 1) or 1)))
    steps_list = data.get("steps_list") or []
    if not isinstance(steps_list, list):
        steps_list = []
    steps_list = sorted({max(4, min(20, int(s))) for s in steps_list if str(s).strip().isdigit()}) or [steps]
    if count * len(steps_list) > 12:
        return jsonify({"ok": False, "error": f"矩阵任务数过大（{count}×{len(steps_list)}），上限 12 个"})
    force = bool(data.get("force", False))
    crop = data.get("crop") if data.get("crop") in ("top", "center", "bottom") else "center"
    SAMPLERS = {"euler", "euler_ancestral", "heun", "dpm_2", "lms", "uni_pc"}
    SCHEDS = {"normal", "simple", "sgm_uniform", "beta", "karras"}
    sampler = data.get("sampler") if data.get("sampler") in SAMPLERS else "euler"
    scheduler = data.get("scheduler") if data.get("scheduler") in SCHEDS else "normal"

    # 模型选择：默认用配置的默认模型；显式指定的必须在已安装列表里
    prof = _model_profile(data.get("model"))
    if data.get("model") and prof["filename"] != DEFAULT_MODEL and prof["filename"] not in _installed_models():
        return jsonify({"ok": False, "error": f"模型未安装：{prof['label']}"})
    if prof["needs_gguf"] and _gguf_ready() is not True:
        return jsonify({"ok": False, "error": "ComfyUI 未检出 GGUF 加载组件（UnetLoaderGGUF），无法使用 GGUF 模型；"
                                             "请先在 ComfyUI 安装 ComfyUI-GGUF 并重启"})

    def upload_b64_image(b64):
        img = Image.open(io.BytesIO(base64.b64decode(b64.split(",")[-1]))).convert("RGB")
        iw, ih = img.size
        uw, uh = fit_dims(iw, ih, max(width, height))
        img = cover_resize(img, uw, uh, crop)
        buf = io.BytesIO()
        img.save(buf, "PNG")
        r = requests.post(f"{COMFY}/upload/image",
                          files={"image": ("panel_i2v.png", buf.getvalue(), "image/png")}, timeout=30)
        if r.status_code != 200:
            return None, "图片上传失败"
        return r.json()["name"], None

    image_name = None
    end_image_name = None
    use_w, use_h = width, height
    if data.get("image_base64"):
        try:
            image_name, err = upload_b64_image(data["image_base64"])
            if err:
                return jsonify({"ok": False, "error": err})
            iw, ih = Image.open(io.BytesIO(base64.b64decode(data["image_base64"].split(",")[-1]))).size
            use_w, use_h = fit_dims(iw, ih, max(width, height))
        except Exception as e:
            return jsonify({"ok": False, "error": f"图片处理失败: {e}"})
    if data.get("end_image_base64"):
        try:
            end_image_name, err = upload_b64_image(data["end_image_base64"])
            if err:
                return jsonify({"ok": False, "error": err})
        except Exception as e:
            return jsonify({"ok": False, "error": f"尾帧处理失败: {e}"})

    tokens = use_w * use_h * length
    if tokens > TOKEN_BUDGET and not force:
        return jsonify({"ok": False, "over_budget": True,
                        "error": f"该组合（{use_w}×{use_h} · {length}帧≈{length//24}秒）超出本机显存预算，"
                                 f"生成会因 OOM 失败。建议：缩短时长，或降低画质档位，或减小画幅。"
                                 f"（token {tokens//10000}万 / 上限 {TOKEN_BUDGET//10000}万）"})

    tasks = []
    steps_list = list(steps_list)
    if prof.get("steps_fixed"):
        steps_list = [int(prof["steps_fixed"])]  # 加速版权重折叠了 LoRA，步数锁定
    turbo_eff = turbo and prof.get("turbo_compat", True)  # 与 TeaCache 互斥的权重自动关加速
    with LOCK:
        for stp in steps_list:
            for i in range(count):
                actual_seed = (seed + i) if seed else random.randint(1, 2**31 - 1)
                wf = build_workflow(prompt, use_w, use_h, length, actual_seed, stp,
                                    image_name, end_image_name, turbo_eff, sampler, scheduler, prof)
                try:
                    r = requests.post(f"{COMFY}/prompt", json={"prompt": wf, "client_id": "h3-panel"}, timeout=10)
                    res = r.json()
                except Exception:
                    return jsonify({"ok": False, "error":
                        f"无法连接 ComfyUI（{COMFY}）。请确认它正在运行，且 config.json 的 comfy_url 配置正确"})
                if "prompt_id" not in res:
                    return jsonify({"ok": False, "error": res.get("error", {}).get("message", "未知错误")})
                tid = res["prompt_id"]
                TASKS[tid] = {"prompt": prompt, "seed": actual_seed, "width": wf["6"]["inputs"]["width"],
                              "height": wf["6"]["inputs"]["height"], "length": length, "steps": stp,
                              "turbo": turbo_eff, "i2v": bool(image_name), "end_frame": bool(end_image_name),
                              "sampler": sampler, "scheduler": scheduler, "crop": crop,
                              "model": prof["filename"], "model_label": prof["label"],
                              "t0": time.time(), "time": int(time.time())}
                tasks.append({"task_id": tid, "seed": actual_seed, "steps": stp})
                threading.Thread(target=_watch_task, args=(tid,), daemon=True).start()
        _save_tasks()

    return jsonify({"ok": True, "tasks": tasks, "steps_list": steps_list,
                    "width": use_w if not image_name else TASKS[tasks[0]["task_id"]]["width"],
                    "height": use_h if not image_name else TASKS[tasks[0]["task_id"]]["height"],
                    "length": length, "count": count, "turbo": turbo_eff,
                    "model": prof["filename"], "model_label": prof["label"]})


def _finalize_task(tid, item):
    """任务收尾（入库/记失败/清理）：task() 与后台 watcher 共用。返回 None=仍在运行"""
    st = item.get("status", {})
    video = _find_mp4(item.get("outputs", {}))
    if not video:
        if st.get("status_str") == "error":
            err_msg = ""
            try:
                for m in ((item.get("status") or {}).get("messages") or []):
                    if isinstance(m, list) and len(m) == 2 and m[0] == "execution_error":
                        d = m[1] or {}
                        msg = d.get("exception_message") or "未知错误"
                        if "out of memory" in msg.lower():
                            msg = "显存不足（OOM）：请缩短时长或降低画质档位后重试"
                        err_msg = f"{d.get('node_type', '')}: {msg}"[:200]
                        break
            except Exception:
                pass
            with LOCK:
                meta = TASKS.pop(tid, None)
                _save_tasks()
            if meta:
                _record_failure(meta, err_msg or "任务执行失败")
            return {"status": "error", "error_msg": err_msg or "任务执行失败，详见 ComfyUI 日志"}
        if st.get("completed") or st.get("status_str") in ("success", "completed"):
            with LOCK:
                TASKS.pop(tid, None)
                _save_tasks()
            return {"status": "done"}
        return None
    with LOCK:
        meta = TASKS.pop(tid, None)
        _save_tasks()
        already = video in META["videos"]
    if meta is None and not already:
        meta = _meta_from_history(item)  # 面板重启丢失跟踪：从历史工作流恢复
    if meta is not None:
        meta.pop("t0", None)
        meta["duration"] = _probe_duration(os.path.join(OUTPUT, video))
        meta["gen_seconds"] = _gen_seconds(item, meta.get("t0"))
        with LOCK:
            META["videos"][video] = meta
            _save_meta()
    return {"status": "done", "video": video}


def _watch_task(tid):
    """后台跟踪任务直到完成并入库——不依赖浏览器页面开着"""
    deadline = time.time() + 7200
    while time.time() < deadline:
        time.sleep(6)
        try:
            h = requests.get(f"{COMFY}/history/{tid}", timeout=10).json()
        except Exception:
            continue
        item = h.get(tid)
        if item is None:
            with LOCK:
                known = tid in TASKS
            if not known:
                return  # 已被取消/移除
            continue
        if _finalize_task(tid, item) is not None:
            return


@app.route("/api/task/<task_id>")
def task(task_id):
    try:
        h = requests.get(f"{COMFY}/history/{task_id}", timeout=5).json()
    except Exception:
        return jsonify({"ok": True, "status": "queued"})
    item = h.get(task_id)
    if item is None:
        in_queue = False
        try:
            q = requests.get(f"{COMFY}/queue", timeout=5).json()
            for entry in q.get("queue_running", []):
                if len(entry) > 1 and entry[1] == task_id:
                    return jsonify({"ok": True, "status": "running"})
            for entry in q.get("queue_pending", []):
                if len(entry) > 1 and entry[1] == task_id:
                    in_queue = True
                    break
        except Exception:
            pass
        if in_queue:
            return jsonify({"ok": True, "status": "queued"})
        with LOCK:
            known = task_id in TASKS
        if not known:
            # ComfyUI 无记录且面板已不认识（被取消清空）
            return jsonify({"ok": True, "status": "cancelled"})
        return jsonify({"ok": True, "status": "queued"})
    res = _finalize_task(task_id, item)
    if res is None:
        return jsonify({"ok": True, "status": "running"})
    if res["status"] == "error":
        return jsonify({"ok": True, "status": "error", "error_msg": res["error_msg"]})
    return jsonify({"ok": True, "status": "done", "video": res.get("video")})


@app.route("/api/cancel", methods=["POST"])
def cancel():
    """中断当前任务并清空排队，同时取消自动续写链"""
    try:
        requests.post(f"{COMFY}/interrupt", timeout=5)
        requests.post(f"{COMFY}/queue", json={"clear": True}, timeout=5)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})
    with LOCK:
        TASKS.clear()
        _save_tasks()
    with LOCK_AUTO:
        for j in AUTO_JOBS.values():
            if j.get("status") == "running":
                j["cancel"] = True
    return jsonify({"ok": True})


@app.route("/api/active")
def active():
    """前端刷新后恢复跟踪：返回仍在进行（6h 内提交）的任务及参数"""
    cutoff = time.time() - 6 * 3600
    with LOCK:
        keys = ("prompt", "seed", "width", "height", "length", "steps", "turbo", "time",
                "model", "model_label")
        items = [{"task_id": tid, **{k: v.get(k) for k in keys}}
                 for tid, v in TASKS.items() if v.get("t0", 0) >= cutoff]
    return jsonify({"ok": True, "tasks": items})


@app.route("/api/queue")
def queue_view():
    """ComfyUI 队列视图（运行中+排队），合并面板元数据供展示/单任务取消"""
    try:
        q = requests.get(f"{COMFY}/queue", timeout=5).json()
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})
    with LOCK:
        # 清理：cancel_req 且已不在 ComfyUI 队列的任务 → 中断已生效
        live = {e[1] for k in ("queue_running", "queue_pending") for e in (q.get(k) or []) if len(e) > 1}
        gone = [tid for tid, v in TASKS.items() if v.get("cancel_req") and tid not in live]
        for tid in gone:
            TASKS.pop(tid, None)
        if gone:
            _save_tasks()
        known = {tid: dict(v) for tid, v in TASKS.items()}
    items = []
    for jid, j in POSTPROC_JOBS.items():
        if j.get("status") == "running":
            items.append({"task_id": jid, "status": "postproc", "postproc": True,
                          "prompt": f"{j.get('label')}：{(j.get('src') or '')[:60]}",
                          "seed": None, "width": None, "height": None, "length": None,
                          "steps": None, "turbo": None, "cancel_req": False})
    for status_key, status in (("queue_running", "running"), ("queue_pending", "queued")):
        for entry in q.get(status_key, []) or []:
            tid = entry[1] if len(entry) > 1 else None
            m = known.get(tid, {})
            items.append({"task_id": tid, "status": status,
                          "prompt": m.get("prompt"), "seed": m.get("seed"),
                          "width": m.get("width"), "height": m.get("height"),
                          "length": m.get("length"), "steps": m.get("steps"),
                          "turbo": m.get("turbo"), "model_label": m.get("model_label"),
                          "cancel_req": bool(m.get("cancel_req"))})
    return jsonify({"ok": True, "items": items})


@app.route("/api/task/<task_id>/cancel", methods=["POST"])
def cancel_task(task_id):
    """取消单个任务：排队的从队列删除，运行中的中断；pp_ 为后处理任务（终止进程）"""
    if task_id.startswith("pp_"):
        j = POSTPROC_JOBS.get(task_id)
        if not j or j.get("status") != "running":
            return jsonify({"ok": False, "error": "任务不存在或已结束"})
        try:
            j["proc"].terminate()
        except Exception:
            pass
        j["status"] = "cancelled"
        return jsonify({"ok": True, "status": "cancelled"})
    try:
        q = requests.get(f"{COMFY}/queue", timeout=5).json()
        running = any(len(e) > 1 and e[1] == task_id for e in q.get("queue_running", []) or [])
        pending = any(len(e) > 1 and e[1] == task_id for e in q.get("queue_pending", []) or [])
        if pending:
            requests.post(f"{COMFY}/queue", json={"delete": [task_id]}, timeout=5)
            with LOCK:
                TASKS.pop(task_id, None)
                _save_tasks()
        elif running:
            # 中断需要几秒才生效：先打标记留在 TASKS，队列视图确认其消失后再清理
            requests.post(f"{COMFY}/interrupt", timeout=5)
            with LOCK:
                if task_id in TASKS:
                    TASKS[task_id]["cancel_req"] = True
        else:
            in_hist = bool(requests.get(f"{COMFY}/history/{task_id}", timeout=5).json().get(task_id))
            with LOCK:
                TASKS.pop(task_id, None)
                _save_tasks()
            if not in_hist:
                return jsonify({"ok": True, "status": "gone"})
            return jsonify({"ok": False, "error": "任务已结束，无法取消"})
        return jsonify({"ok": True, "status": "cancelled"})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/api/stats")
def stats():
    """真实耗时分桶（按 分辨率×帧数×步数×加速）+ OOM 历史阈值，供前端校准 ETA"""
    with LOCK:
        vids = list(META["videos"].values())
        failures = list(META.get("failures", []))
    buckets = {}
    for m in vids:
        gs, w, h, ln, st = m.get("gen_seconds"), m.get("width"), m.get("height"), m.get("length"), m.get("steps")
        if not gs or not w or not h or not ln or m.get("merged") or m.get("chain") or m.get("postproc"):
            continue  # 拼接片/接龙段/后处理片不计入生成耗时分桶
        mdl = m.get("model") or UNET  # 历史数据无模型字段 → 都是出厂基线
        key = f"{w}x{h}x{ln}x{st}x{1 if m.get('turbo') else 0}x{mdl}"
        b = buckets.setdefault(key, {"w": w, "h": h, "len": ln, "steps": st, "model": mdl,
                                     "turbo": bool(m.get("turbo")), "n": 0, "total": 0})
        b["n"] += 1
        b["total"] += gs
    out = [{"w": b["w"], "h": b["h"], "len": b["len"], "steps": b["steps"],
            "turbo": b["turbo"], "model": b["model"], "n": b["n"], "avg": round(b["total"] / b["n"])}
           for b in buckets.values()]
    oom = [f.get("tokens") for f in failures
           if f.get("tokens") and ("OOM" in (f.get("error") or "") or "out of memory" in (f.get("error") or "").lower())]
    return jsonify({"ok": True, "buckets": out,
                    "oom_tokens_max": max(oom) if oom else None})


# ---------------- 模型库：已装扫描 / 显存适配推荐 / 后台下载 / 选用 ----------------
_DL_LOCK = threading.Lock()
DL = {"active": None}  # {id,filename,label,size_gb,kind,status,downloaded,total,speed,eta_s,started,cancel,error}
_HW_CACHE = {"t": 0, "data": None}
_GGUF_CACHE = {"t": 0, "value": None}


def _dl_restore():
    """面板重启后恢复下载状态：进行中的标记为中断（.part 仍在，支持断点续传）"""
    try:
        with open(DL_FILE, encoding="utf-8") as f:
            st = json.load(f).get("active")
        if st and st.get("status") in ("downloading", "starting"):
            st["status"] = "interrupted"
            st["error"] = "面板重启导致下载中断，重新点下载即可续传"
        DL["active"] = st
    except Exception:
        DL["active"] = None


def _dl_save():
    try:
        tmp = DL_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"active": DL.get("active")}, f, ensure_ascii=False)
        os.replace(tmp, DL_FILE)
    except Exception:
        pass


def _installed_models():
    try:
        return [fn for fn in sorted(os.listdir(MODELS_DIR))
                if not fn.startswith(".") and not fn.endswith((".part", ".tmp"))
                and fn.lower().endswith(MODEL_EXTS) and os.path.isfile(os.path.join(MODELS_DIR, fn))]
    except OSError:
        return []


def _model_profile(name):
    """生成请求里的模型名 → 工作流档案；空/未知回落默认模型"""
    fn = os.path.basename(str(name or "")) or DEFAULT_MODEL
    e = CATALOG.get(fn) or {}
    kind = e.get("kind") or ("gguf" if fn.lower().endswith(".gguf") else "safetensors")
    return {"filename": fn, "label": e.get("label") or fn, "kind": kind,
            "steps_fixed": e.get("steps_fixed"), "turbo_compat": e.get("turbo_compat", True),
            "needs_gguf": kind == "gguf"}


def _hardware():
    """GPU/显存/内存（缓存 30s，ComfyUI 离线时返回 online=False）"""
    if time.time() - _HW_CACHE["t"] < 30 and _HW_CACHE["data"]:
        return _HW_CACHE["data"]
    try:
        d = requests.get(f"{COMFY}/system_stats", timeout=4).json()
        dev = (d.get("devices") or [{}])[0]
        data = {"gpu": dev.get("name") or "未知", "vram_gb": round((dev.get("vram_total") or 0) / 1024 ** 3, 1),
                "ram_gb": round(((d.get("system", {}) or {}).get("ram_total") or 0) / 1024 ** 3, 1), "online": True}
    except Exception:
        data = {"gpu": None, "vram_gb": None, "ram_gb": None, "online": False}
    _HW_CACHE.update({"t": time.time(), "data": data})
    return data


def _gguf_ready():
    """ComfyUI 是否装有 GGUF 加载节点（True/False 缓存 5 分钟；离线返回 None 不缓存）"""
    if time.time() - _GGUF_CACHE["t"] < 300 and _GGUF_CACHE["value"] is not None:
        return _GGUF_CACHE["value"]
    try:
        ok = requests.get(f"{COMFY}/object_info/UnetLoaderGGUF", timeout=4).status_code == 200
    except Exception:
        return None
    _GGUF_CACHE.update({"t": time.time(), "value": ok})
    return ok


def _rec_tier(size_gb, vram_gb):
    """按 体积 vs 显存 给适配档位：fit=可常驻 / stream=流式加载 / heavy=不建议"""
    if not size_gb or not vram_gb:
        return None, "显存未知（ComfyUI 离线），无法评估适配"
    if size_gb <= vram_gb * 1.05:
        return "fit", f"约 {size_gb}GB ≤ 显存 {vram_gb}GB，权重可常驻显存，速度最优"
    if size_gb <= vram_gb * 2.2:
        return "stream", f"约 {size_gb}GB 大于显存 {vram_gb}GB，权重流式加载，速度受磁盘读取影响"
    return "heavy", f"约 {size_gb}GB 远超显存 {vram_gb}GB，本机使用大概率频繁换页，不建议"


def _download_worker(entry):
    a = DL.get("active") or {}
    fn = entry["filename"]
    final = os.path.join(MODELS_DIR, fn)
    part = final + ".part"
    a.update({"status": "downloading", "error": "",
              "downloaded": os.path.getsize(part) if os.path.isfile(part) else 0,
              "total": int(round((entry.get("size_gb") or 0) * 1024 ** 3)) or None,
              "speed": 0, "eta_s": None})
    _dl_save()
    last_save = 0.0
    for url in entry.get("urls") or []:
        headers = {"User-Agent": "h3-panel/5.0"}
        if a["downloaded"]:
            headers["Range"] = f"bytes={a['downloaded']}-"
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(10, 60), allow_redirects=True) as r:
                if r.status_code not in (200, 206):
                    a["error"] = f"{url.split('/')[2]} 返回 HTTP {r.status_code}"
                    continue
                if r.status_code == 200:
                    a["downloaded"] = 0  # 源不支持续传，重头下
                ct = r.headers.get("Content-Length")
                if ct:
                    a["total"] = int(ct) + (a["downloaded"] if r.status_code == 206 else 0)
                t0, b0 = time.time(), a["downloaded"]
                with open(part, "ab" if r.status_code == 206 else "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if a.get("cancel"):
                            a.update({"status": "cancelled", "speed": 0, "eta_s": None,
                                      "error": "已取消，已下载部分已清理"})
                            try:
                                os.remove(part)
                            except OSError:
                                pass
                            _dl_save()
                            return
                        if not chunk:
                            continue
                        f.write(chunk)
                        a["downloaded"] += len(chunk)
                        now = time.time()
                        if now - t0 >= 1:
                            a["speed"] = round((a["downloaded"] - b0) / (now - t0), 0)
                            a["eta_s"] = int((a["total"] - a["downloaded"]) / a["speed"]) \
                                if a.get("total") and a.get("speed") else None
                        if now - last_save >= 2:
                            _dl_save()
                            last_save = now
            os.replace(part, final)  # 同目录原子改名：下载完成即刻对 ComfyUI 可见
            a.update({"status": "done", "speed": 0, "eta_s": None, "error": ""})
            _dl_save()
            return
        except Exception as e:
            a["error"] = str(e)[:200]
            continue
    a.update({"status": "error", "speed": 0, "eta_s": None,
              "error": a.get("error") or "所有下载源均失败"})
    _dl_save()


def _model_entry_view(fn, hw):
    e = dict(CATALOG.get(fn) or {})
    e.setdefault("id", fn)
    e.setdefault("label", fn)
    e.setdefault("kind", "gguf" if fn.lower().endswith(".gguf") else "safetensors")
    e.setdefault("desc", "本地已安装的模型文件（未在内置目录登记）")
    e["filename"] = fn
    try:
        e["size_gb"] = round(os.path.getsize(os.path.join(MODELS_DIR, fn)) / 1024 ** 3, 2)
    except OSError:
        pass
    tier, note = _rec_tier(e.get("size_gb"), hw.get("vram_gb"))
    e.update({"rec": tier, "rec_note": note})
    return e


@app.route("/api/models")
def models_info():
    hw = _hardware()
    inst = _installed_models()
    act = DL.get("active")
    items = []
    for fn in inst:
        e = _model_entry_view(fn, hw)
        e.update({"installed": True, "is_default": fn == DEFAULT_MODEL,
                  "downloading": bool(act and act.get("filename") == fn and act.get("status") == "downloading")})
        items.append(e)
    known = {i["filename"] for i in items}
    avail = []
    gguf_ok = _gguf_ready()
    for fn, raw in CATALOG.items():
        if fn in known:
            continue
        e = dict(raw)
        tier, note = _rec_tier(e.get("size_gb"), hw.get("vram_gb"))
        e.update({"installed": False, "is_default": False, "rec": tier, "rec_note": note,
                  "need_gguf": e.get("kind") == "gguf" and gguf_ok is False})
        avail.append(e)
    return jsonify({"ok": True, "models_dir": MODELS_DIR, "default_model": DEFAULT_MODEL,
                    "hardware": hw, "gguf_ready": gguf_ok,
                    "installed": items, "available": avail, "download": act})


@app.route("/api/models/download", methods=["POST"])
def models_download():
    fn = os.path.basename((request.json or {}).get("filename") or "")
    entry = CATALOG.get(fn)
    if not entry or not entry.get("filename"):
        return jsonify({"ok": False, "error": "内置目录中没有这个模型"})
    with _DL_LOCK:
        act = DL.get("active")
        if act and act.get("status") in ("downloading", "starting"):
            return jsonify({"ok": False, "error": f"已有下载任务在进行：{act.get('label')}"})
        if fn in _installed_models():
            return jsonify({"ok": False, "error": "该模型已安装，无需重复下载"})
        need = int(round((entry.get("size_gb") or 0) * 1024 ** 3))
        if need:
            try:
                free = shutil.disk_usage(MODELS_DIR).free
            except Exception:
                free = None
            if free is not None and free < need * 1.05:
                return jsonify({"ok": False, "error": f"磁盘空间不足：需约 {entry['size_gb']}GB，"
                                                      f"目标盘仅剩 {free // 1024 ** 3}GB"})
        st = {"id": entry.get("id") or fn, "filename": fn, "label": entry.get("label") or fn,
              "size_gb": entry.get("size_gb"), "kind": entry.get("kind"), "status": "starting",
              "downloaded": 0, "total": None, "speed": 0, "eta_s": None,
              "started": int(time.time()), "cancel": False, "error": ""}
        DL["active"] = st
        _dl_save()
    threading.Thread(target=_download_worker, args=(entry,), daemon=True).start()
    return jsonify({"ok": True})


@app.route("/api/models/download/cancel", methods=["POST"])
def models_download_cancel():
    act = DL.get("active")
    if not act or act.get("status") not in ("downloading", "starting"):
        return jsonify({"ok": False, "error": "没有进行中的下载"})
    act["cancel"] = True
    return jsonify({"ok": True})


@app.route("/api/models/default", methods=["POST"])
def models_default():
    global DEFAULT_MODEL
    fn = os.path.basename((request.json or {}).get("filename") or "")
    if fn not in _installed_models():
        return jsonify({"ok": False, "error": "模型未安装，不能设为默认"})
    DEFAULT_MODEL = fn
    try:
        with open(PANEL_CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["default_model"] = fn
        tmp = PANEL_CONFIG_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        os.replace(tmp, PANEL_CONFIG_PATH)
    except Exception as e:
        return jsonify({"ok": False, "error": f"写入配置失败: {e}"})
    return jsonify({"ok": True, "default_model": fn})


@app.route("/api/models/<path:fn>", methods=["DELETE"])
def models_delete(fn):
    fn = os.path.basename(fn)
    path = os.path.join(MODELS_DIR, fn)
    if not os.path.isfile(path):
        return jsonify({"ok": False, "error": "文件不存在"})
    if fn == DEFAULT_MODEL:
        return jsonify({"ok": False, "error": "默认模型不可删除；请先把其他模型设为默认"})
    act = DL.get("active")
    if act and act.get("filename") == fn and act.get("status") == "downloading":
        return jsonify({"ok": False, "error": "该模型正在下载中，先取消下载"})
    try:
        os.remove(path)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})
    return jsonify({"ok": True})


# ---------------- 后处理：RIFE 插帧 ×2 / Real-ESRGAN 超分 ×2 ----------------
POSTPROC_JOBS = {}  # job_id(pp_*) -> {status,mode,src,out,proc,label,t0,error}
PP_PYTHON = CONFIG["python"]
PP_CLI = os.path.join(CONFIG["postproc_dir"], "postproc_cli.py")
PP_VRAM_NEED = {"rife": 2.5, "x2": 4.5, "rife_x2": 4.5}


def _vram_free_gb():
    try:
        d = requests.get(f"{COMFY}/system_stats", timeout=5).json()
        dev = (d.get("devices") or [{}])[0]
        return round(dev.get("vram_free", 0) / 1e9, 2)
    except Exception:
        return 0.0


def _pp_engine_check(mode):
    """引擎文件完整性预检，缺什么提示装什么"""
    missing = []
    fi = os.path.join(CONFIG["postproc_dir"], "Frame-Interpolation")
    if not os.path.isdir(fi):
        missing.append("Frame-Interpolation（在 postproc 目录运行 install.sh）")
    if mode in ("rife", "rife_x2") and not os.path.isfile(os.path.join(CONFIG["postproc_dir"], "ckpts", "rife47.pth")):
        missing.append("rife47.pth（install.sh 自动下载）")
    if mode in ("x2", "rife_x2") and not os.path.isfile(os.path.join(CONFIG["postproc_dir"], "ckpts", "x2plus.pth")):
        missing.append("x2plus.pth（install.sh 自动下载）")
    if missing:
        return "AI 增强引擎未就绪，缺少：" + "、".join(missing)
    return None


def _ensure_vram(need_gb):
    """显存不够先让 ComfyUI 卸载闲置模型，再查一次；仍不足则拒绝"""
    if _vram_free_gb() >= need_gb:
        return None
    try:
        requests.post(f"{COMFY}/free", json={"unload_models": True}, timeout=30)
    except Exception:
        pass
    time.sleep(2)
    free = _vram_free_gb()
    if free >= need_gb:
        return None
    return f"显存不足（当前仅 {free}GB 可用，需约 {need_gb}GB）。请等正在生成的任务结束后再试"


def _pp_out_name(src, tag):
    base = (src[:-4] if src.endswith(".mp4") else src).rstrip("_")
    cand = f"{base}_{tag}.mp4"
    i = 2
    while os.path.exists(os.path.join(OUTPUT, cand)):
        cand = f"{base}_{tag}_{i}.mp4"
        i += 1
    return cand


def _probe_dim(path):
    out = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=width,height", "-of", "csv=p=0", path],
                         capture_output=True, text=True, timeout=10)
    w, h = out.stdout.strip().split(",")
    return int(w), int(h)


def _pp_run(name, mode, out_path):
    """跑一次后处理 CLI 并把产物写入元数据"""
    env = dict(os.environ)
    if CONFIG["postproc_home"]:
        env["HOME"] = CONFIG["postproc_home"]
    proc = subprocess.Popen([PP_PYTHON, PP_CLI, mode, os.path.join(OUTPUT, name), out_path],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    try:
        out, err = proc.communicate(timeout=1800)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError("处理超时（30 分钟上限）")
    if proc.returncode != 0:
        raise RuntimeError((err or "").strip()[-300:] or "处理失败")
    info = json.loads(out.strip().splitlines()[-1])
    w, h = _probe_dim(out_path)
    with LOCK:
        sm = META["videos"].get(name, {})
        META["videos"][os.path.basename(out_path)] = {
            "prompt": sm.get("prompt") or name, "seed": sm.get("seed"),
            "width": w, "height": h, "length": sm.get("length"),
            "steps": sm.get("steps"), "turbo": sm.get("turbo"), "i2v": sm.get("i2v"),
            "sampler": sm.get("sampler"), "scheduler": sm.get("scheduler"),
            "model": sm.get("model"), "model_label": sm.get("model_label"),
            "duration": _probe_duration(out_path), "postproc": mode, "src": name,
            "gen_seconds": info.get("proc_seconds"), "time": int(time.time()),
        }
        _save_meta()
    return info


def run_postproc(job_id, name, mode):
    job = POSTPROC_JOBS[job_id]
    try:
        if mode == "rife_x2":
            # 全链配方：先插帧（正常入库）→ 对插帧产物再超分（入库）
            r1 = _pp_out_name(name, "rife")
            _pp_run(name, "rife", os.path.join(OUTPUT, r1))
            r2 = _pp_out_name(r1, "x2")
            info = _pp_run(r1, "x2", os.path.join(OUTPUT, r2))
            job.update({"status": "done", "out": r2, "seconds": info.get("total_seconds")})
        else:
            out_name = _pp_out_name(name, mode)
            info = _pp_run(name, mode, os.path.join(OUTPUT, out_name))
            job.update({"status": "done", "out": out_name, "seconds": info.get("total_seconds")})
    except Exception as e:
        job.update({"status": "error", "error": str(e)[:300]})


@app.route("/api/postproc", methods=["POST"])
def postproc():
    data = request.json or {}
    name = os.path.basename(data.get("name") or "")
    mode = data.get("mode") if data.get("mode") in ("rife", "x2", "rife_x2") else None
    if not name or not mode:
        return jsonify({"ok": False, "error": "参数缺失"})
    if not os.path.isfile(os.path.join(OUTPUT, name)):
        return jsonify({"ok": False, "error": "视频不存在"})
    if name.startswith("merged_") and mode == "x2":
        return jsonify({"ok": False, "error": "拼接长片较长，暂不支持超分（可对单段超分后再拼接）"})
    engine_err = _pp_engine_check(mode)
    if engine_err:
        return jsonify({"ok": False, "error": engine_err})
    err = _ensure_vram(PP_VRAM_NEED[mode])
    if err:
        return jsonify({"ok": False, "error": err})
    running = [j for j in POSTPROC_JOBS.values() if j.get("status") == "running"]
    if len(running) >= 2:
        return jsonify({"ok": False, "error": "后处理任务过多，请等当前任务完成"})
    job_id = f"pp_{int(time.time()*1000):x}"
    label = {"rife": "🪄 插帧 ×2", "x2": "🪄 AI 超分 ×2", "rife_x2": "🪄 插帧→超分 全链"}[mode]
    POSTPROC_JOBS[job_id] = {"status": "running", "mode": mode, "src": name,
                             "label": label, "t0": time.time(), "error": ""}
    threading.Thread(target=run_postproc, args=(job_id, name, mode), daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id, "mode": mode})


@app.route("/video/<path:filename>")
def video(filename):
    return send_from_directory(OUTPUT, filename)


@app.route("/api/videos")
def videos():
    files = sorted(glob.glob(os.path.join(OUTPUT, "h3_video*.mp4")) +
                   glob.glob(os.path.join(OUTPUT, "merged_*.mp4")),
                   key=os.path.getmtime, reverse=True)
    with LOCK:
        known = dict(META["videos"])
    out = []
    for f in files:
        name = os.path.basename(f)
        m = known.get(name, {})
        out.append({"name": name, "size": os.path.getsize(f), "time": int(os.path.getmtime(f)),
                    "prompt": m.get("prompt"), "seed": m.get("seed"),
                    "width": m.get("width"), "height": m.get("height"),
                    "length": m.get("length"), "duration": m.get("duration"),
                    "steps": m.get("steps"), "turbo": m.get("turbo"), "i2v": m.get("i2v"),
                    "sampler": m.get("sampler"), "scheduler": m.get("scheduler"), "crop": m.get("crop"),
                    "gen_seconds": m.get("gen_seconds"), "end_frame": m.get("end_frame"),
                    "model": m.get("model"), "model_label": m.get("model_label"),
                    "postproc": m.get("postproc"), "src": m.get("src"),
                    "merged": m.get("merged"), "segments": m.get("segments"), "star": m.get("star")})
    return jsonify(out)


@app.route("/api/video/<path:name>", methods=["DELETE"])
def del_video(name):
    name = os.path.basename(name)
    if not name.endswith(".mp4"):
        return jsonify({"ok": False, "error": "仅支持删除 mp4"})
    path = os.path.join(OUTPUT, name)
    if not os.path.isfile(path):
        return jsonify({"ok": False, "error": "文件不存在"})
    try:
        os.remove(path)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})
    with LOCK:
        META["videos"].pop(name, None)
        _save_meta()
    return jsonify({"ok": True})


@app.route("/api/lastframe/<path:name>")
def lastframe(name):
    """提取某个已生成视频的最后一帧，返回 data URL，用于续写接龙"""
    name = os.path.basename(name)
    path = os.path.join(OUTPUT, name)
    if not os.path.isfile(path):
        return jsonify({"ok": False, "error": "视频不存在"})
    fd, tmp = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        r = subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-sseof", "-0.05",
                            "-i", path, "-frames:v", "1", tmp], timeout=30)
        if r.returncode != 0 or os.path.getsize(tmp) == 0:
            return jsonify({"ok": False, "error": "尾帧提取失败"})
        with open(tmp, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return jsonify({"ok": True, "data_url": "data:image/png;base64," + b64})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


@app.route("/api/concat", methods=["POST"])
def concat():
    """把多段视频按顺序拼成一条；分辨率一致时无损 copy，否则统一转码"""
    names = (request.json or {}).get("names") or []
    if not (2 <= len(names) <= 8):
        return jsonify({"ok": False, "error": "请选择 2-8 段视频"})
    paths, dims = [], []
    for n in names:
        n = os.path.basename(n)
        p = os.path.join(OUTPUT, n)
        if not os.path.isfile(p):
            return jsonify({"ok": False, "error": f"文件不存在: {n}"})
        out = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                              "-show_entries", "stream=width,height", "-of", "csv=p=0", p],
                             capture_output=True, text=True, timeout=10)
        try:
            w, h = [int(x) for x in out.stdout.strip().split(",")]
        except Exception:
            return jsonify({"ok": False, "error": f"无法解析视频: {n}"})
        paths.append(p)
        dims.append((w, h))
    # 输出文件名递增
    idx = 1
    while os.path.exists(os.path.join(OUTPUT, f"merged_{idx:04d}.mp4")):
        idx += 1
    out_name = f"merged_{idx:04d}.mp4"
    out_path = os.path.join(OUTPUT, out_name)
    fd, lst = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    with open(lst, "w") as f:
        for p in paths:
            f.write(f"file '{p}'\n")
    same = len(set(dims)) == 1
    try:
        if same:
            cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                   "-i", lst, "-c", "copy", out_path]
        else:
            w0, h0 = dims[0]
            vf = f"scale={w0}:{h0}:force_original_aspect_ratio=decrease,pad={w0}:{h0}:(ow-iw)/2:(oh-ih)/2"
            cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                   "-i", lst, "-vf", vf, "-r", "24", "-c:v", "libx264", "-crf", "19",
                   "-preset", "veryfast", "-c:a", "aac", "-b:a", "192k", out_path]
        r = subprocess.run(cmd, timeout=600)
        if r.returncode != 0 or not os.path.isfile(out_path):
            return jsonify({"ok": False, "error": "拼接失败"})
    finally:
        try:
            os.unlink(lst)
        except Exception:
            pass
    dur = _probe_duration(out_path)
    with LOCK:
        META["videos"][out_name] = {"prompt": "（拼接 " + str(len(paths)) + " 段长片）", "seed": None,
                                    "width": dims[0][0], "height": dims[0][1],
                                    "length": None, "duration": dur, "merged": True,
                                    "segments": [os.path.basename(p) for p in paths],
                                    "time": int(time.time())}
        _save_meta()
    return jsonify({"ok": True, "name": out_name, "duration": dur})



# ---------------- 提示词优化 ----------------
CFG_FILE = os.path.join(CONFIG["data_dir"], "llm_config.json")
CAMERA = ["缓慢推近的镜头", "平滑跟随的运镜", "缓慢环绕的镜头", "固定机位，轻微手持呼吸感",
          "由远及近的推轨镜头", "低角度仰拍缓缓上升"]
LIGHT = ["自然柔和的光线", "黄金时刻的暖色光", "电影级布光，明暗对比分明", "逆光轮廓光，体积光氛围",
         "柔和的散射光"]
TAIL = ["电影感构图", "画面细腻，色彩层次丰富", "主体清晰，细节丰富", "氛围感强"]

CATEGORY = [
    (("猫", "狗", "宠物", "动物", "鸟", "马", "象", "鹿"), "动物毛发与动作姿态自然生动"),
    (("人", "女孩", "男孩", "行人", "舞者", "老人", "孩子", "少女"), "人物动作流畅自然，表情生动传神"),
    (("城市", "街头", "东京", "都市", "霓虹", "建筑", "街景"), "城市环境细节丰富，景深层次分明"),
    (("海", "山", "森林", "草原", "沙漠", "天空", "湖", "河", "自然"), "自然光影变化细腻，大气透视感强"),
    (("美食", "蛋糕", "拉面", "特写", "料理", "咖啡", "巧克力"), "微距细节清晰，色彩饱满有食欲感"),
    (("产品", "商品", "耳机", "手机", "手表", "包装"), "商业级打光，材质质感突出"),
    (("太空", "宇宙", "宇宙飞船", "科幻", "赛博", "未来", "机器人"), "未来感氛围浓郁，光效绚丽"),
    (("车", "汽车", "跑车", "驾驶"), "车身光影流动，速度感与机械质感并存"),
]


def _pick(pool, variant, exclude):
    for i in range(len(pool)):
        item = pool[(variant + i) % len(pool)]
        if not any(e and e in item for e in exclude) and item not in exclude:
            return item
    return None


def local_optimize(prompt, variant=0):
    """本地规则增强：按内容类别补充镜头/光影/画质描述，去重后拼接"""
    have = prompt
    parts = [prompt.rstrip("。，,．. ")]
    for kws, desc in CATEGORY:
        if any(k in prompt for k in kws):
            if not any(x in have for x in desc.split("，")):
                parts.append(desc)
            break
    cam = _pick(CAMERA, variant, [have])
    if cam:
        parts.append(cam)
    light = _pick(LIGHT, variant, have + "".join(parts))
    if light:
        parts.append(light)
    tail = _pick(TAIL, variant, have + "".join(parts))
    if tail:
        parts.append(tail)
    return "，".join(parts)


def llm_optimize(prompt, cfg, variant=0):
    """可选：调用 OpenAI 兼容接口优化提示词"""
    sys_prompt = ("你是专业的 AI 视频生成提示词工程师。把用户的简短想法扩写成一条适合视频生成模型的中文提示词："
                  "保留原意，补充主体细节、动作过程、镜头运动、光线、氛围和画质描述，控制在 90 字以内。"
                  "只输出优化后的提示词本身，不要任何解释、引号或前后缀。")
    r = requests.post(
        cfg["api_base"].rstrip("/") + "/chat/completions",
        headers={"Authorization": "Bearer " + cfg["api_key"]},
        json={"model": cfg.get("api_model") or "gpt-4o-mini",
              "messages": [{"role": "system", "content": sys_prompt},
                           {"role": "user", "content": prompt + (f"（变化风格第 {variant + 1} 版）" if variant else "")}],
              "temperature": 0.7, "max_tokens": 300},
        timeout=25)
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"].strip()
    text = text.strip(chr(34) + chr(39) + "“”‘’")
    if not text:
        raise ValueError("empty response")
    return text


def _load_cfg():
    try:
        with open(CFG_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


@app.route("/api/optimize", methods=["POST"])
def optimize():
    data = request.json or {}
    prompt = (data.get("prompt") or "").strip()
    variant = int(data.get("variant", 0) or 0)
    if len(prompt) < 4:
        return jsonify({"ok": False, "error": "提示词太短"})
    cfg = _load_cfg()
    if cfg.get("api_base") and cfg.get("api_key"):
        try:
            text = llm_optimize(prompt, cfg, variant)
            return jsonify({"ok": True, "optimized": text, "mode": "云端LLM"})
        except Exception as e:
            # 云端失败回落本地
            return jsonify({"ok": True, "optimized": local_optimize(prompt, variant),
                            "mode": "本地规则（云端失败: %s）" % str(e)[:60]})
    return jsonify({"ok": True, "optimized": local_optimize(prompt, variant), "mode": "本地规则"})


@app.route("/api/settings", methods=["GET", "POST"])
def settings():
    if request.method == "GET":
        cfg = _load_cfg()
        key = cfg.get("api_key") or ""
        return jsonify({"api_base": cfg.get("api_base") or "",
                        "api_model": cfg.get("api_model") or "",
                        "has_key": bool(key), "key_tail": key[-4:] if key else ""})
    data = request.json or {}
    cfg = _load_cfg()
    cfg["api_base"] = (data.get("api_base") or "").strip()
    cfg["api_model"] = (data.get("api_model") or "").strip()
    if (data.get("api_key") or "").strip():
        cfg["api_key"] = data.get("api_key").strip()
    elif data.get("clear_key"):
        cfg.pop("api_key", None)
    with open(CFG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)
    return jsonify({"ok": True, "llm": bool(cfg.get("api_base") and cfg.get("api_key"))})


@app.route("/api/star", methods=["POST"])
def star():
    data = request.json or {}
    name = os.path.basename(data.get("name") or "")
    if not name:
        return jsonify({"ok": False, "error": "缺少文件名"})
    path = os.path.join(OUTPUT, name)
    if not os.path.isfile(path):
        return jsonify({"ok": False, "error": "文件不存在"})
    with LOCK:
        meta = META["videos"].setdefault(name, {"time": int(os.path.getmtime(path))})
        meta["star"] = bool(data.get("star"))
        _save_meta()
    return jsonify({"ok": True, "star": meta["star"]})



# ---------------- 自动续写长片（后端编排） ----------------
AUTO_JOBS = {}
LOCK_AUTO = threading.Lock()

def _extract_err(item):
    try:
        for m in ((item.get("status") or {}).get("messages") or []):
            if isinstance(m, list) and len(m) == 2 and m[0] == "execution_error":
                d = m[1] or {}
                msg = d.get("exception_message") or "未知错误"
                if "out of memory" in msg.lower():
                    msg = "显存不足（OOM）：请降低每段时长或画质档位"
                return f"{d.get('node_type', '')}: {msg}"[:200]
    except Exception:
        pass
    return "任务执行失败，详见 ComfyUI 日志"


def _wait_task(tid, job, timeout=1500):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if job.get("cancel"):
            return ("cancelled", None)
        try:
            h = requests.get(f"{COMFY}/history/{tid}", timeout=10).json()
        except Exception:
            h = {}
        item = h.get(tid)
        if item is not None:
            st = item.get("status", {})
            if st.get("status_str") == "error":
                return ("error", _extract_err(item))
            vid = _find_mp4(item.get("outputs", {}))
            if vid:
                return ("done", vid)
            if st.get("completed"):
                return ("done", None)
        time.sleep(4)
    return ("timeout", "单段生成超时")


def _last_frame_upload(video_name):
    """提取视频最后一帧并上传到 ComfyUI，返回 image name"""
    path = os.path.join(OUTPUT, os.path.basename(video_name))
    fd, tmp = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        r = subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-sseof", "-0.05",
                            "-i", path, "-frames:v", "1", tmp], timeout=30)
        if r.returncode != 0 or os.path.getsize(tmp) == 0:
            return None
        with open(tmp, "rb") as f:
            up = requests.post(f"{COMFY}/upload/image",
                               files={"image": ("chain_frame.png", f.read(), "image/png")}, timeout=30)
        if up.status_code == 200:
            return up.json()["name"]
    except Exception:
        pass
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass
    return None


def _record_meta(video_name, prompt, w, h, length, steps, turbo, i2v, seg, chain_id, gen_seconds=None,
                 model=None, model_label=None):
    with LOCK:
        META["videos"][video_name] = {
            "prompt": prompt, "seed": None, "width": w, "height": h,
            "length": length, "steps": steps, "turbo": turbo, "i2v": i2v,
            "duration": _probe_duration(os.path.join(OUTPUT, video_name)),
            "gen_seconds": gen_seconds,
            "model": model, "model_label": model_label,
            "chain": chain_id, "segment": seg, "time": int(time.time()),
        }
        _save_meta()


def run_auto_job(job_id, p):
    job = AUTO_JOBS[job_id]
    job.update({"status": "running", "current": 0, "videos": [], "merged": None,
                "error": "", "phase": "generating"})
    prev_image = None
    try:
        for i in range(job["total"]):
            if job.get("cancel"):
                job["status"] = "cancelled"
                return
            actual_seed = (p["seed"] + i) if p["seed"] else random.randint(1, 2 ** 31 - 1)
            wf = build_workflow(p["prompt"], p["width"], p["height"], p["length"], actual_seed,
                                p["steps"], prev_image, None, p["turbo"], p["sampler"], p["scheduler"],
                                {"filename": p["model"], "kind": p["model_kind"]})
            try:
                r = requests.post(f"{COMFY}/prompt", json={"prompt": wf, "client_id": "h3-panel"}, timeout=10).json()
            except Exception:
                job["status"] = "error"
                job["error"] = f"无法连接 ComfyUI（{COMFY}），请确认其正在运行"
                return
            if "prompt_id" not in r:
                job["status"] = "error"
                job["error"] = r.get("error", {}).get("message", "工作流校验失败")
                return
            job["current"] = i + 1
            job["phase"] = "generating"
            seg_t0 = time.time()
            status, vid = _wait_task(r["prompt_id"], job)
            if job.get("cancel"):
                job["status"] = "cancelled"
                return
            if status != "done" or not vid:
                job["status"] = "error"
                job["error"] = vid or "未知错误"
                if len(job["videos"]) >= 2:
                    job["phase"] = "concat-partial"
                return
            job["videos"].append(vid)
            _record_meta(vid, p["prompt"], p["width"], p["height"], p["length"], p["steps"],
                         p["turbo"], i > 0, i + 1, job_id, gen_seconds=round(time.time() - seg_t0),
                         model=p.get("model"), model_label=p.get("model_label"))
            if i < job["total"] - 1:
                job["phase"] = "extracting"
                prev_image = _last_frame_upload(vid)
                if not prev_image:
                    job["status"] = "error"
                    job["error"] = f"第 {i + 1} 段尾帧提取失败，链式中断"
                    return
        # 全部完成 → 拼接
        if len(job["videos"]) >= 2:
            job["phase"] = "concat"
            fd, lst = tempfile.mkstemp(suffix=".txt")
            os.close(fd)
            with open(lst, "w") as f:
                for v in job["videos"]:
                    f.write(f"file '{os.path.join(OUTPUT, v)}'\n")
            idx = 1
            while os.path.exists(os.path.join(OUTPUT, f"merged_{idx:04d}.mp4")):
                idx += 1
            out_name = f"merged_{idx:04d}.mp4"
            try:
                r = subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat",
                                    "-safe", "0", "-i", lst, "-c", "copy",
                                    os.path.join(OUTPUT, out_name)], timeout=600)
            finally:
                try:
                    os.unlink(lst)
                except Exception:
                    pass
            if r.returncode == 0:
                job["merged"] = out_name
                with LOCK:
                    META["videos"][out_name] = {
                        "prompt": p["prompt"] + f"（自动续写 {job['total']} 段长片）", "seed": None,
                        "width": p["width"], "height": p["height"], "length": None,
                        "duration": _probe_duration(os.path.join(OUTPUT, out_name)),
                        "merged": True, "chain": job_id, "time": int(time.time()),
                    }
                    _save_meta()
            else:
                job["error"] = "拼接失败（各段仍保留在历史中）"
        job["status"] = "done"
    except Exception as e:
        job["status"] = "error"
        job["error"] = f"内部错误: {e}"


@app.route("/api/autochain", methods=["POST"])
def autochain():
    data = request.json or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"ok": False, "error": "请输入提示词"})
    with LOCK_AUTO:
        active = [j for j in AUTO_JOBS.values() if j.get("status") == "running"]
        if active:
            return jsonify({"ok": False, "error": "已有一个自动续写任务在进行，请等它完成或点取消"})
    total = max(2, min(8, int(data.get("segments", 3) or 3)))
    width = int(data.get("width", 864)); height = int(data.get("height", 480))
    length = int(data.get("length", 124))
    tokens = width * height * length
    if tokens > TOKEN_BUDGET and not bool(data.get("force", False)):
        return jsonify({"ok": False, "over_budget": True,
                        "error": f"单段组合（{width}×{height} · {length}帧≈{length//24}秒）超出显存预算，长片无法生成。"
                                 f"请降低每段时长或画质档位。"})
    # 长片接龙同样支持模型选择（含加速版的步数锁定 / TeaCache 互斥）
    prof = _model_profile(data.get("model"))
    if data.get("model") and prof["filename"] != DEFAULT_MODEL and prof["filename"] not in _installed_models():
        return jsonify({"ok": False, "error": f"模型未安装：{prof['label']}"})
    if prof["needs_gguf"] and _gguf_ready() is not True:
        return jsonify({"ok": False, "error": "ComfyUI 未检出 GGUF 加载组件，无法使用 GGUF 模型"})
    p = {"prompt": prompt, "width": width, "height": height, "length": length,
         "steps": int(prof["steps_fixed"]) if prof.get("steps_fixed") else max(4, min(20, int(data.get("steps", 8) or 8))),
         "seed": int(data.get("seed", 0) or 0),
         "turbo": bool(data.get("turbo", True)) and prof.get("turbo_compat", True),
         "sampler": data.get("sampler") or "euler", "scheduler": data.get("scheduler") or "normal",
         "model": prof["filename"], "model_kind": prof["kind"], "model_label": prof["label"]}
    job_id = f"chain_{int(time.time()*1000):x}"
    AUTO_JOBS[job_id] = {"total": total, "status": "starting", "params": {k: v for k, v in p.items()}}
    threading.Thread(target=run_auto_job, args=(job_id, p), daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id, "total": total,
                    "width": width, "height": height, "length": length})


@app.route("/api/autojob/<job_id>")
def autojob(job_id):
    job = AUTO_JOBS.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "任务不存在"})
    out = {"ok": True, "status": job.get("status"), "current": job.get("current", 0),
           "total": job.get("total"), "phase": job.get("phase"), "merged": job.get("merged"),
           "error": job.get("error", ""), "videos": job.get("videos", [])}
    return jsonify(out)


# ---------------- 通宵批量：提示词池无人值守编排 ----------------
BATCH_FILE = os.path.join(CONFIG["data_dir"], "batch.json")
BATCH_JOBS = {}  # id -> job（同时只跑一个批次；文件里保留最近一个供重启后查看汇总）
BATCH_LOCK = threading.Lock()
_BATCH_LAST_ID = None
BATCH_MAX_PROMPTS = 60
BATCH_MAX_TASKS = 240
PP_EST = {"rife": 25, "x2": 85, "rife_x2": 110}  # 自动后处理单任务秒级估算


def _batch_save():
    try:
        tmp = BATCH_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"job": BATCH_JOBS.get(_BATCH_LAST_ID)}, f, ensure_ascii=False)
        os.replace(tmp, BATCH_FILE)
    except Exception:
        pass


def _estimate_seconds_per(p):
    """单任务耗时估算：优先同参数（含模型）的历史实测，退化为线性模型，加自动后处理估算"""
    per = None
    try:
        with LOCK:
            vals = [m["gen_seconds"] for m in META["videos"].values()
                    if m.get("gen_seconds") and m.get("width") == p["width"] and m.get("height") == p["height"]
                    and m.get("length") == p["length"] and m.get("steps") == p["steps"]
                    and bool(m.get("turbo")) == p["turbo"] and (m.get("model") or UNET) == p["model"]
                    and not m.get("merged") and not m.get("chain") and not m.get("postproc")]
        if vals:
            per = sum(vals) / len(vals)
    except Exception:
        pass
    if not per:
        tokens = p["width"] * p["height"] * p["length"]
        per = 143 * tokens / (864 * 480 * 124) * (p["steps"] / 8) * (1 if p["turbo"] else 1.43) \
              * (1 + tokens / TOKEN_BUDGET * 0.35)
    return round(per + PP_EST.get(p.get("autopost") or "", 0))


def _batch_autopost(video, p, failures, prompt, seed):
    """批次内同步执行自动后处理；引擎/显存不就绪记一条说明，不中断批次"""
    mode = p.get("autopost")
    if not mode or not video:
        return
    eng = _pp_engine_check(mode)
    if eng:
        failures.append({"prompt": prompt, "seed": seed, "step": p["steps"],
                         "error": "成片已入库，自动处理跳过：" + eng})
        return
    vram_err = _ensure_vram(PP_VRAM_NEED[mode])
    if vram_err:
        failures.append({"prompt": prompt, "seed": seed, "step": p["steps"],
                         "error": "成片已入库，自动处理跳过：" + vram_err})
        return
    try:
        if mode == "rife_x2":
            r1 = _pp_out_name(video, "rife")
            _pp_run(video, "rife", os.path.join(OUTPUT, r1))
            r2 = _pp_out_name(r1, "x2")
            _pp_run(r1, "x2", os.path.join(OUTPUT, r2))
        else:
            _pp_run(video, mode, os.path.join(OUTPUT, _pp_out_name(video, mode)))
    except Exception as e:
        failures.append({"prompt": prompt, "seed": seed, "step": p["steps"],
                         "error": f"成片已入库，自动处理失败：{str(e)[:150]}"})


def _batch_submit_one(prompt, seed, p):
    wf = build_workflow(prompt, p["width"], p["height"], p["length"], seed, p["steps"],
                        None, None, p["turbo"], p["sampler"], p["scheduler"],
                        {"filename": p["model"], "kind": p["model_kind"]})
    r = requests.post(f"{COMFY}/prompt", json={"prompt": wf, "client_id": "h3-panel"}, timeout=10)
    res = r.json()
    if "prompt_id" not in res:
        raise RuntimeError(res.get("error", {}).get("message", "工作流校验失败"))
    return res["prompt_id"], wf["6"]["inputs"]["width"], wf["6"]["inputs"]["height"]


def run_batch_job(job_id):
    """通宵编排器：逐个提交-等待-入库，失败跳过，连接中断按 60s×30 次重试"""
    job = BATCH_JOBS[job_id]
    p = job["params"]
    try:
        while True:
            if job.get("cancel"):
                job["status"] = "cancelled"
                job["current"] = None
                break
            # 重启续跑：先收尾上个进程提交、尚未等完的在途任务（队列里没有它，不会重复跑）
            if job.get("pending_tid"):
                ptid = job["pending_tid"]
                job["current"] = {"index": "·", "prompt": "（重启前在途任务收尾中）", "seed": None}
                t0 = time.time()
                status, vid = _wait_task(ptid, {"cancel": False}, timeout=max(1800, p["length"] * 15))
                job["pending_tid"] = None
                job["current"] = None
                if status == "done" and vid:
                    job["videos"].append(vid)
                    job["done"] += 1
                    _record_meta(vid, "（重启前提交）", p["width"], p["height"], p["length"],
                                 p["steps"], p["turbo"], False, None, None,
                                 gen_seconds=round(time.time() - t0),
                                 model=p.get("model"), model_label=p.get("model_label"))
                    with LOCK:
                        META["videos"][vid]["batch"] = job_id
                        _save_meta()
                    if p.get("autopost"):
                        _batch_autopost(vid, p, job["failures"], "（重启前提交）", None)
                else:
                    job["failed"] += 1
                    job["failures"].append({"prompt": "（重启前提交）", "seed": None, "step": p["steps"],
                                            "error": vid if isinstance(vid, str) and vid else "在途任务超时或无输出"})
                _batch_save()
                continue
            with BATCH_LOCK:
                if not job["queue"]:
                    job["status"] = "done"
                    job["current"] = None
                    break
                prompt, seed = job["queue"][0]
                job["current"] = {"index": job["total"] - len(job["queue"]) + 1,
                                  "prompt": prompt, "seed": seed}
                job["current_tid"] = None
            # 提交（连接类异常 60s×30 重试；工作流校验失败属参数性问题，记失败即跳过）
            tid = w = h = None
            dead = False
            for attempt in range(30):
                if job.get("cancel"):
                    break
                try:
                    tid, w, h = _batch_submit_one(prompt, seed, p)
                    job["conn_retry"] = 0
                    break
                except requests.RequestException:
                    job["conn_retry"] = attempt + 1
                    if attempt < 29:
                        time.sleep(60)
                except RuntimeError as e:
                    job["failures"].append({"prompt": prompt, "seed": seed, "step": p["steps"],
                                            "error": str(e)[:200]})
                    break
            if job.get("cancel"):
                job["status"] = "cancelled"
                job["current"] = None
                break
            with BATCH_LOCK:
                job["queue"].pop(0)
            if tid is None:
                if job["conn_retry"]:
                    job["failures"].append({"prompt": prompt, "seed": seed, "step": p["steps"],
                                            "error": "ComfyUI 持续无法连接（已重试 30 分钟）"})
                    job["status"] = "error"
                    job["error"] = "ComfyUI 持续无法连接，批次已终止"
                    job["current"] = None
                    _batch_save()
                    return
                job["failed"] += 1
                job["current"] = None
                _batch_save()
                continue
            job["current_tid"] = tid
            _batch_save()
            t0 = time.time()
            # 传无 cancel 标记的 shim：当前片跑完再停，半途不打断
            status, vid = _wait_task(tid, {"cancel": False}, timeout=max(1800, p["length"] * 15))
            job["current"] = None
            if status == "done" and vid:
                job["videos"].append(vid)
                job["done"] += 1
                _record_meta(vid, prompt, w, h, p["length"], p["steps"], p["turbo"], False,
                             None, None, gen_seconds=round(time.time() - t0),
                             model=p.get("model"), model_label=p.get("model_label"))
                with LOCK:
                    META["videos"][vid]["batch"] = job_id
                    _save_meta()
                if p.get("autopost"):
                    _batch_autopost(vid, p, job["failures"], prompt, seed)
            else:
                job["failed"] += 1
                job["failures"].append({"prompt": prompt, "seed": seed, "step": p["steps"],
                                        "error": vid if isinstance(vid, str) and vid else "任务超时或无输出"})
            if job.get("eta_seconds_per"):
                job["eta_end"] = int(time.time() + len(job["queue"]) * job["eta_seconds_per"])
            _batch_save()
        _batch_save()
    except Exception as e:
        job["status"] = "error"
        job["error"] = f"内部错误: {e}"[:200]
        _batch_save()


def _batch_restore():
    """面板重启后：running 批次续跑（在途任务依赖 TASKS watcher 收尾），已完成批次保留汇总"""
    global _BATCH_LAST_ID
    try:
        with open(BATCH_FILE, encoding="utf-8") as f:
            job = json.load(f).get("job")
    except Exception:
        return
    if not job or not job.get("id"):
        return
    _BATCH_LAST_ID = job["id"]
    if job.get("status") != "running":
        BATCH_JOBS[job["id"]] = job
        return
    job["resumed"] = True
    job["current"] = None
    job["pending_tid"] = job.get("current_tid")
    BATCH_JOBS[job["id"]] = job
    print(f"[panel] 恢复通宵批次 {job['id']}：剩余 {len(job.get('queue') or [])} 个任务")
    threading.Thread(target=run_batch_job, args=(job["id"],), daemon=True).start()


@app.route("/api/batch", methods=["POST"])
def batch_start():
    data = request.json or {}
    raw = str(data.get("prompts") or "")
    prompts = [ln.strip()[:1000] for ln in raw.splitlines() if ln.strip()]
    if not prompts:
        return jsonify({"ok": False, "error": "请输入提示词（每行一条）"})
    if len(prompts) > BATCH_MAX_PROMPTS:
        return jsonify({"ok": False, "error": f"提示词最多 {BATCH_MAX_PROMPTS} 行"})
    per = max(1, min(4, int(data.get("per_prompt", 1) or 1)))
    total = len(prompts) * per
    if total > BATCH_MAX_TASKS:
        return jsonify({"ok": False, "error": f"共 {total} 个任务，超出单批上限 {BATCH_MAX_TASKS}（减少提示词或种子数）"})
    with BATCH_LOCK:
        if any(j.get("status") == "running" for j in BATCH_JOBS.values()):
            return jsonify({"ok": False, "error": "已有一个通宵批次在跑，可先停止或等它结束"})
    width = int(data.get("width", 864))
    height = int(data.get("height", 480))
    length = int(data.get("length", 124))
    if width * height * length > TOKEN_BUDGET and not bool(data.get("force", False)):
        return jsonify({"ok": False, "over_budget": True,
                        "error": f"该组合（{width}×{height} · {length}帧≈{length//24}秒）超出显存预算，"
                                 f"每个任务都会失败。请回创作台降低画质或时长。"})
    SAMPLERS = {"euler", "euler_ancestral", "heun", "dpm_2", "lms", "uni_pc"}
    SCHEDS = {"normal", "simple", "sgm_uniform", "beta", "karras"}
    prof = _model_profile(data.get("model"))
    if data.get("model") and prof["filename"] != DEFAULT_MODEL and prof["filename"] not in _installed_models():
        return jsonify({"ok": False, "error": f"模型未安装：{prof['label']}"})
    if prof["needs_gguf"] and _gguf_ready() is not True:
        return jsonify({"ok": False, "error": "ComfyUI 未检出 GGUF 加载组件，无法使用 GGUF 模型"})
    p = {"per_prompt": per,
         "width": width, "height": height, "length": length,
         "steps": int(prof["steps_fixed"]) if prof.get("steps_fixed") else max(4, min(20, int(data.get("steps", 8) or 8))),
         "turbo": bool(data.get("turbo", True)) and prof.get("turbo_compat", True),
         "sampler": data.get("sampler") if data.get("sampler") in SAMPLERS else "euler",
         "scheduler": data.get("scheduler") if data.get("scheduler") in SCHEDS else "normal",
         "autopost": data.get("autopost") if data.get("autopost") in ("rife", "x2", "rife_x2") else "",
         "model": prof["filename"], "model_kind": prof["kind"], "model_label": prof["label"]}
    queue = [[prompt, random.randint(1, 2 ** 31 - 1)] for prompt in prompts for _ in range(per)]
    per_sec = _estimate_seconds_per(p)
    job_id = f"batch_{int(time.time() * 1000):x}"
    job = {"id": job_id, "status": "running", "created": int(time.time()),
           "prompts_total": len(prompts), "total": total,
           "done": 0, "failed": 0, "videos": [], "failures": [],
           "queue": queue, "current": None, "current_tid": None,
           "params": p, "eta_seconds_per": per_sec,
           "eta_end": int(time.time() + total * per_sec),
           "conn_retry": 0, "cancel": False, "error": ""}
    with BATCH_LOCK:
        BATCH_JOBS[job_id] = job
    global _BATCH_LAST_ID
    _BATCH_LAST_ID = job_id
    _batch_save()
    threading.Thread(target=run_batch_job, args=(job_id,), daemon=True).start()
    return jsonify({"ok": True, "job_id": job_id, "total": total,
                    "eta_seconds_per": per_sec, "eta_end": job["eta_end"]})


@app.route("/api/batch/active")
def batch_active():
    with BATCH_LOCK:
        job = next((j for j in BATCH_JOBS.values() if j.get("status") == "running"), None) \
            or (BATCH_JOBS.get(_BATCH_LAST_ID) if _BATCH_LAST_ID else None)
    if not job:
        return jsonify({"ok": True, "job": None})
    p = dict(job.get("params") or {})
    p.pop("model_kind", None)
    return jsonify({"ok": True, "job": {
        "id": job.get("id"), "status": job.get("status"),
        "created": job.get("created"), "prompts_total": job.get("prompts_total"),
        "total": job.get("total"), "done": job.get("done"), "failed": job.get("failed"),
        "videos": job.get("videos"), "failures": job.get("failures"),
        "current": job.get("current"), "conn_retry": job.get("conn_retry"),
        "eta_end": job.get("eta_end"), "eta_seconds_per": job.get("eta_seconds_per"),
        "error": job.get("error"), "params": p}})


@app.route("/api/batch/<job_id>/cancel", methods=["POST"])
def batch_cancel(job_id):
    job = BATCH_JOBS.get(job_id)
    if not job or job.get("status") != "running":
        return jsonify({"ok": False, "error": "没有正在运行的通宵批次"})
    job["cancel"] = True
    return jsonify({"ok": True, "status": "stopping"})


if __name__ == "__main__":
    # 重启后遗留的未完成任务：补挂后台 watcher 自动收尾入库
    for _tid in list(TASKS):
        threading.Thread(target=_watch_task, args=(_tid,), daemon=True).start()
    _dl_restore()
    _batch_restore()
    print(f"[panel] http://0.0.0.0:{CONFIG['port']}  ComfyUI={COMFY}  output={OUTPUT}")
    print(f"[panel] 模型库目录={MODELS_DIR}  默认模型={DEFAULT_MODEL}  目录条目={len(CATALOG)} 个")
    if not CONFIG["access_password"]:
        print("[panel] 未设置访问口令（config.access_password），面板为无鉴权模式，请勿直接暴露公网")
    app.run(host="0.0.0.0", port=CONFIG["port"])
