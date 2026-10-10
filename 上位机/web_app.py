#!/usr/bin/env python3
"""Local PiPER ACT console. Run on the acquisition computer, one process only."""
import argparse
import atexit
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
from flask import g, Flask, jsonify, request, send_file

ROOT = Path(__file__).resolve().parent
COLLECTION_ROOT = ROOT.parent / "数据采集"
sys.path.insert(0, str(COLLECTION_ROOT))

if os.name == "nt":
    from piper_act_recorder_windows_dual_rgbd_v7 import HDF5EpisodeWriter, OrbbecCamera, RealSenseCamera, PiperReader, next_episode_path
    RGBD_RECORDER = True
else:
    from piper_act_recorder_Linux import HDF5EpisodeWriter, OrbbecCamera, PiperReader, next_episode_path
    RGBD_RECORDER = False
from dual_validation import validate_dual
from workflow import runtime_environment, PROJECT_ROOT, ManagedJob, TaskStore, execution_command, training_command
DEFAULTS = dict(can="0" if os.name == "nt" else "can0", can_backend="agx_cando", fps=20, wrist_width=640, wrist_height=480 if os.name == "nt" else 0,
                wrist_fps=30, camera_name="wrist", max_image_age=0.10,
                wrist_depth_min_mm=150, wrist_depth_max_mm=1500, wrist_depth_align=True,
                duration=0, startup_timeout=15, judge_flag=(os.name != "nt"), mode="hardware", episode_index=None)

DEFAULTS.update(external_width=640, external_height=480, external_fps=30, external_serial="",
                external_depth_min_mm=250, external_depth_max_mm=3000, max_camera_sync_delta=.05)

ENV_MODULES = ("piper_sdk", "pyorbbecsdk", "pyrealsense2", "pyAgxArm", "torch", "cv2", "h5py", "matplotlib")


def _environment_python(prefix):
    prefix = Path(prefix)
    return prefix / ("python.exe" if os.name == "nt" else "bin/python")


def discover_environments():
    """Return trusted Conda prefixes and a lightweight dependency inventory."""
    prefixes = []
    details = {}
    candidates = [os.environ.get("CONDA_EXE"), shutil.which("conda")]
    current = Path(sys.prefix).resolve()
    candidates += [current / "Scripts" / "conda.exe", current.parent.parent / "Scripts" / "conda.exe",
                   Path.home() / "anaconda3" / "Scripts" / "conda.exe",
                   Path.home() / "miniconda3" / "Scripts" / "conda.exe"]
    conda = next((str(p) for p in candidates if p and Path(p).is_file() and Path(p).suffix.lower() == ".exe"), None)
    if conda:
        try:
            result = subprocess.run([conda, "env", "list", "--json"], capture_output=True, text=True,
                                    encoding="utf-8", errors="replace", timeout=10, check=True)
            payload = json.loads(result.stdout)
            prefixes.extend(payload.get("envs", []))
            details = payload.get("envs_details", {})
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    prefixes.append(str(current))
    unique = []
    seen = set()
    for raw in prefixes:
        prefix = Path(raw).resolve()
        key = str(prefix).casefold()
        if key not in seen and _environment_python(prefix).is_file():
            seen.add(key)
            unique.append(prefix)

    probe = "import importlib.util,json; m=%r; print(json.dumps({x:bool(importlib.util.find_spec(x)) for x in m}))" % (ENV_MODULES,)
    rows = []
    for prefix in unique:
        python = _environment_python(prefix)
        modules = {name: False for name in ENV_MODULES}
        try:
            checked = subprocess.run([str(python), "-c", probe], capture_output=True, text=True,
                                     encoding="utf-8", errors="replace", timeout=8, check=True)
            modules.update(json.loads(checked.stdout.strip().splitlines()[-1]))
        except (OSError, ValueError, subprocess.SubprocessError, IndexError):
            pass
        info = details.get(str(prefix), {})
        name = info.get("name") or ("base" if prefix == prefix.parent else prefix.name)
        rows.append({"name": name, "prefix": str(prefix), "python": str(python),
                     "current": str(prefix).casefold() == str(current).casefold(),
                     "modules": modules,
                     "capture_ready": modules["piper_sdk"] and modules["pyorbbecsdk"] and modules["pyrealsense2"],
                     "training_ready": modules["torch"] and modules["cv2"] and modules["h5py"],
                     "execution_ready": all(modules[name] for name in ("pyAgxArm", "pyorbbecsdk", "pyrealsense2", "torch", "cv2", "h5py"))})
    return rows


def settings(data):
    c = {**DEFAULTS, **{k: v for k, v in data.items() if k in DEFAULTS}}
    for key, low, high in [("fps", 1, 120), ("wrist_width", 1, 4096),
                           ("wrist_height", 0, 2160), ("wrist_fps", 1, 120),
                           ("max_image_age", .001, 5), ("duration", 0, 86400),
                           ("startup_timeout", 1, 120), ("wrist_depth_min_mm", 1, 10000),
                           ("wrist_depth_max_mm", 1, 10000)]:
        v = float(c[key])
        if not math.isfinite(v) or not low <= v <= high:
            raise ValueError(f"{key} 必须在 {low}–{high} 之间")
        if key in ("wrist_width", "wrist_height", "wrist_fps"):
            if v != int(v):
                raise ValueError(f"{key} 必须是整数")
            v = int(v)
        c[key] = v
    if c["wrist_depth_min_mm"] >= c["wrist_depth_max_mm"]:
        raise ValueError("wrist_depth_min_mm 必须小于 wrist_depth_max_mm")
    for key in ("can", "can_backend", "camera_name"):
        if not isinstance(c[key], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", c[key]):
            raise ValueError(f"{key} 格式无效")
    if c["mode"] not in ("hardware", "demo") or not isinstance(c["judge_flag"], bool) or not isinstance(c["wrist_depth_align"], bool):
        raise ValueError("采集模式或 CAN 检查参数无效")
    if c["episode_index"] not in (None, ""):
        value = float(c["episode_index"])
        if not math.isfinite(value) or value < 0 or value != int(value):
            raise ValueError("episode_index 必须是非负整数，或留空自动编号")
        c["episode_index"] = int(value)
    else:
        c["episode_index"] = None
    for key, low, high in [("external_width",1,4096),("external_height",1,2160),("external_fps",1,120),
                           ("external_depth_min_mm",1,10000),("external_depth_max_mm",1,10000),("max_camera_sync_delta",.001,5)]:
        value=float(c[key])
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key} 超出范围")
        if key in ("external_width","external_height","external_fps"):
            if value != int(value): raise ValueError(f"{key} 必须为整数")
            value=int(value)
        c[key]=value
    if (c["wrist_width"],c["wrist_height"]) != (c["external_width"],c["external_height"]):
        raise ValueError("两路相机的宽高必须一致")
    if c["external_depth_min_mm"] >= c["external_depth_max_mm"]:
        raise ValueError("外部相机深度下限必须小于上限")
    if not isinstance(c["external_serial"],str) or not re.fullmatch(r"[A-Za-z0-9_-]{0,80}",c["external_serial"]):
        raise ValueError("RealSense 序列号格式无效")
    return c


class Recorder:
    def __init__(self, dataset):
        self.dataset = Path(dataset).resolve()
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None
        self.writer = None
        self.image = None
        self.config = DEFAULTS.copy()
        self.status = dict(state="disconnected", count=0, elapsed=0, skipped=0,
                           image_age_ms=None, qpos_hz=0, action_state="—",
                           error="", last_saved="", qpos=[], action=[])

    def folder(self, mode=None):
        return self.dataset / "demo" if (mode or self.config["mode"]) == "demo" else self.dataset

    def snapshot(self):
        with self.lock:
            s = dict(self.status, config=self.config.copy(), dataset_dir=str(self.folder()),
                     worker_active=bool(self.thread and self.thread.is_alive()))
            if self.writer:
                s["elapsed"] = round(time.monotonic() - self.started, 2)
            return s

    def connect(self, data):
        c = settings(data)
        with self.lock:
            if self.thread and self.thread.is_alive():
                raise ValueError("设备已连接或正在连接，请先断开")
            self.config = c
            self.stop_event.clear()
            self.image = None
            self.status.update(state="connecting", error="", count=0, elapsed=0,
                               skipped=0, image_age_ms=None, qpos_hz=0, qpos=[], action=[])
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()

    def start(self):
        with self.lock:
            if self.status["state"] != "ready":
                raise ValueError("请先连接设备并等待就绪")
            if self.status["image_age_ms"] is None or self.status["image_age_ms"] > self.config["max_image_age"] * 1000 or self.status["qpos_hz"] <= 0:
                raise ValueError("相机或从臂反馈已过期，请检查设备")
            c = self.config
            self.output = next_episode_path(self.folder(), c["episode_index"])
            if c["episode_index"] is not None and (self.output.exists() or self.output.with_suffix(".hdf5.partial").exists()):
                raise ValueError("指定 Episode 编号已存在，请换一个编号")
            # Reserve the name exclusively. A failed/in-progress recording is never
            # exposed as a complete episode, and existing data is never overwritten.
            while self.output.exists() or self.output.with_suffix(".hdf5.partial").exists():
                idx = int(self.output.stem.split("_")[-1]) + 1
                self.output = self.output.with_name(f"episode_{idx:06d}.hdf5")
            self.partial = self.output.with_suffix(".hdf5.partial")
            self.partial.touch(exist_ok=False)
            if RGBD_RECORDER:
                self.writer = HDF5EpisodeWriter(
                    self.partial, c["fps"], c["can"], c["can_backend"],
                    {key:c["wrist_"+key] for key in ("width","height","fps","depth_min_mm","depth_max_mm","depth_align")},
                    {key:c["external_"+key] for key in ("width","height","fps","depth_min_mm","depth_max_mm","serial")},
                )
            else:
                self.writer = HDF5EpisodeWriter(self.partial, c["fps"], c["can"], c["camera_name"],
                                                c["wrist_width"], c["wrist_height"], c["wrist_fps"])
            self.writer.file.attrs["sim"] = c["mode"] == "demo"
            self.writer.file.attrs["web_console"] = True
            if c["mode"] == "demo":
                self.writer.file.attrs["camera_backend"] = "synthetic_demo"
            self.started = self.next_tick = time.monotonic()
            self.status.update(state="recording", count=0, skipped=0, elapsed=0, error="")

    def save(self, error=""):
        with self.lock:
            if self.writer is None:
                return
            writer, self.writer = self.writer, None
            self.status["state"] = "saving"
            self.status["elapsed"] = round(time.monotonic() - self.started, 2)
            try:
                if error:
                    writer.file.attrs["recording_error"] = error
                writer.close()
                if self.output.exists():
                    raise FileExistsError("目标文件已存在，保留 .partial 文件")
                self.partial.rename(self.output)
                self.status.update(last_saved=self.output.name, count=writer.count, state="ready")
            except Exception:
                if writer.file.id.valid:
                    writer.file.close()
                self.status.update(state="error", error="保存失败；已保留 .partial 文件，请检查磁盘")
                raise

    def disconnect(self):
        self.stop_event.set()
        t = self.thread
        if t:
            t.join(timeout=5)
        if t and t.is_alive():
            return  # Remain connecting until SDK exits; never start a second reader.
        with self.lock:
            if self.status["state"] != "error":
                self.status["state"] = "disconnected"

    def demo_sample(self, now):
        t = now - self.demo_start
        action = np.array([.4 * np.sin(t * .6 + j * .4) for j in range(6)] + [.035 + .015 * np.sin(t)], np.float32)
        qpos = action + np.array([.02 * np.cos(t + j) for j in range(6)] + [.001 * np.cos(t)], np.float32)
        img = np.zeros((360, 640, 3), np.uint8)
        img[:] = (18, 30, 45)
        for x in range(0, 640, 40):
            cv2.line(img, (x, 0), (x, 360), (35, 54, 72), 1)
        for y in range(0, 360, 40):
            cv2.line(img, (0, y), (640, y), (35, 54, 72), 1)
        cv2.circle(img, (int(320 + 120 * np.sin(t)), 180), 38, (45, 212, 191), -1)
        cv2.putText(img, "SIMULATED / NOT TRAINING DATA", (28, 44), cv2.FONT_HERSHEY_SIMPLEX, .65, (240, 190, 80), 1)
        depth = np.full((360, 640), int(700 + 80 * np.sin(t)), dtype=np.uint16)
        scale = np.array([1e-3 * np.pi / 180] * 6 + [1e-6])
        return img, depth, now, dict(ready=True, action=action, qpos=qpos,
            action_raw=(action / scale).astype(np.int32), qpos_raw=(qpos / scale).astype(np.int32),
            qpos_hz=100, action_updated=True, action_sdk_stamp=now,
            qpos_sdk_stamp=now, action_age_s=0)

    def run(self):
        camera = external = robot = None
        error = ""
        try:
            c = self.config
            if c["mode"] == "hardware":
                if RGBD_RECORDER:
                    camera = OrbbecCamera(c["wrist_width"], c["wrist_height"], c["wrist_fps"], c["wrist_depth_align"], c["wrist_depth_min_mm"], c["wrist_depth_max_mm"])
                    external = RealSenseCamera(c["external_width"],c["external_height"],c["external_fps"],
                                               c["external_depth_min_mm"],c["external_depth_max_mm"],c["external_serial"])
                    robot = PiperReader(c["can"], c["can_backend"], c["judge_flag"])
                else:
                    camera = OrbbecCamera(c["wrist_width"], c["wrist_height"], c["wrist_fps"])
                    robot = PiperReader(c["can"], c["judge_flag"])
            self.demo_start = time.monotonic()
            deadline = self.demo_start + c["startup_timeout"]
            preview_tick = 0
            while not self.stop_event.is_set():
                now = time.monotonic()
                if camera:
                    pack = camera.get_latest()
                    r = robot.read()
                    if RGBD_RECORDER:
                        img, depth, stamp = pack if pack is not None else (None, None, 0)
                    else:
                        img, stamp = pack if pack is not None else (None, 0)
                        depth = None
                else:
                    img, depth, stamp, r = self.demo_sample(now)
                if external:
                    external_pack = external.get_latest()
                    ext_img, ext_depth, ext_stamp = external_pack if external_pack is not None else (None,None,0)
                else:
                    ext_img, ext_depth, ext_stamp = img, depth, stamp
                wrist_age = max(0, now-stamp)
                external_age = max(0, now-ext_stamp)
                sync_delta = abs(stamp-ext_stamp)
                age = max(wrist_age,external_age)
                valid = img is not None and ext_img is not None and sync_delta <= c["max_camera_sync_delta"] and age <= c["max_image_age"] and r["ready"] and r["qpos_hz"] > 0
                with self.lock:
                    self.status.update(image_age_ms=round(age * 1000, 1) if img is not None else None,
                        qpos_hz=r["qpos_hz"], action_state="NEW" if r["action_updated"] else "HOLD",
                        qpos=r["qpos"].tolist(), action=r["action"].tolist() if r["ready"] else [])
                    if img is not None and now >= preview_tick:
                        success, jpeg = cv2.imencode(".jpg", cv2.cvtColor(np.concatenate([img,cv2.resize(ext_img,(img.shape[1],img.shape[0]))],axis=1) if ext_img is not None else img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if success:
                            self.image = jpeg.tobytes()
                        preview_tick = now + 1 / 15
                    if self.status["state"] == "connecting":
                        if valid:
                            self.status["state"] = "ready"
                        elif now > deadline:
                            raise RuntimeError("初始化超时：检查相机、CAN 和从臂反馈，并移动一次主臂以初始化 action")
                    if self.writer and c["duration"] and now - self.started >= c["duration"]:
                        self.save()
                    if self.writer and now >= self.next_tick:
                        self.next_tick = max(self.next_tick + 1 / c["fps"], now + .001)
                        if not valid:
                            self.status["skipped"] += 1
                        else:
                            sample = dict(image=img, qpos=r["qpos"], action=r["action"],
                                qpos_raw=r["qpos_raw"], action_raw=r["action_raw"], sample_monotonic=now,
                                camera_host_stamp=stamp, action_sdk_stamp=r["action_sdk_stamp"],
                                qpos_sdk_stamp=r["qpos_sdk_stamp"], image_age_s=age,
                                action_age_s=r["action_age_s"], action_updated=r["action_updated"])
                            if RGBD_RECORDER:
                                sample.update(wrist_image=img,wrist_depth=depth,external_image=ext_img,external_depth=ext_depth,
                                              wrist_camera_host_stamp=stamp,external_camera_host_stamp=ext_stamp,
                                              wrist_image_age_s=wrist_age,external_image_age_s=external_age,camera_sync_delta_s=sync_delta)
                            self.writer.add(sample)
                            self.status["count"] = self.writer.count + len(self.writer.buffers)
                self.stop_event.wait(.005 if camera else .015)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            if "MFCreateDeviceSource" in detail and "0x80070005" in detail:
                error = ("Windows 拒绝访问 DaBai 彩色相机（0x80070005）。请关闭相机/会议/浏览器视频程序，"
                         "将相机 USB 线拔下 5 秒后重新插入 USB 3.x 接口，再点击连接。若仍失败，请在设备管理器中重启 Dabai DC1。")
            else:
                error = detail
        finally:
            try:
                self.save(error)
            except Exception as exc:
                error = f"保存失败，保留 .partial 文件: {exc}"
            for device in (robot, camera, external):
                if device:
                    try:
                        device.close()
                    except Exception:
                        pass
            with self.lock:
                self.status.update(state="error" if error else "disconnected", error=error)
                self.image = None


def camera_dataset(f):
    group = f.get("observations/images")
    if not isinstance(group, h5py.Group) or not len(group):
        raise ValueError("缺少相机图像")
    name = f.attrs.get("camera_name", next(iter(group)))
    if isinstance(name, bytes):
        name = name.decode()
    return group[name] if name in group else group[next(iter(group))]


def metadata(path):
    with h5py.File(path, "r") as f:
        ds = camera_dataset(f)
        n = int(ds.shape[0])
        hz = float(f.attrs.get("sample_rate_hz", 20))
        hz = hz if math.isfinite(hz) and hz > 0 else 20
        return dict(name=path.name, frames=n, fps=hz, duration=round(n / hz, 2),
                    size_mb=round(path.stat().st_size / 1048576, 2), sim=bool(f.attrs.get("sim", False)),
                    recording_error=str(f.attrs.get("recording_error", "")))


def create_app(dataset=None, task_store_path=None):
    dataset = Path(dataset or PROJECT_ROOT / "Dataset" / "stand_bottle_rgbd").resolve()
    app = Flask(__name__, static_folder=str(ROOT / "frontend" / "dist"), static_url_path="/")
    app.config["MAX_CONTENT_LENGTH"] = 16384
    recorder = Recorder(dataset)
    store_path = Path(task_store_path) if task_store_path else (dataset / ".tasks.json" if dataset != (PROJECT_ROOT / "Dataset" / "stand_bottle_rgbd").resolve() else PROJECT_ROOT / ".piper_tasks.json")
    tasks = TaskStore(store_path, dataset)
    selected_task = tasks.current(required=False)
    if selected_task:
        recorder.dataset = Path(selected_task["dataset_dir"])
    trainer = ManagedJob("training")
    executor = ManagedJob("execution")
    environment_lock = threading.Lock()
    environment_cache = None
    runtime_python = str(Path(sys.executable).resolve())
    app.extensions["recorder"] = recorder
    app.extensions["tasks"] = tasks
    app.extensions["trainer"] = trainer
    app.extensions["executor"] = executor
    validation_lock = threading.Lock()
    operation_lock = threading.RLock()
    atexit.register(recorder.disconnect)
    atexit.register(trainer.terminate)
    atexit.register(executor.terminate)

    @app.before_request
    def local_requests():
        if request.method == "POST":
            operation_lock.acquire()
            g.operation_locked = True
        # No CORS. A custom header prevents cross-site form/CSRF device actions.
        if request.method == "POST" and request.headers.get("X-Console-Request") != "1":
            return jsonify(error="请求来源无效"), 403
        if request.headers.get("Origin") and request.headers["Origin"] != request.host_url.rstrip("/"):
            return jsonify(error="仅允许同源访问"), 403

    @app.teardown_request
    def release_operation(error=None):
        if getattr(g, "operation_locked", False):
            g.operation_locked = False
            operation_lock.release()

    @app.errorhandler(Exception)
    def error(exc):
        from werkzeug.exceptions import HTTPException
        code = exc.code if isinstance(exc, HTTPException) else 400 if isinstance(exc, (ValueError, FileNotFoundError, OSError)) else 500
        return jsonify(error=str(exc)), code

    def episode(name):
        if not re.fullmatch(r"episode_\d+\.hdf5", name):
            raise ValueError("Episode 文件名无效")
        mode = request.args.get("mode", "hardware")
        if mode not in ("hardware", "demo"):
            raise ValueError("数据集模式无效")
        folder = recorder.folder(mode).resolve()
        path = (folder / name).resolve()
        if path.parent != folder or not path.is_file():
            raise FileNotFoundError("未找到 Episode")
        return path

    @app.get("/")
    def index():
        return app.send_static_file("index.html")

    @app.get("/api/health")
    def console_health():
        return jsonify(application="piper-act-console", root=str(ROOT), pid=os.getpid())

    @app.get("/api/status")
    def status():
        return jsonify(recorder.snapshot())

    def require_idle():
        if recorder.snapshot()["worker_active"] or trainer.active() or executor.active():
            raise ValueError("请先停止并断开当前阶段，再切换任务")

    def environments():
        nonlocal environment_cache
        with environment_lock:
            if environment_cache is None:
                environment_cache = discover_environments()
            selected = str(Path(runtime_python).resolve()).casefold()
            return [dict(row, selected=str(Path(row["python"]).resolve()).casefold() == selected)
                    for row in environment_cache]

    def selected_environment():
        return next((row for row in environments() if row["selected"]), None)

    @app.get("/api/environments")
    def environment_list():
        return jsonify(environments())

    @app.post("/api/environments/select")
    def environment_select():
        nonlocal runtime_python
        require_idle()
        name = str((request.get_json() or {}).get("name", ""))
        row = next((item for item in environments() if item["name"] == name), None)
        if row is None:
            raise ValueError("未找到所选 Conda 环境")
        runtime_python = row["python"]
        return jsonify(next(item for item in environments() if item["selected"]))

    @app.get("/api/tasks")
    def task_list():
        with operation_lock:
            if not (recorder.snapshot()["worker_active"] or trainer.active() or executor.active()):
                tasks.sync()
                current = tasks.current(required=False)
                if current:
                    recorder.dataset = Path(current["dataset_dir"])
            return jsonify(tasks.list())

    @app.post("/api/tasks")
    def task_create():
        require_idle()
        task = tasks.create((request.get_json() or {}).get("name"))
        recorder.dataset = Path(task["dataset_dir"])
        return jsonify(task), 201

    @app.post("/api/tasks/<task_id>/select")
    def task_select(task_id):
        require_idle()
        task = tasks.select(task_id)
        recorder.dataset = Path(task["dataset_dir"])
        return jsonify(task)

    @app.post("/api/tasks/<task_id>/rename")
    def task_rename(task_id):
        require_idle()
        task = tasks.rename(task_id, (request.get_json() or {}).get("name"))
        if task["id"] == tasks.current()["id"]:
            recorder.dataset = Path(task["dataset_dir"])
        return jsonify(task)

    @app.get("/api/tasks/<task_id>/delete-plan")
    def task_delete_plan(task_id):
        require_idle()
        return jsonify(tasks.deletion_plan(task_id))

    @app.post("/api/tasks/<task_id>/delete")
    def task_delete(task_id):
        require_idle()
        data = request.get_json() or {}
        result = tasks.delete(task_id, data.get("confirmation"), data.get("token"))
        current = tasks.current(required=False)
        recorder.dataset = Path(current["dataset_dir"]) if current else dataset
        recorder.status.update(last_saved="", count=0, elapsed=0)
        return jsonify(result)

    @app.post("/api/tasks/<task_id>/paths")
    def task_paths(task_id):
        require_idle()
        task = tasks.configure_paths(task_id, request.get_json() or {})
        if task["id"] == tasks.current()["id"]:
            recorder.dataset = Path(task["dataset_dir"])
        return jsonify(task)

    @app.get("/api/workflow")
    def workflow_status():
        env = selected_environment()
        return jsonify(task=tasks.current(required=False), training=trainer.snapshot(), execution=executor.snapshot(),
                       environment={"name": env["name"], "python": env["python"]} if env else None)

    @app.post("/api/training/start")
    def training_start():
        if recorder.snapshot()["worker_active"] or executor.active() or trainer.active():
            raise ValueError("请先停止并断开其他阶段")
        task = tasks.current()
        env = selected_environment()
        if not env or not env["training_ready"]:
            raise ValueError("所选 Conda 环境缺少训练依赖：torch、OpenCV 或 h5py")
        command = training_command(task, request.get_json() or {}, runtime_python)
        env = {**runtime_environment(runtime_python), "MPLBACKEND": "Agg"}
        trainer.start(command, task["id"], PROJECT_ROOT / "act", env=env)
        return jsonify(trainer.snapshot()), 202

    @app.post("/api/training/stop")
    def training_stop():
        trainer.stop()
        return jsonify(trainer.snapshot()), 202

    @app.post("/api/execution/start")
    def execution_start():
        if recorder.snapshot()["worker_active"] or trainer.active() or executor.active():
            raise ValueError("请先停止并断开其他阶段")
        task = tasks.current()
        runtime_dir = PROJECT_ROOT / ".piper_runtime"
        env = selected_environment()
        if not env or not env["execution_ready"]:
            raise ValueError("所选 Conda 环境缺少执行依赖：pyAgxArm、pyorbbecsdk、pyrealsense2、torch、OpenCV 或 h5py")
        command, stop_file, preview_file = execution_command(task, request.get_json() or {}, runtime_dir, runtime_python)
        env = runtime_environment(runtime_python)
        executor.start(command, task["id"], PROJECT_ROOT / "act", env=env, stop_file=stop_file, preview_file=preview_file)
        return jsonify(executor.snapshot()), 202

    @app.post("/api/execution/stop")
    def execution_stop():
        executor.stop()
        return jsonify(executor.snapshot()), 202

    @app.get("/api/execution/preview.jpg")
    def execution_preview():
        path = executor.preview_file
        if not path or not path.is_file():
            return "", 204
        try:
            image = path.read_bytes()
        except (FileNotFoundError, PermissionError):
            return "", 204
        return send_file(io.BytesIO(image), mimetype="image/jpeg", max_age=0)

    @app.post("/api/connect")
    def connect():
        tasks.current()
        data = request.get_json() or {}
        if data.get("mode", DEFAULTS["mode"]) == "hardware":
            env = selected_environment()
            if not env or not env["capture_ready"]:
                raise ValueError("所选 Conda 环境缺少采集依赖：piper_sdk 或 pyorbbecsdk")
            if str(Path(env["python"]).resolve()).casefold() != str(Path(sys.executable).resolve()).casefold():
                raise ValueError(f"采集服务当前运行在 {Path(sys.prefix).name} 环境，请用所选的 {env['name']} 环境启动网页服务")
        recorder.connect(data)
        return jsonify(recorder.snapshot()), 202

    @app.post("/api/start")
    def start():
        tasks.current()
        recorder.start()
        return jsonify(recorder.snapshot())

    @app.post("/api/stop")
    def stop():
        recorder.save()
        return jsonify(recorder.snapshot())

    @app.post("/api/disconnect")
    def disconnect():
        recorder.disconnect()
        return jsonify(recorder.snapshot())

    @app.get("/api/preview.jpg")
    def preview():
        with recorder.lock:
            image = recorder.image
        if image is None:
            return "", 204
        return send_file(io.BytesIO(image), mimetype="image/jpeg", max_age=0)

    @app.get("/api/episodes")
    def episodes():
        mode = request.args.get("mode", "hardware")
        if mode not in ("hardware", "demo"):
            raise ValueError("数据集模式无效")
        rows = []
        for p in sorted(recorder.folder(mode).glob("episode_*.hdf5"), reverse=True):
            try:
                rows.append(metadata(p))
            except Exception as exc:
                rows.append(dict(name=p.name, error=str(exc), frames=0, duration=0, size_mb=0))
        return jsonify(rows)

    @app.get("/api/episodes/<name>/data")
    def data(name):
        path = episode(name)
        meta = metadata(path)
        with h5py.File(path, "r") as f:
            q, a = f["observations/qpos"], f["action"]
            if q.ndim != 2 or a.ndim != 2 or q.shape != a.shape or q.shape[1] != 7 or not len(q):
                raise ValueError("qpos/action 必须为相同长度的非空 [T,7] 数组")
            ids = np.unique(np.linspace(0, len(q) - 1, min(1500, len(q)), dtype=int))
            ts = f.get("timestamps/sample_monotonic")
            x = (ts[ids] - ts[0]) if ts is not None and ts.shape == (len(q),) else ids / meta["fps"]
            def clean(values):
                arr = np.asarray(values, dtype=object)
                arr[~np.isfinite(np.asarray(values, dtype=float))] = None
                return arr.tolist()
            return jsonify(**meta, indices=ids.tolist(), time=clean(x), qpos=clean(q[ids]), action=clean(a[ids]))

    @app.get("/api/episodes/<name>/frame/<int:frame>")
    def frame(name, frame):
        with h5py.File(episode(name), "r") as f:
            ds = camera_dataset(f)
            if ds.ndim != 4 or ds.shape[-1] != 3 or frame >= len(ds):
                raise ValueError("图像格式或帧索引无效")
            images = [f["observations/images"][cam][frame] for cam in ("wrist","external") if cam in f["observations/images"]]
            rgb = np.concatenate([cv2.resize(im,(ds.shape[2],ds.shape[1])) for im in images],axis=1) if len(images)==2 else ds[frame]
            success, jpg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if not success:
                raise ValueError("图像编码失败")
        return send_file(io.BytesIO(jpg.tobytes()), mimetype="image/jpeg")

    @app.get("/api/episodes/<name>/download")
    def download(name):
        return send_file(episode(name), as_attachment=True)

    @app.post("/api/episodes/<name>/validate")
    def validate(name):
        path = episode(name)
        opts = request.get_json() or {}
        age = float(opts.get("max_image_age", .1))
        if not math.isfinite(age) or not .001 <= age <= 5:
            raise ValueError("图像延迟阈值必须在 0.001–5 秒之间")
        if not validation_lock.acquire(blocking=False):
            return jsonify(error="已有验证任务正在运行"), 409
        try:
            validation_env = next((row for row in environments()
                                   if row["name"] == "base" and row["modules"]["h5py"] and row["modules"]["matplotlib"]), None)
            validation_python = validation_env["python"] if validation_env else sys.executable
            args = [validation_python, str(COLLECTION_ROOT / "validate_act_episode.py"), str(path), "--max-image-age", str(age)]
            if opts.get("plots"):
                args.append("--plots")
            result = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180,
                                    env={**runtime_environment(validation_python), "MPLBACKEND": "Agg"})
            report = result.stdout + result.stderr
            with h5py.File(path, "r") as f:
                recording_error = str(f.attrs.get("recording_error", ""))
            dual_status, dual_report = validate_dual(path, age, float(opts.get("max_camera_sync_delta", .05)))
            report += dual_report
            if recording_error:
                report += "\n[WARN] Recording interrupted: " + recording_error
            status = "fail" if result.returncode else "review" if "[WARN]" in report else "pass"
            if dual_status == "fail": status = "fail"
            elif dual_status == "review" and status == "pass": status = "review"
            plots = [f"{j}_qpos_action.png" for j in ["j1", "j2", "j3", "j4", "j5", "j6", "gripper"]] if opts.get("plots") and result.returncode == 0 else []
            return jsonify(status=status, report=report, plots=plots)
        finally:
            validation_lock.release()

    @app.get("/api/episodes/<name>/plots/<plot>")
    def plot(name, plot):
        p = episode(name)
        if not re.fullmatch(r"(j[1-6]|gripper)_qpos_action\.png", plot):
            raise ValueError("曲线文件名无效")
        return send_file(p.parent / (p.stem + "_validation") / plot)

    return app


if __name__ == "__main__":
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default=str(PROJECT_ROOT / "Dataset" / "stand_bottle_rgbd"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    app = create_app(args.dataset_dir)
    try:
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    finally:
        app.extensions["recorder"].disconnect()
        app.extensions["trainer"].terminate()
        app.extensions["executor"].terminate()
