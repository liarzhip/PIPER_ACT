"""Task registry and subprocess bridges for the local PiPER ACT web console."""
from __future__ import annotations

import json
import hashlib
import shutil
import stat
import math
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = PROJECT_ROOT / "act" / "train_piper_act_dual_rgbd_global_v7.py"
RUN_SCRIPT = PROJECT_ROOT / "act" / "run_piper_act_windows_smooth_cpv_v7_dual_rgbd_global.py"


def _safe_number(data, key, default, low, high, integer=False):
    raw = data.get(key, default)
    value = int(raw) if integer else float(raw)
    if not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{key} 必须在 {low}–{high} 之间")
    return value


def _safe_token(data, key, default, choices=None):
    value = str(data.get(key, default))
    if choices and value not in choices:
        raise ValueError(f"{key} 参数无效")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", value):
        raise ValueError(f"{key} 格式无效")
    return value


class TaskStore:
    def __init__(self, path, initial_dataset):
        self.path = Path(path)
        self.lock = threading.RLock()
        self.initial_dataset = Path(initial_dataset).resolve()
        if self.initial_dataset.is_relative_to(PROJECT_ROOT):
            self.dataset_root = PROJECT_ROOT / "Dataset"
            self.checkpoint_root = PROJECT_ROOT / "checkpoints"
        else:
            self.dataset_root = self.initial_dataset / "tasks"
            self.checkpoint_root = self.initial_dataset / "checkpoints"
        self.data = self._load()
        self.sync()

    def _load(self):
        if self.path.is_file():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data.get("tasks"), list):
                return data
            raise ValueError("任务注册表格式无效")
        # Outside the production Dataset tree, retain explicitly supplied datasets.
        if self.initial_dataset.is_dir() and self.initial_dataset.parent != self.dataset_root:
            task = self._task("默认任务", self.initial_dataset, self._default_checkpoint())
            return {"selected": task["id"], "tasks": [task]}
        return {"selected": None, "tasks": []}

    @staticmethod
    def _is_link(path):
        try:
            return path.is_symlink() or bool(getattr(path.lstat(), "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)
        except FileNotFoundError:
            return False

    def sync(self):
        """Reconcile immediate Dataset folders; preserve IDs for unchanged paths."""
        with self.lock:
            before = json.dumps(self.data, sort_keys=True)
            known = {str(Path(t["dataset_dir"]).resolve()).casefold(): t for t in self.data["tasks"]}
            found = []
            if self.dataset_root.is_dir():
                for folder in sorted(self.dataset_root.iterdir(), key=lambda p: p.name.casefold()):
                    if folder.name.startswith('.') or not folder.is_dir() or self._is_link(folder):
                        continue
                    key = str(folder.resolve()).casefold()
                    task = known.pop(key, None)
                    if task is None:
                        task = self._task(folder.name, folder, self.checkpoint_root / folder.name)
                    found.append(task)
            # Preserve explicitly configured external datasets only while they exist.
            for task in known.values():
                path = Path(task["dataset_dir"])
                if path.is_dir() and not path.resolve().is_relative_to(self.dataset_root.resolve()) and not self._is_link(path):
                    found.append(task)
            self.data["tasks"] = found
            if not any(t["id"] == self.data.get("selected") for t in found):
                self.data["selected"] = found[0]["id"] if found else None
            if before != json.dumps(self.data, sort_keys=True) or not self.path.exists():
                self._save()

    def _default_checkpoint(self):
        preferred = self.checkpoint_root / self.initial_dataset.name
        if (preferred / "policy_best.ckpt").is_file():
            return preferred
        candidates = [p for p in self.checkpoint_root.glob(f"{self.initial_dataset.name}*") if (p / "policy_best.ckpt").is_file()]
        return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else preferred

    def _save(self, data=None):
        data = data or self.data
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(self.path)

    def _task(self, name, dataset_dir, ckpt_dir):
        return {
            "id": uuid.uuid4().hex[:10],
            "name": name,
            "dataset_dir": str(Path(dataset_dir).resolve()),
            "ckpt_dir": str(Path(ckpt_dir).resolve()),
            "created_at": time.strftime("%Y-%m-%d %H:%M"),
        }

    def list(self):
        with self.lock:
            rows = []
            for task in self.data["tasks"]:
                row = dict(task, selected=task["id"] == self.data["selected"])
                dataset = Path(task["dataset_dir"])
                checkpoint = Path(task["ckpt_dir"])
                row["episodes"] = len(list(dataset.glob("episode_*.hdf5"))) if dataset.is_dir() else 0
                row["trained"] = (checkpoint / "policy_best.ckpt").is_file()
                row["dataset_folder"] = dataset.name
                match = re.search(r"_([0-9a-f]{6})$", dataset.name, re.IGNORECASE)
                row["folder_suffix"] = match.group(1) if match else ""
                rows.append(row)
            return rows

    def current(self, required=True):
        with self.lock:
            for task in self.data["tasks"]:
                if task["id"] == self.data["selected"]:
                    return dict(task)
        if required:
            raise ValueError("暂无任务，请先创建或选择 Dataset 中的任务")
        return None

    def create(self, name):
        name = self._validate_name(name)
        with self.lock:
            if any(t["name"].casefold() == name.casefold() for t in self.data["tasks"]):
                raise ValueError("已有同名任务")
            slug = re.sub(r"[^\w-]+", "_", name, flags=re.UNICODE).strip("_") or "task"
            suffix = uuid.uuid4().hex[:6]
            task = self._task(name, self.dataset_root / f"{slug}_{suffix}", self.checkpoint_root / f"{slug}_{suffix}")
            Path(task["dataset_dir"]).mkdir(parents=True, exist_ok=False)
            Path(task["ckpt_dir"]).mkdir(parents=True, exist_ok=True)
            self.data["tasks"].append(task)
            self.data["selected"] = task["id"]
            self._save()
            return dict(task)

    @staticmethod
    def _validate_name(name):
        name = str(name or "").strip()
        if not 1 <= len(name) <= 40 or any(c in '<>:"/\\|?*' for c in name):
            raise ValueError("任务名称应为 1–40 个字符，且不能包含路径特殊字符")
        return name

    def rename(self, task_id, name):
        name = self._validate_name(name)
        with self.lock:
            task = next((t for t in self.data["tasks"] if t["id"] == task_id), None)
            if task is None:
                raise FileNotFoundError("未找到任务")
            if any(t["id"] != task_id and t["name"].casefold() == name.casefold() for t in self.data["tasks"]):
                raise ValueError("已有同名任务")
            dataset_source = Path(task["dataset_dir"]).resolve()
            checkpoint_source = Path(task["ckpt_dir"]).resolve()
            dataset_root = self.dataset_root.resolve()
            checkpoint_root = self.checkpoint_root.resolve()
            if not dataset_source.is_relative_to(dataset_root) or not checkpoint_source.is_relative_to(checkpoint_root):
                raise ValueError("任务目录超出工作区，无法安全重命名")
            if not dataset_source.is_dir():
                raise FileNotFoundError("数据集或模型文件夹不存在，无法同步重命名")

            slug = re.sub(r"[^\w-]+", "_", name, flags=re.UNICODE).strip("_") or "task"
            match = re.search(r"_([0-9a-f]{6})$", dataset_source.name, re.IGNORECASE)
            folder_name = f"{slug}_{match.group(1)}" if match else slug
            dataset_target = dataset_root / folder_name
            checkpoint_target = checkpoint_root / folder_name

            def same_path(left, right):
                return str(left).casefold() == str(right).casefold()

            for source, target in ((dataset_source, dataset_target), (checkpoint_source, checkpoint_target)):
                if target.exists() and not same_path(source, target):
                    raise FileExistsError(f"目标文件夹已存在：{target.name}")

            moved = []

            def move(source, target):
                if str(source) == str(target):
                    return
                if same_path(source, target):
                    temporary = source.with_name(f".__rename_{uuid.uuid4().hex}")
                    source.rename(temporary)
                    try:
                        temporary.rename(target)
                    except Exception:
                        temporary.rename(source)
                        raise
                else:
                    source.rename(target)
                moved.append((target, source))

            old_task = dict(task)
            try:
                move(dataset_source, dataset_target)
                if checkpoint_source.is_dir():
                    move(checkpoint_source, checkpoint_target)
                task.update(name=name, dataset_dir=str(dataset_target), ckpt_dir=str(checkpoint_target))
                self._save()
            except Exception:
                task.update(old_task)
                for target, source in reversed(moved):
                    if target.exists() and not source.exists():
                        target.rename(source)
                raise
            return dict(task)

    def deletion_plan(self, task_id):
        with self.lock:
            task = next((t for t in self.data["tasks"] if t["id"] == task_id), None)
            if task is None:
                raise FileNotFoundError("未找到任务，请刷新列表")
            paths = []
            for key, root in (("dataset_dir", self.dataset_root), ("ckpt_dir", self.checkpoint_root)):
                raw = Path(task[key])
                resolved = raw.resolve()
                # Only direct child folders of the intended roots are deletable.
                if self._is_link(root) or self._is_link(raw) or resolved.parent != root.resolve():
                    raise ValueError("删除范围仅限 Dataset 和 checkpoints 的直接子目录；外部路径或链接不能删除")
                if raw.exists() and not raw.is_dir():
                    raise ValueError("任务路径不是文件夹")
                for other in self.data["tasks"]:
                    if other["id"] == task_id:
                        continue
                    for field in ("dataset_dir", "ckpt_dir"):
                        other_path = Path(other[field]).resolve()
                        if other_path == resolved or other_path.is_relative_to(resolved) or resolved.is_relative_to(other_path):
                            raise ValueError("目录被其他任务共用，不能删除")
                if raw.exists():
                    for base, dirs, files in os.walk(raw, followlinks=False):
                        if any(self._is_link(Path(base) / n) for n in dirs + files):
                            raise ValueError("目录内含符号链接或目录联接，不能自动删除")
                paths.append({"kind": key, "path": str(resolved), "exists": raw.exists()})
            token = hashlib.sha256(json.dumps([task["id"],task["name"],paths], sort_keys=True).encode()).hexdigest()
            return {"id": task_id, "name": task["name"], "paths": paths, "token": token}

    def delete(self, task_id, confirmation, token):
        with self.lock:
            plan = self.deletion_plan(task_id)
            if confirmation != plan["name"] or token != plan["token"]:
                raise ValueError("请重新确认任务名称与删除目录")
            old = json.loads(json.dumps(self.data))
            staged = []
            try:
                for item in plan["paths"]:
                    if not item["exists"]:
                        continue
                    source = Path(item["path"])
                    target = source.parent / (".task-delete-" + uuid.uuid4().hex)
                    source.rename(target)
                    staged.append((source, target))
                self.data["tasks"] = [t for t in self.data["tasks"] if t["id"] != task_id]
                if self.data["selected"] == task_id:
                    self.data["selected"] = self.data["tasks"][0]["id"] if self.data["tasks"] else None
                self._save()
            except Exception:
                self.data = old
                for source, target in reversed(staged):
                    target.rename(source)
                raise
            warnings = []
            for source, target in staged:
                # Validate final absolute targets again before recursive deletion.
                if target.resolve().parent != source.parent.resolve() or not target.name.startswith(".task-delete-"):
                    raise ValueError("删除临时目录校验失败")
                try:
                    shutil.rmtree(target)
                except OSError as exc:
                    warnings.append(f"任务已移除，但文件清理未完成：{target} ({exc})")
            return {"deleted": task_id, "warnings": warnings}

    def configure_paths(self, task_id, data):
        with self.lock:
            task = next((t for t in self.data["tasks"] if t["id"] == task_id), None)
            if task is None:
                raise FileNotFoundError("未找到任务")
            changes = {}
            for key in ("dataset_dir", "ckpt_dir"):
                if key not in data:
                    continue
                raw = str(data[key]).strip()
                path = Path(raw).expanduser()
                if not raw or not path.is_absolute():
                    raise ValueError(f"{key} 请填写绝对路径")
                path = path.resolve()
                if path.exists() and not path.is_dir():
                    raise ValueError(f"{key} 不是目录")
                changes[key] = str(path)
            updated = dict(task, **changes)
            if updated["dataset_dir"] == updated["ckpt_dir"]:
                raise ValueError("数据目录和模型目录不能相同")
            previous = dict(task)
            task.update(changes)
            try:
                self._save()
            except Exception:
                task.update(previous)
                raise
            return dict(task)

    def select(self, task_id):
        with self.lock:
            if not any(t["id"] == task_id for t in self.data["tasks"]):
                raise FileNotFoundError("未找到任务")
            self.data["selected"] = task_id
            self._save()
            return self.current()


class ManagedJob:
    def __init__(self, kind):
        self.kind = kind
        self.lock = threading.RLock()
        self.process = None
        self.thread = None
        self.stop_file = None
        self.preview_file = None
        self.reset()

    def reset(self):
        self.state = "idle"
        self.task_id = None
        self.started_at = None
        self.ended_at = None
        self.returncode = None
        self.error = ""
        self.logs = deque(maxlen=240)
        self.metrics = []
        self.latest = {}
        self.command = []

    def active(self):
        with self.lock:
            return bool(self.process and self.process.poll() is None)

    def snapshot(self):
        with self.lock:
            elapsed = 0
            if self.started_at:
                elapsed = (self.ended_at or time.time()) - self.started_at
            return {
                "state": self.state,
                "task_id": self.task_id,
                "pid": self.process.pid if self.active() else None,
                "elapsed": round(elapsed, 1),
                "returncode": self.returncode,
                "error": self.error,
                "logs": list(self.logs),
                "metrics": list(self.metrics),
                "latest": dict(self.latest),
                "has_preview": bool(self.preview_file and self.preview_file.is_file()),
            }

    def start(self, command, task_id, cwd, env=None, stop_file=None, preview_file=None):
        with self.lock:
            if self.active():
                raise ValueError("已有任务正在运行")
            self.reset()
            self.state = "starting"
            self.task_id = task_id
            self.started_at = time.time()
            self.stop_file = Path(stop_file) if stop_file else None
            self.preview_file = Path(preview_file) if preview_file else None
            for path in (self.stop_file, self.preview_file):
                if path and path.exists():
                    path.unlink()
            flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            self.command = [str(x) for x in command]
            self.process = subprocess.Popen(
                self.command, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
                bufsize=1, creationflags=flags, env=env,
            )
            self.thread = threading.Thread(target=self._watch, daemon=True)
            self.thread.start()

    def _watch(self):
        proc = self.process
        try:
            for raw in proc.stdout:
                line = raw.rstrip()
                if not line:
                    continue
                with self.lock:
                    self.logs.append(line)
                    if self.state == "starting":
                        self.state = "running"
                    self._parse(line)
            code = proc.wait()
            with self.lock:
                self.returncode = code
                self.ended_at = time.time()
                if self.state == "stopping":
                    self.state = "stopped"
                elif code == 0:
                    self.state = "completed"
                else:
                    self.state = "error"
                    self.error = self.logs[-1] if self.logs else f"进程退出码 {code}"
        except Exception as exc:
            with self.lock:
                self.ended_at = time.time()
                self.state = "error"
                self.error = f"{type(exc).__name__}: {exc}"

    def _parse(self, line):
        if line.startswith("WEB_METRICS "):
            try:
                item = json.loads(line[12:])
                self.metrics.append(item)
                self.latest = item
            except ValueError:
                pass
        elif line.startswith("WEB_RUNTIME "):
            try:
                self.latest = json.loads(line[12:])
            except ValueError:
                pass

    def stop(self):
        with self.lock:
            if not self.active():
                raise ValueError("当前没有正在运行的进程")
            self.state = "stopping"
            if self.stop_file:
                self.stop_file.parent.mkdir(parents=True, exist_ok=True)
                self.stop_file.touch()
            else:
                self.process.terminate()

    def terminate(self):
        with self.lock:
            if self.active():
                self.process.terminate()


PARAMETER_SPECS = json.loads((Path(__file__).parent / "parameter_specs.json").read_text(encoding="utf-8"))


def parameter_values(phase, data):
    values = {}
    for spec in PARAMETER_SPECS[phase]:
        key = spec["name"]
        value = data.get(key, spec["default"])
        if value is None or value == "":
            if spec["optional"]:
                continue
            raise ValueError(f"{key} 不能为空")
        if spec["type"] == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"{key} 必须是布尔值")
        elif spec["type"] in ("int", "float"):
            if isinstance(value, bool):
                raise ValueError(f"{key} 必须是数值")
            value = float(value)
            if not math.isfinite(value) or not spec["min"] <= value <= spec["max"]:
                raise ValueError(f"{key} 必须在 {spec['min']}–{spec['max']} 之间")
            if spec["type"] == "int":
                if value != int(value):
                    raise ValueError(f"{key} 必须是整数")
                value = int(value)
        elif key == "camera_names":
            value = value.split() if isinstance(value, str) else value
            if value != ["wrist", "external"]:
                raise ValueError("V7 相机顺序必须是 wrist external")
        elif key == "depth_stats_json":
            value = str(Path(value).expanduser().resolve())
            if not Path(value).is_file():
                raise FileNotFoundError("深度统计 JSON 不存在")
        else:
            value = _safe_token({key:value}, key, spec["default"], spec.get("choices"))
        values[key] = value
    if phase == "training" and values["hidden_dim"] % values["nheads"]:
        raise ValueError("hidden_dim 必须能被 nheads 整除")
    if phase == "execution" and values["gripper_min_m"] >= values["gripper_max_m"]:
        raise ValueError("夹爪最小开度必须小于最大开度")
    if phase == "execution" and (values["wrist_width"], values["wrist_height"]) != (values["external_width"], values["external_height"]):
        raise ValueError("两路相机的宽高必须一致")
    return values


def parameter_arguments(phase, data):
    args = []
    for key, value in parameter_values(phase, data).items():
        if isinstance(value, list):
            args.extend([f"--{key}", *value])
        elif isinstance(value, bool):
            args.append(f"--{key}" if value else f"--no-{key}")
        else:
            args.extend([f"--{key}", str(value)])
    return args


def runtime_environment(python_executable):
    """Provide the selected Conda environment's DLL paths to child processes."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    if os.name == "nt":
        prefix = Path(python_executable).resolve().parent
        directories = [prefix, prefix / "Library/mingw-w64/bin", prefix / "Library/usr/bin",
                       prefix / "Library/bin", prefix / "Scripts", prefix / "bin"]
        env["PATH"] = os.pathsep.join(str(p) for p in directories if p.is_dir()) + os.pathsep + env.get("PATH", "")
        env["CONDA_PREFIX"] = str(prefix)
    return env


def training_command(task, data, python_executable=None):
    if not TRAIN_SCRIPT.is_file():
        raise FileNotFoundError(f"训练脚本不存在：{TRAIN_SCRIPT}")
    args = parameter_arguments("training", data)
    if not any(Path(task["dataset_dir"]).glob("episode_*.hdf5")):
        raise ValueError("当前任务没有可训练的 Episode")
    return [str(python_executable or sys.executable), "-u", str(Path(__file__).parent / "training_entry.py"),
            "--dataset_dir", task["dataset_dir"], "--ckpt_dir", task["ckpt_dir"], *args]


def execution_command(task, data, runtime_dir, python_executable=None):
    if not RUN_SCRIPT.is_file():
        raise FileNotFoundError(f"执行脚本不存在：{RUN_SCRIPT}")
    args = parameter_arguments("execution", data)
    name = parameter_values("execution", data)["ckpt_name"]
    if not (Path(task["ckpt_dir"]) / name).is_file():
        raise FileNotFoundError(f"未找到模型：{name}")
    control = data.get("control", False)
    if not isinstance(control, bool) or (control and data.get("motion_confirmed") is not True):
        raise ValueError("真实控制前必须确认安全检查")
    runtime_dir = Path(runtime_dir)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    preview, stop = runtime_dir / "execution_preview.jpg", runtime_dir / "execution.stop"
    command = [str(python_executable or sys.executable), "-u", str(RUN_SCRIPT),
               "--ckpt_dir", task["ckpt_dir"], "--headless", "--auto_start",
               "--web_preview_file", str(preview), "--web_stop_file", str(stop), *args]
    if control:
        command.extend(["--control", "--web_authorized"])
    return command, stop, preview
