#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PiPER-X Leader/Follower + Orbbec DaBai DC1 ACT Dataset Recorder

Data mapping:
    Leader/master joint + gripper control -> /action
    Follower/slave joint + gripper feedback -> /observations/qpos
    Orbbec RGB image via pyorbbecsdk -> /observations/images/wrist

ACT-ready units:
    joints: rad
    gripper: m
    image: RGB uint8

Raw PiPER integer values are additionally stored under /raw.
"""

import argparse
import re
import threading
import time
from pathlib import Path

import cv2
import h5py
import numpy as np

# Hardware SDKs are loaded only when connecting. Dataset tools and the web
# console can therefore inspect recordings on machines without the hardware.


JOINT_RAW_TO_RAD = 1e-3 * np.pi / 180.0
GRIPPER_RAW_TO_M = 1e-6


# ============================================================
# Orbbec DaBai DC1
# ============================================================

def color_frame_to_rgb(frame):
    """Convert Orbbec ColorFrame into contiguous RGB uint8 image."""
    from pyorbbecsdk import OBFormat
    width = frame.get_width()
    height = frame.get_height()
    fmt = frame.get_format()
    data = np.asanyarray(frame.get_data())

    if fmt == OBFormat.RGB:
        image = np.resize(data, (height, width, 3))
        return np.ascontiguousarray(image, dtype=np.uint8)

    if fmt == OBFormat.BGR:
        image = np.resize(data, (height, width, 3))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(image, dtype=np.uint8)

    if fmt == OBFormat.MJPG:
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    if fmt == OBFormat.YUYV:
        yuv = np.resize(data, (height, width, 2))
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_YUY2)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    if fmt == OBFormat.UYVY:
        yuv = np.resize(data, (height, width, 2))
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_UYVY)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    if fmt == OBFormat.I420:
        yuv = np.resize(data, (height * 3 // 2, width))
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    if fmt == OBFormat.NV12:
        yuv = np.resize(data, (height * 3 // 2, width))
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    if fmt == OBFormat.NV21:
        yuv = np.resize(data, (height * 3 // 2, width))
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    return None


class OrbbecCamera:
    """Background RGB capture for legacy pyorbbecsdk v1.x."""

    def __init__(self, width=640, height=480, fps=30):
        from pyorbbecsdk import Config, OBError, OBFormat, OBSensorType, Pipeline
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = Pipeline()
        self.config = Config()

        self.lock = threading.Lock()
        self.latest_rgb = None
        self.latest_host_stamp = 0.0
        self.latest_format = None
        self.running = False
        self.thread = None

        profile_list = self.pipeline.get_stream_profile_list(
            OBSensorType.COLOR_SENSOR
        )
        if profile_list is None:
            raise RuntimeError("No Orbbec COLOR_SENSOR found.")

        profile = None

        # First choice: RGB. If the camera does not expose RGB directly,
        # try MJPG, then fall back to the camera's default color profile.
        try:
            profile = profile_list.get_video_stream_profile(
                width,
                height,
                OBFormat.RGB,
                fps,
            )
            print(f"[INFO] Orbbec RGB profile selected: {profile}")
        except OBError:
            try:
                profile = profile_list.get_video_stream_profile(
                    width,
                    height,
                    OBFormat.MJPG,
                    fps,
                )
                print(f"[INFO] Orbbec MJPG profile selected: {profile}")
            except OBError:
                profile = profile_list.get_default_video_stream_profile()
                print(f"[WARN] Falling back to default color profile: {profile}")

        self.profile = profile
        self.config.enable_stream(profile)
        self.pipeline.start(self.config)

        self.running = True
        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
        )
        self.thread.start()

    def _capture_loop(self):
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(100)
                if frames is None:
                    continue

                color_frame = frames.get_color_frame()
                if color_frame is None:
                    continue

                rgb = color_frame_to_rgb(color_frame)
                if rgb is None:
                    continue

                host_stamp = time.monotonic()

                with self.lock:
                    self.latest_rgb = rgb
                    self.latest_host_stamp = host_stamp
                    self.latest_format = str(color_frame.get_format())

            except Exception as exc:
                if self.running:
                    print(f"[WARN] Orbbec capture error: {exc}")
                time.sleep(0.02)

    def wait_until_ready(self, timeout_s=10.0):
        print(
            f"[INFO] Waiting for DaBai DC1 RGB frame, "
            f"timeout={timeout_s:.1f}s ..."
        )
        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline:
            frame = self.get_latest()
            if frame is not None:
                image, _ = frame
                print(
                    f"[OK] Camera ready: "
                    f"{image.shape[1]}x{image.shape[0]} RGB"
                )
                return
            time.sleep(0.05)

        raise RuntimeError(
            "No valid Orbbec RGB frame received. "
            "First verify pyorbbecsdk examples/color_viewer.py works."
        )

    def get_latest(self):
        with self.lock:
            if self.latest_rgb is None:
                return None
            return self.latest_rgb.copy(), self.latest_host_stamp

    def close(self):
        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=1.0)

        try:
            self.pipeline.stop()
        except Exception:
            pass


# ============================================================
# PiPER Leader / Follower
# ============================================================

class PiperReader:
    """
    Read PiPER Leader/Follower traffic on one CAN bus.

    Leader/master control frames become ACT action.
    Follower/slave feedback becomes ACT qpos.

    The Leader only transmits control frames while it is being moved.
    Therefore this reader caches the latest valid Leader command and
    reuses it while the Leader is stationary (zero-order hold).
    """

    def __init__(self, can_name: str, judge_flag: bool = True):
        try:
            from piper_sdk import C_PiperInterface_V2 as PiperInterface
        except ImportError:
            from piper_sdk import C_PiperInterface as PiperInterface
        self.piper = PiperInterface(can_name, judge_flag)
        self.piper.ConnectPort()

        self.last_action_raw = None
        self.last_action = None
        self.last_action_sdk_stamp = 0.0
        self.last_action_host_stamp = 0.0
        self.leader_initialized = False

    @staticmethod
    def _joint_raw(joint_obj):
        return np.asarray(
            [
                joint_obj.joint_1,
                joint_obj.joint_2,
                joint_obj.joint_3,
                joint_obj.joint_4,
                joint_obj.joint_5,
                joint_obj.joint_6,
            ],
            dtype=np.int32,
        )

    @staticmethod
    def _to_si(raw7: np.ndarray):
        out = np.empty(7, dtype=np.float32)
        out[:6] = raw7[:6].astype(np.float32) * JOINT_RAW_TO_RAD
        out[6] = np.float32(raw7[6]) * GRIPPER_RAW_TO_M
        return out

    def read(self):
        joint_ctrl_msg = self.piper.GetArmJointCtrl()
        gripper_ctrl_msg = self.piper.GetArmGripperCtrl()

        joint_state_msg = self.piper.GetArmJointMsgs()
        gripper_state_msg = self.piper.GetArmGripperMsgs()

        qpos_raw = np.concatenate(
            [
                self._joint_raw(joint_state_msg.joint_state),
                np.asarray(
                    [gripper_state_msg.gripper_state.grippers_angle],
                    dtype=np.int32,
                ),
            ]
        )
        qpos = self._to_si(qpos_raw)

        joint_action_hz = float(joint_ctrl_msg.Hz)
        gripper_action_hz = float(gripper_ctrl_msg.Hz)

        # A fresh Leader command can come from arm motion or gripper motion.
        action_updated = (
            joint_action_hz > 0.0
            or gripper_action_hz > 0.0
        )

        if action_updated:
            action_raw = np.concatenate(
                [
                    self._joint_raw(joint_ctrl_msg.joint_ctrl),
                    np.asarray(
                        [gripper_ctrl_msg.gripper_ctrl.grippers_angle],
                        dtype=np.int32,
                    ),
                ]
            )

            self.last_action_raw = action_raw.copy()
            self.last_action = self._to_si(action_raw)
            self.last_action_sdk_stamp = float(joint_ctrl_msg.time_stamp)
            self.last_action_host_stamp = time.monotonic()
            self.leader_initialized = True

        if not self.leader_initialized:
            return {
                "ready": False,
                "action": None,
                "qpos": qpos,
                "action_raw": None,
                "qpos_raw": qpos_raw,
                "action_updated": False,
                "action_hz": joint_action_hz,
                "gripper_action_hz": gripper_action_hz,
                "qpos_hz": float(joint_state_msg.Hz),
                "action_sdk_stamp": 0.0,
                "qpos_sdk_stamp": float(joint_state_msg.time_stamp),
                "action_age_s": float("inf"),
            }

        now = time.monotonic()

        return {
            "ready": True,
            "action": self.last_action.copy(),
            "qpos": qpos,
            "action_raw": self.last_action_raw.copy(),
            "qpos_raw": qpos_raw,
            "action_updated": bool(action_updated),
            "action_hz": joint_action_hz,
            "gripper_action_hz": gripper_action_hz,
            "qpos_hz": float(joint_state_msg.Hz),
            "action_sdk_stamp": self.last_action_sdk_stamp,
            "qpos_sdk_stamp": float(joint_state_msg.time_stamp),
            "action_age_s": now - self.last_action_host_stamp,
        }

    def close(self):
        if hasattr(self.piper, "DisconnectPort"):
            try:
                self.piper.DisconnectPort()
            except Exception:
                pass


# ============================================================
# HDF5 writer
# ============================================================

class HDF5EpisodeWriter:
    """Append synchronized samples to an ACT-style HDF5 episode."""

    def __init__(
        self,
        path: Path,
        sample_rate_hz: float,
        can_name: str,
        camera_name: str,
        camera_width: int,
        camera_height: int,
        camera_fps: int,
        batch_size: int = 20,
    ):
        self.path = path
        self.batch_size = batch_size
        self.count = 0
        self.buffers = []
        self.initialized = False

        self.file = h5py.File(path, "w")

        self.file.attrs["sim"] = False
        self.file.attrs["sample_rate_hz"] = float(sample_rate_hz)
        self.file.attrs["can_name"] = can_name
        self.file.attrs["camera_name"] = camera_name
        self.file.attrs["camera_backend"] = "pyorbbecsdk"
        self.file.attrs["camera_width_requested"] = int(camera_width)
        self.file.attrs["camera_height_requested"] = int(camera_height)
        self.file.attrs["camera_fps_requested"] = int(camera_fps)
        self.file.attrs["joint_unit"] = "rad"
        self.file.attrs["gripper_unit"] = "m"
        self.file.attrs["image_encoding"] = "rgb8"
        self.file.attrs["created_unix_time"] = time.time()

        self.obs_group = self.file.create_group("observations")
        self.image_group = self.obs_group.create_group("images")
        self.raw_group = self.file.create_group("raw")
        self.ts_group = self.file.create_group("timestamps")

        self.camera_name = camera_name

    def _init_datasets(self, image_shape):
        h, w, c = image_shape

        if c != 3:
            raise ValueError(
                f"Expected RGB image with 3 channels, got {image_shape}"
            )

        self.ds_qpos = self.obs_group.create_dataset(
            "qpos",
            shape=(0, 7),
            maxshape=(None, 7),
            dtype=np.float32,
            chunks=(256, 7),
        )

        self.ds_action = self.file.create_dataset(
            "action",
            shape=(0, 7),
            maxshape=(None, 7),
            dtype=np.float32,
            chunks=(256, 7),
        )

        self.ds_image = self.image_group.create_dataset(
            self.camera_name,
            shape=(0, h, w, 3),
            maxshape=(None, h, w, 3),
            dtype=np.uint8,
            chunks=(1, h, w, 3),
            compression="lzf",
        )

        self.ds_qpos_raw = self.raw_group.create_dataset(
            "qpos",
            shape=(0, 7),
            maxshape=(None, 7),
            dtype=np.int32,
            chunks=(256, 7),
        )

        self.ds_action_raw = self.raw_group.create_dataset(
            "action",
            shape=(0, 7),
            maxshape=(None, 7),
            dtype=np.int32,
            chunks=(256, 7),
        )

        def make_ts(name):
            return self.ts_group.create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                dtype=np.float64,
                chunks=(512,),
            )

        self.ds_sample_ts = make_ts("sample_monotonic")
        self.ds_camera_host_ts = make_ts("camera_host_monotonic")
        self.ds_action_sdk_ts = make_ts("action_sdk")
        self.ds_qpos_sdk_ts = make_ts("qpos_sdk")
        self.ds_image_age = make_ts("image_age_s")
        self.ds_action_age = make_ts("action_age_s")

        self.ds_action_updated = self.file.create_dataset(
            "action_updated",
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint8,
            chunks=(512,),
        )

        self.image_shape = tuple(image_shape)
        self.initialized = True

    @staticmethod
    def _append(ds, data):
        old = ds.shape[0]
        n = data.shape[0]
        ds.resize(old + n, axis=0)
        ds[old : old + n] = data

    def add(self, sample):
        image = sample["image"]

        if not self.initialized:
            self._init_datasets(image.shape)
        elif tuple(image.shape) != self.image_shape:
            raise RuntimeError(
                "Camera resolution changed during recording: "
                f"{self.image_shape} -> {image.shape}"
            )

        self.buffers.append(sample)

        if len(self.buffers) >= self.batch_size:
            self.flush()

    def flush(self):
        if not self.buffers:
            return

        qpos = np.stack([x["qpos"] for x in self.buffers]).astype(np.float32)
        action = np.stack([x["action"] for x in self.buffers]).astype(np.float32)
        images = np.stack([x["image"] for x in self.buffers]).astype(np.uint8)
        qpos_raw = np.stack([x["qpos_raw"] for x in self.buffers]).astype(np.int32)
        action_raw = np.stack([x["action_raw"] for x in self.buffers]).astype(np.int32)

        sample_ts = np.asarray(
            [x["sample_monotonic"] for x in self.buffers], dtype=np.float64
        )
        camera_host_ts = np.asarray(
            [x["camera_host_stamp"] for x in self.buffers], dtype=np.float64
        )
        action_sdk_ts = np.asarray(
            [x["action_sdk_stamp"] for x in self.buffers], dtype=np.float64
        )
        qpos_sdk_ts = np.asarray(
            [x["qpos_sdk_stamp"] for x in self.buffers], dtype=np.float64
        )
        image_age = np.asarray(
            [x["image_age_s"] for x in self.buffers], dtype=np.float64
        )
        action_age = np.asarray(
            [x["action_age_s"] for x in self.buffers], dtype=np.float64
        )
        action_updated = np.asarray(
            [1 if x["action_updated"] else 0 for x in self.buffers],
            dtype=np.uint8,
        )

        self._append(self.ds_qpos, qpos)
        self._append(self.ds_action, action)
        self._append(self.ds_image, images)
        self._append(self.ds_qpos_raw, qpos_raw)
        self._append(self.ds_action_raw, action_raw)
        self._append(self.ds_sample_ts, sample_ts)
        self._append(self.ds_camera_host_ts, camera_host_ts)
        self._append(self.ds_action_sdk_ts, action_sdk_ts)
        self._append(self.ds_qpos_sdk_ts, qpos_sdk_ts)
        self._append(self.ds_image_age, image_age)
        self._append(self.ds_action_age, action_age)
        self._append(self.ds_action_updated, action_updated)

        self.count += len(self.buffers)
        self.buffers.clear()
        self.file.flush()

    def close(self):
        self.flush()
        self.file.attrs["num_samples"] = int(self.count)
        self.file.close()


# ============================================================
# Helpers
# ============================================================

def next_episode_path(dataset_dir: Path, requested_index=None):
    dataset_dir.mkdir(parents=True, exist_ok=True)

    if requested_index is not None:
        return dataset_dir / f"episode_{requested_index:06d}.hdf5"

    pattern = re.compile(r"episode_(\d+)\.hdf5$")
    ids = []

    for path in dataset_dir.glob("episode_*.hdf5"):
        match = pattern.match(path.name)
        if match:
            ids.append(int(match.group(1)))

    next_id = max(ids) + 1 if ids else 0
    return dataset_dir / f"episode_{next_id:06d}.hdf5"


def wait_for_robot(robot: PiperReader, timeout_s: float):
    print(
        "[INFO] Waiting for robot initialization...\n"
        "       Follower qpos must be online.\n"
        "       Move the Leader arm once to initialize the ACT action."
    )

    deadline = time.monotonic() + timeout_s
    last = None
    follower_seen = False
    leader_seen = False

    while time.monotonic() < deadline:
        last = robot.read()

        if last["qpos_hz"] > 0.0:
            follower_seen = True

        if last["ready"]:
            leader_seen = True

        if follower_seen and leader_seen:
            print(
                "[OK] Robot ready: "
                f"qpos={last['qpos_hz']:.1f} Hz, "
                "Leader action initialized. "
                "When the Leader stops, the recorder will HOLD the "
                "last valid action."
            )
            return

        time.sleep(0.02)

    qpos_hz = 0.0 if last is None else last["qpos_hz"]

    if not follower_seen:
        raise RuntimeError(
            "Follower qpos was not received "
            f"(qpos={qpos_hz:.1f} Hz). "
            "Check CAN, Follower mode, CAN termination, and power."
        )

    raise RuntimeError(
        "Follower is online, but no valid Leader action was observed. "
        "Move the Leader arm during startup so at least one control "
        "command can be cached."
    )


def draw_preview(
    rgb,
    state,
    sample_count=0,
    qpos_hz=0.0,
    action_state="HOLD",
    image_age_ms=0.0,
):
    """Draw the wrist-camera preview and status overlay."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if state == "READY":
        label = "READY  |  S: start recording  |  Q/ESC: quit"
        color = (0, 220, 255)
    else:
        label = "REC  |  Q/ESC: stop & save"
        color = (0, 0, 255)

    cv2.rectangle(
        bgr,
        (0, 0),
        (bgr.shape[1], 78),
        (0, 0, 0),
        thickness=-1,
    )

    cv2.putText(
        bgr,
        label,
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        2,
        cv2.LINE_AA,
    )

    status = (
        f"samples={sample_count}  "
        f"qpos={qpos_hz:.0f}Hz  "
        f"action={action_state}  "
        f"image_age={image_age_ms:.0f}ms"
    )

    cv2.putText(
        bgr,
        status,
        (12, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return bgr


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Record PiPER-X Leader/Follower + Orbbec DaBai DC1 data "
            "for ACT imitation learning."
        )
    )

    parser.add_argument(
        "--can",
        default="can0",
        help="SocketCAN interface, default: can0",
    )

    parser.add_argument(
        "--camera-name",
        default="wrist",
        help="HDF5 camera name under /observations/images/<camera-name>",
    )

    parser.add_argument(
        "--camera-width",
        type=int,
        default=640,
        help="Requested Orbbec color width, default: 640",
    )

    parser.add_argument(
        "--camera-height",
        type=int,
        default=0,
        help="Requested Orbbec color height; 0 lets SDK choose, default: 0",
    )

    parser.add_argument(
        "--camera-fps",
        type=int,
        default=30,
        help="Requested Orbbec color FPS, default: 30",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Dataset sampling frequency, default: 20 Hz",
    )

    parser.add_argument(
        "--max-image-age",
        type=float,
        default=0.10,
        help=(
            "Skip a sample if latest RGB frame is older than this many "
            "seconds, default: 0.10"
        ),
    )

    parser.add_argument(
        "--dataset_stand_bottle-dir",
        default="../dataset_stand_bottle",
        help="Dataset output directory",
    )

    parser.add_argument(
        "--episode-index",
        type=int,
        default=None,
        help="Optional fixed episode index; otherwise auto-increment.",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help=(
            "Automatically stop N seconds after recording starts. "
            "0 means stop with Q/ESC (preview) or Enter (--no-preview)."
        ),
    )

    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=10.0,
        help="Camera/robot startup timeout in seconds",
    )

    parser.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Show live wrist-camera preview. Default: enabled. "
            "Use --no-preview for terminal-only recording."
        ),
    )

    parser.add_argument(
        "--judge-flag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "piper_sdk CAN judge flag. Use --no-judge-flag if your "
            "CAN hardware requires it."
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be > 0")

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_path = next_episode_path(dataset_dir, args.episode_index)

    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing file: {output_path}"
        )

    camera = None
    robot = None
    writer = None

    try:
        print("[INFO] Starting Orbbec DaBai DC1...")
        camera = OrbbecCamera(
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
        )
        camera.wait_until_ready(args.startup_timeout)

        print(f"[INFO] Connecting PiPER on {args.can}...")
        robot = PiperReader(args.can, args.judge_flag)
        wait_for_robot(robot, args.startup_timeout)

        print("\nDataset mapping:")
        print(
            "  /observations/qpos               "
            "= Follower actual [q1..q6, gripper]"
        )
        print(
            "  /action                          "
            "= Leader target  [q1..q6, gripper]"
        )
        print(
            f"  /observations/images/{args.camera_name:<10} "
            "= DaBai DC1 RGB"
        )
        print("  joint unit = rad")
        print("  gripper unit = m")
        print(f"  dataset_stand_bottle sampling = {args.fps:.1f} Hz")
        print(f"  output = {output_path}\n")

        # ----------------------------------------------------
        # READY stage
        # ----------------------------------------------------
        latest_robot = robot.read()

        if args.preview:
            print(
                "[READY] Use the wrist-camera window to guide the robot.\n"
                "        Press S in the image window to START recording.\n"
                "        Press Q or ESC to quit without recording."
            )

            cv2.namedWindow(
                "PiPER ACT - Wrist Camera",
                cv2.WINDOW_NORMAL,
            )

            while True:
                latest_robot = robot.read()
                image_pack = camera.get_latest()

                if image_pack is not None:
                    image, camera_host_stamp = image_pack
                    now = time.monotonic()
                    image_age_ms = (
                        max(0.0, now - camera_host_stamp) * 1000.0
                    )

                    action_state = (
                        "NEW"
                        if latest_robot.get("action_updated", False)
                        else "HOLD"
                    )

                    preview = draw_preview(
                        image,
                        state="READY",
                        sample_count=0,
                        qpos_hz=latest_robot.get("qpos_hz", 0.0),
                        action_state=action_state,
                        image_age_ms=image_age_ms,
                    )

                    cv2.imshow(
                        "PiPER ACT - Wrist Camera",
                        preview,
                    )

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    print("[INFO] Quit before recording.")
                    return

                if key in (ord("s"), ord("S")):
                    break

                time.sleep(0.005)

        else:
            input(
                "Move both arms to the episode start pose, "
                "then press Enter to START..."
            )

        # ----------------------------------------------------
        # Create HDF5 only after user actually starts.
        # ----------------------------------------------------
        writer = HDF5EpisodeWriter(
            output_path,
            sample_rate_hz=args.fps,
            can_name=args.can,
            camera_name=args.camera_name,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            camera_fps=args.camera_fps,
        )

        print("[REC] Recording started.")

        stop_event = threading.Event()

        if not args.preview and args.duration <= 0:
            def wait_enter():
                try:
                    input(
                        "\n[REC] Recording... "
                        "press Enter to STOP and save.\n"
                    )
                    stop_event.set()
                except EOFError:
                    pass

            threading.Thread(
                target=wait_enter,
                daemon=True,
            ).start()

        period = 1.0 / args.fps
        next_tick = time.monotonic()
        start_time = next_tick

        skipped_old_image = 0
        skipped_no_image = 0
        skipped_no_robot = 0
        last_print_count = 0

        latest_robot = robot.read()
        last_image = None
        last_image_stamp = 0.0

        while not stop_event.is_set():
            now = time.monotonic()

            if (
                args.duration > 0
                and now - start_time >= args.duration
            ):
                break

            # Always refresh the displayed image even between dataset_stand_bottle
            # sampling ticks, so teleoperation is visually responsive.
            image_pack = camera.get_latest()
            if image_pack is not None:
                last_image, last_image_stamp = image_pack

            if args.preview and last_image is not None:
                image_age_ms = (
                    max(0.0, now - last_image_stamp) * 1000.0
                )

                action_state = (
                    "NEW"
                    if latest_robot.get("action_updated", False)
                    else "HOLD"
                )

                current_count = (
                    writer.count + len(writer.buffers)
                )

                preview = draw_preview(
                    last_image,
                    state="REC",
                    sample_count=current_count,
                    qpos_hz=latest_robot.get("qpos_hz", 0.0),
                    action_state=action_state,
                    image_age_ms=image_age_ms,
                )

                cv2.imshow(
                    "PiPER ACT - Wrist Camera",
                    preview,
                )

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break

            # Dataset sampling clock.
            if now < next_tick:
                time.sleep(min(next_tick - now, 0.002))
                continue

            if now - next_tick > 2.0 * period:
                next_tick = now

            sample_stamp = next_tick
            next_tick += period

            if last_image is None:
                skipped_no_image += 1
                continue

            image_age = now - last_image_stamp

            if image_age > args.max_image_age:
                skipped_old_image += 1
                continue

            latest_robot = robot.read()

            # Follower feedback must remain live.
            if latest_robot["qpos_hz"] <= 0.0:
                skipped_no_robot += 1
                continue

            # Leader only needs one valid action. After initialization,
            # action_hz == 0 means HOLD the previous target.
            if not latest_robot["ready"]:
                skipped_no_robot += 1
                continue

            writer.add(
                {
                    "image": last_image,
                    "qpos": latest_robot["qpos"],
                    "action": latest_robot["action"],
                    "qpos_raw": latest_robot["qpos_raw"],
                    "action_raw": latest_robot["action_raw"],
                    "sample_monotonic": sample_stamp,
                    "camera_host_stamp": last_image_stamp,
                    "action_sdk_stamp": latest_robot[
                        "action_sdk_stamp"
                    ],
                    "qpos_sdk_stamp": latest_robot[
                        "qpos_sdk_stamp"
                    ],
                    "image_age_s": image_age,
                    "action_age_s": latest_robot["action_age_s"],
                    "action_updated": latest_robot[
                        "action_updated"
                    ],
                }
            )

            current_count = (
                writer.count + len(writer.buffers)
            )

            if (
                current_count
                >= last_print_count + max(1, int(args.fps))
            ):
                last_print_count = current_count
                elapsed = now - start_time

                print(
                    f"\r[REC] {current_count:5d} samples | "
                    f"{elapsed:6.1f}s | "
                    f"image age={image_age * 1000:5.1f} ms | "
                    f"action="
                    f"{'NEW' if latest_robot['action_updated'] else 'HOLD'} | "
                    f"action age="
                    f"{latest_robot['action_age_s'] * 1000:5.0f} ms | "
                    f"skip old/noimg/norobot="
                    f"{skipped_old_image}/"
                    f"{skipped_no_image}/"
                    f"{skipped_no_robot}",
                    end="",
                    flush=True,
                )

        print("\n[INFO] Stopping recording...")

    except KeyboardInterrupt:
        print(
            "\n[INFO] Ctrl+C received, "
            "saving the current episode..."
        )

    finally:
        if writer is not None:
            writer.close()

            if writer.count > 0:
                print(
                    f"[OK] Saved {writer.count} samples "
                    f"-> {output_path}"
                )
            else:
                print(
                    f"[WARN] 0 samples saved -> {output_path}"
                )

        if robot is not None:
            robot.close()

        if camera is not None:
            camera.close()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
