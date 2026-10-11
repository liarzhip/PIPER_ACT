#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PiPER-X Leader/Follower + Orbbec wrist RGB-D + RealSense external RGB-D ACT Dataset Recorder (Windows)

Data mapping:
    Leader/master joint + gripper control -> /action
    Follower/slave joint + gripper feedback -> /observations/qpos
    Orbbec wrist RGB -> /observations/images/wrist
    Orbbec wrist aligned depth -> /observations/depth/wrist
    RealSense external RGB -> /observations/images/external
    RealSense external aligned depth -> /observations/depth/external

ACT-ready units:
    joints: rad
    gripper: m
    RGB image: uint8
    depth: uint16 millimeters (0 = invalid)

Raw PiPER integer values are additionally stored under /raw.

Windows CAN backend:
    piper_sdk >= 0.6.2 + python-can-agx-cando
    interface/bustype = agx_cando
    channel = "0" by default
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
    """Background synchronized RGB-D capture for Orbbec DaBai DC1."""

    def __init__(
        self,
        width=640,
        height=480,
        fps=30,
        depth_align=True,
        depth_min_mm=150.0,
        depth_max_mm=1500.0,
    ):
        from pyorbbecsdk import Config, OBError, OBFormat, OBSensorType, Pipeline

        self.width = width
        self.height = height
        self.fps = fps
        self.depth_align = bool(depth_align)
        self.depth_min_mm = float(depth_min_mm)
        self.depth_max_mm = float(depth_max_mm)

        self.pipeline = Pipeline()
        self.config = Config()

        self.lock = threading.Lock()
        self.latest_rgb = None
        self.latest_depth_mm = None
        self.latest_host_stamp = 0.0
        self.latest_format = None
        self.running = False
        self.thread = None
        self._warned_depth_scale = False

        # ---------------- Color stream ----------------
        color_profiles = self.pipeline.get_stream_profile_list(
            OBSensorType.COLOR_SENSOR
        )
        if color_profiles is None:
            raise RuntimeError("No Orbbec COLOR_SENSOR found.")

        color_profile = None
        try:
            color_profile = color_profiles.get_video_stream_profile(
                width,
                height,
                OBFormat.RGB,
                fps,
            )
            print(f"[INFO] Orbbec RGB profile selected: {color_profile}")
        except OBError:
            try:
                color_profile = color_profiles.get_video_stream_profile(
                    width,
                    height,
                    OBFormat.MJPG,
                    fps,
                )
                print(f"[INFO] Orbbec MJPG profile selected: {color_profile}")
            except OBError:
                color_profile = color_profiles.get_default_video_stream_profile()
                print(
                    f"[WARN] Falling back to default color profile: "
                    f"{color_profile}"
                )

        # ---------------- Depth stream ----------------
        depth_profiles = self.pipeline.get_stream_profile_list(
            OBSensorType.DEPTH_SENSOR
        )
        if depth_profiles is None:
            raise RuntimeError("No Orbbec DEPTH_SENSOR found.")

        try:
            # Prefer a depth stream at the same FPS. Width/height=0 are wildcards
            # in supported pyorbbecsdk builds.
            depth_profile = depth_profiles.get_video_stream_profile(
                0,
                0,
                OBFormat.Y16,
                fps,
            )
            print(f"[INFO] Orbbec depth profile selected: {depth_profile}")
        except Exception:
            depth_profile = depth_profiles.get_default_video_stream_profile()
            print(f"[WARN] Falling back to default depth profile: {depth_profile}")

        self.color_profile = color_profile
        self.depth_profile = depth_profile
        self.config.enable_stream(color_profile)
        self.config.enable_stream(depth_profile)

        # Ask the SDK for synchronized color/depth frames when available.
        if hasattr(self.pipeline, "enable_frame_sync"):
            try:
                self.pipeline.enable_frame_sync()
                print("[INFO] Orbbec frame sync enabled.")
            except Exception as exc:
                print(f"[WARN] Could not enable Orbbec frame sync: {exc}")

        # Software D2C alignment keeps the depth map in the RGB image geometry.
        self.align_filter = None
        if self.depth_align:
            try:
                from pyorbbecsdk import AlignFilter, OBStreamType
                self.align_filter = AlignFilter(
                    align_to_stream=OBStreamType.COLOR_STREAM
                )
                print("[INFO] Depth-to-color alignment enabled (software D2C).")
            except Exception as exc:
                raise RuntimeError(
                    "Depth alignment was requested but this pyorbbecsdk build "
                    "does not expose AlignFilter/OBStreamType. Install a build "
                    "that supports D2C alignment, or run with --no-depth-align "
                    "only for camera diagnostics (not recommended for RGB-D training). "
                    f"Original error: {exc}"
                ) from exc

        self.pipeline.start(self.config)

        self.running = True
        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="orbbec_rgbd_capture",
        )
        self.thread.start()

    @staticmethod
    def _depth_frame_to_mm(depth_frame):
        h = int(depth_frame.get_height())
        w = int(depth_frame.get_width())

        raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
        expected = h * w
        if raw.size != expected:
            raise ValueError(
                f"Depth buffer has {raw.size} uint16 values, expected {expected} "
                f"for {w}x{h}."
            )

        raw = raw.reshape(h, w)

        scale = 1.0
        if hasattr(depth_frame, "get_depth_scale"):
            scale = float(depth_frame.get_depth_scale())

        # Orbbec depth scale converts raw uint16 depth values to millimeters.
        depth_mm = raw.astype(np.float32) * scale
        return depth_mm

    def _capture_loop(self):
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(100)
                if frames is None:
                    continue

                if self.align_filter is not None:
                    aligned = self.align_filter.process(frames)
                    if aligned is None:
                        continue
                    if hasattr(aligned, "as_frame_set"):
                        aligned = aligned.as_frame_set()
                    frames = aligned

                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame()
                if color_frame is None or depth_frame is None:
                    continue

                rgb = color_frame_to_rgb(color_frame)
                if rgb is None:
                    continue

                depth_mm = self._depth_frame_to_mm(depth_frame)

                # Keep invalid depth as 0. Clipping here also makes the on-disk
                # representation robust against far-range outliers.
                valid = (
                    np.isfinite(depth_mm)
                    & (depth_mm >= self.depth_min_mm)
                    & (depth_mm <= self.depth_max_mm)
                )
                depth_u16 = np.zeros(depth_mm.shape, dtype=np.uint16)
                depth_u16[valid] = np.rint(depth_mm[valid]).astype(np.uint16)

                if depth_u16.shape != rgb.shape[:2]:
                    if self.align_filter is None:
                        # This path is for diagnostics only. It is NOT geometric D2C.
                        depth_u16 = cv2.resize(
                            depth_u16,
                            (rgb.shape[1], rgb.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    else:
                        raise RuntimeError(
                            "Aligned depth shape does not match RGB shape: "
                            f"depth={depth_u16.shape}, rgb={rgb.shape[:2]}"
                        )

                host_stamp = time.monotonic()

                with self.lock:
                    self.latest_rgb = rgb
                    self.latest_depth_mm = depth_u16
                    self.latest_host_stamp = host_stamp
                    self.latest_format = (
                        f"color={color_frame.get_format()}, "
                        f"depth={depth_frame.get_format()}"
                    )

            except Exception as exc:
                if self.running:
                    print(f"[WARN] Orbbec RGB-D capture error: {exc}")
                time.sleep(0.02)

    def wait_until_ready(self, timeout_s=10.0):
        print(
            f"[INFO] Waiting for DaBai DC1 RGB-D frame, "
            f"timeout={timeout_s:.1f}s ..."
        )
        deadline = time.monotonic() + timeout_s

        while time.monotonic() < deadline:
            frame = self.get_latest()
            if frame is not None:
                rgb, depth_mm, _ = frame
                valid_ratio = float(np.count_nonzero(depth_mm)) / float(depth_mm.size)
                print(
                    f"[OK] Camera ready: {rgb.shape[1]}x{rgb.shape[0]} RGB-D | "
                    f"depth valid={valid_ratio * 100.0:.1f}%"
                )
                return
            time.sleep(0.05)

        raise RuntimeError(
            "No valid synchronized Orbbec RGB-D frame received. Verify that both "
            "color and depth streams work in the official pyorbbecsdk examples."
        )

    def get_latest(self):
        with self.lock:
            if self.latest_rgb is None or self.latest_depth_mm is None:
                return None
            return (
                self.latest_rgb.copy(),
                self.latest_depth_mm.copy(),
                float(self.latest_host_stamp),
            )

    def close(self):
        self.running = False

        if self.thread is not None:
            self.thread.join(timeout=1.0)

        try:
            self.pipeline.stop()
        except Exception:
            pass



# ============================================================
# Intel RealSense external RGB-D camera
# ============================================================

class RealSenseCamera:
    """Background synchronized RGB-D capture using pyrealsense2.

    Depth is geometrically aligned to the RealSense color stream with rs.align.
    On disk depth is uint16 millimeters, with 0 reserved for invalid/out-of-range.
    """

    def __init__(
        self,
        width=640,
        height=480,
        fps=30,
        depth_min_mm=250.0,
        depth_max_mm=3000.0,
        serial="",
    ):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError(
                "pyrealsense2 is not installed. Run: pip install pyrealsense2"
            ) from exc

        self.rs = rs
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.depth_min_mm = float(depth_min_mm)
        self.depth_max_mm = float(depth_max_mm)
        self.serial = str(serial).strip()

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        if self.serial:
            self.config.enable_device(self.serial)

        self.config.enable_stream(
            rs.stream.depth,
            self.width,
            self.height,
            rs.format.z16,
            self.fps,
        )
        self.config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs.format.bgr8,
            self.fps,
        )

        self.profile = self.pipeline.start(self.config)

        device = self.profile.get_device()
        try:
            self.device_name = device.get_info(rs.camera_info.name)
        except Exception:
            self.device_name = "RealSense"

        try:
            self.device_serial = device.get_info(rs.camera_info.serial_number)
        except Exception:
            self.device_serial = self.serial or "unknown"

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale_m = float(depth_sensor.get_depth_scale())
        self.align = rs.align(rs.stream.color)

        self.lock = threading.Lock()
        self.latest_rgb = None
        self.latest_depth_mm = None
        self.latest_host_stamp = 0.0
        self.running = True

        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="realsense_external_rgbd_capture",
        )
        self.thread.start()

        print(
            f"[OK] RealSense started: {self.device_name} "
            f"serial={self.device_serial} | "
            f"{self.width}x{self.height}@{self.fps}"
        )
        print(
            f"[INFO] RealSense depth scale = "
            f"{self.depth_scale_m * 1000.0:.6f} mm/raw-unit"
        )

    def _capture_loop(self):
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(1000)
                aligned = self.align.process(frames)

                depth_frame = aligned.get_depth_frame()
                color_frame = aligned.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                # pyrealsense2 BGR8 -> RGB uint8.
                bgr = np.asanyarray(color_frame.get_data())
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

                raw_depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)
                depth_mm = raw_depth * self.depth_scale_m * 1000.0

                valid = (
                    np.isfinite(depth_mm)
                    & (depth_mm >= self.depth_min_mm)
                    & (depth_mm <= self.depth_max_mm)
                )

                depth_u16 = np.zeros(
                    depth_mm.shape,
                    dtype=np.uint16,
                )
                depth_u16[valid] = np.rint(
                    depth_mm[valid]
                ).astype(np.uint16)

                if depth_u16.shape != rgb.shape[:2]:
                    raise RuntimeError(
                        "RealSense aligned depth/RGB shape mismatch: "
                        f"depth={depth_u16.shape}, rgb={rgb.shape[:2]}"
                    )

                host_stamp = time.monotonic()

                with self.lock:
                    self.latest_rgb = np.ascontiguousarray(
                        rgb,
                        dtype=np.uint8,
                    )
                    self.latest_depth_mm = depth_u16
                    self.latest_host_stamp = host_stamp

            except Exception as exc:
                if self.running:
                    print(f"[WARN] RealSense capture error: {exc}")
                time.sleep(0.02)

    def wait_until_ready(self, timeout_s=10.0):
        print(
            f"[INFO] Waiting for external RealSense RGB-D, "
            f"timeout={timeout_s:.1f}s ..."
        )
        deadline = time.monotonic() + float(timeout_s)

        while time.monotonic() < deadline:
            pack = self.get_latest()
            if pack is not None:
                rgb, depth_mm, _ = pack
                valid_ratio = (
                    float(np.count_nonzero(depth_mm))
                    / float(depth_mm.size)
                )
                print(
                    f"[OK] RealSense ready: "
                    f"{rgb.shape[1]}x{rgb.shape[0]} RGB-D | "
                    f"depth valid={valid_ratio * 100.0:.1f}%"
                )
                return
            time.sleep(0.05)

        raise RuntimeError(
            "No valid RealSense RGB-D frame received. "
            "Verify the device in RealSense Viewer and check USB 3.x."
        )

    def get_latest(self):
        with self.lock:
            if (
                self.latest_rgb is None
                or self.latest_depth_mm is None
            ):
                return None

            return (
                self.latest_rgb.copy(),
                self.latest_depth_mm.copy(),
                float(self.latest_host_stamp),
            )

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

    def __init__(
        self,
        can_channel: str = "0",
        can_backend: str = "agx_cando",
        judge_flag: bool = False,
    ):
        """
        Windows connection path.

        Keep piper_sdk for recording because it exposes BOTH:
            GetArmJointCtrl()/GetArmGripperCtrl()   -> Leader action
            GetArmJointMsgs()/GetArmGripperMsgs()  -> Follower qpos
        """
        try:
            import piper_sdk
            from piper_sdk import C_PiperInterface_V2 as PiperInterface
        except ImportError as exc:
            raise ImportError(
                "piper_sdk is required for Windows Leader/Follower recording. "
                "Install the current official SDK and python-can-agx-cando."
            ) from exc

        version = getattr(piper_sdk, "__version__", "unknown")
        print(f"[INFO] piper_sdk version: {version}")
        print(
            f"[INFO] Opening PiPER CAN: "
            f"backend={can_backend}, channel={can_channel}"
        )

        self.piper = PiperInterface(
            can_name=str(can_channel),
            judge_flag=bool(judge_flag),
            can_auto_init=False,
        )

        try:
            self.piper.CreateCanBus(
                can_name=str(can_channel),
                bustype=str(can_backend),
                expected_bitrate=1_000_000,
                judge_flag=bool(judge_flag),
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to create Windows PiPER CAN bus. "
                "Use piper_sdk >= 0.6.2 and install "
                "python-can-agx-cando. "
                f"Original error: {exc}"
            ) from exc

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
# Dual-camera HDF5 writer
# ============================================================

class HDF5EpisodeWriter:
    """Append synchronized dual RGB-D + robot samples to an ACT-style HDF5 episode."""

    def __init__(
        self,
        path: Path,
        sample_rate_hz: float,
        can_channel: str,
        can_backend: str,
        wrist_config: dict,
        external_config: dict,
        batch_size: int = 20,
    ):
        self.path = path
        self.batch_size = int(batch_size)
        self.count = 0
        self.buffers = []
        self.initialized = False

        self.camera_names = ["wrist", "external"]
        self.camera_configs = {
            "wrist": dict(wrist_config),
            "external": dict(external_config),
        }

        self.file = h5py.File(path, "w")

        self.file.attrs["sim"] = False
        self.file.attrs["sample_rate_hz"] = float(sample_rate_hz)
        self.file.attrs["can_channel"] = str(can_channel)
        self.file.attrs["can_backend"] = str(can_backend)
        self.file.attrs["camera_names"] = np.asarray(
            self.camera_names,
            dtype=h5py.string_dtype("utf-8"),
        )
        self.file.attrs["joint_unit"] = "rad"
        self.file.attrs["gripper_unit"] = "m"
        self.file.attrs["image_encoding"] = "rgb8"
        self.file.attrs["depth_encoding"] = "uint16_mm"
        self.file.attrs["depth_unit"] = "mm"
        self.file.attrs["depth_invalid_value"] = 0
        self.file.attrs["created_unix_time"] = time.time()

        for cam_name, cfg in self.camera_configs.items():
            prefix = f"{cam_name}_"
            for key, value in cfg.items():
                if value is None:
                    continue
                self.file.attrs[prefix + key] = value

        self.obs_group = self.file.create_group("observations")
        self.image_group = self.obs_group.create_group("images")
        self.depth_group = self.obs_group.create_group("depth")
        self.raw_group = self.file.create_group("raw")
        self.ts_group = self.file.create_group("timestamps")

    def _init_datasets(self, sample):
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

        self.ds_images = {}
        self.ds_depths = {}
        self.image_shapes = {}
        self.depth_shapes = {}

        expected_hw = None

        for cam in self.camera_names:
            image = sample[f"{cam}_image"]
            depth = sample[f"{cam}_depth"]

            h, w, c = image.shape
            if c != 3:
                raise ValueError(
                    f"{cam}: expected RGB HWC3, got {image.shape}"
                )
            if tuple(depth.shape) != (h, w):
                raise ValueError(
                    f"{cam}: RGB/depth mismatch "
                    f"rgb={image.shape}, depth={depth.shape}"
                )

            if expected_hw is None:
                expected_hw = (h, w)
            elif (h, w) != expected_hw:
                raise ValueError(
                    "ACT camera tensors must share one HxW. "
                    f"Expected {expected_hw}, got {cam}={(h,w)}."
                )

            self.ds_images[cam] = self.image_group.create_dataset(
                cam,
                shape=(0, h, w, 3),
                maxshape=(None, h, w, 3),
                dtype=np.uint8,
                chunks=(1, h, w, 3),
                compression="lzf",
            )
            self.ds_depths[cam] = self.depth_group.create_dataset(
                cam,
                shape=(0, h, w),
                maxshape=(None, h, w),
                dtype=np.uint16,
                chunks=(1, h, w),
                compression="lzf",
            )
            self.image_shapes[cam] = tuple(image.shape)
            self.depth_shapes[cam] = tuple(depth.shape)

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
        self.ds_camera_host_ts = {
            cam: make_ts(f"{cam}_camera_host_monotonic")
            for cam in self.camera_names
        }
        self.ds_image_age = {
            cam: make_ts(f"{cam}_image_age_s")
            for cam in self.camera_names
        }
        self.ds_camera_sync_delta = make_ts("camera_sync_delta_s")
        self.ds_action_sdk_ts = make_ts("action_sdk")
        self.ds_qpos_sdk_ts = make_ts("qpos_sdk")
        self.ds_action_age = make_ts("action_age_s")

        self.ds_action_updated = self.file.create_dataset(
            "action_updated",
            shape=(0,),
            maxshape=(None,),
            dtype=np.uint8,
            chunks=(512,),
        )

        self.initialized = True

    @staticmethod
    def _append(ds, data):
        old = ds.shape[0]
        n = data.shape[0]
        ds.resize(old + n, axis=0)
        ds[old : old + n] = data

    def add(self, sample):
        if not self.initialized:
            self._init_datasets(sample)
        else:
            for cam in self.camera_names:
                image = sample[f"{cam}_image"]
                depth = sample[f"{cam}_depth"]
                if tuple(image.shape) != self.image_shapes[cam]:
                    raise RuntimeError(
                        f"{cam} RGB resolution changed: "
                        f"{self.image_shapes[cam]} -> {image.shape}"
                    )
                if tuple(depth.shape) != self.depth_shapes[cam]:
                    raise RuntimeError(
                        f"{cam} depth resolution changed: "
                        f"{self.depth_shapes[cam]} -> {depth.shape}"
                    )

        self.buffers.append(sample)
        if len(self.buffers) >= self.batch_size:
            self.flush()

    def flush(self):
        if not self.buffers:
            return

        qpos = np.stack(
            [x["qpos"] for x in self.buffers]
        ).astype(np.float32)
        action = np.stack(
            [x["action"] for x in self.buffers]
        ).astype(np.float32)
        qpos_raw = np.stack(
            [x["qpos_raw"] for x in self.buffers]
        ).astype(np.int32)
        action_raw = np.stack(
            [x["action_raw"] for x in self.buffers]
        ).astype(np.int32)

        self._append(self.ds_qpos, qpos)
        self._append(self.ds_action, action)

        for cam in self.camera_names:
            images = np.stack(
                [x[f"{cam}_image"] for x in self.buffers]
            ).astype(np.uint8)
            depths = np.stack(
                [x[f"{cam}_depth"] for x in self.buffers]
            ).astype(np.uint16)
            self._append(self.ds_images[cam], images)
            self._append(self.ds_depths[cam], depths)

        self._append(self.ds_qpos_raw, qpos_raw)
        self._append(self.ds_action_raw, action_raw)

        self._append(
            self.ds_sample_ts,
            np.asarray(
                [x["sample_monotonic"] for x in self.buffers],
                dtype=np.float64,
            ),
        )

        for cam in self.camera_names:
            self._append(
                self.ds_camera_host_ts[cam],
                np.asarray(
                    [x[f"{cam}_camera_host_stamp"] for x in self.buffers],
                    dtype=np.float64,
                ),
            )
            self._append(
                self.ds_image_age[cam],
                np.asarray(
                    [x[f"{cam}_image_age_s"] for x in self.buffers],
                    dtype=np.float64,
                ),
            )

        self._append(
            self.ds_camera_sync_delta,
            np.asarray(
                [x["camera_sync_delta_s"] for x in self.buffers],
                dtype=np.float64,
            ),
        )
        self._append(
            self.ds_action_sdk_ts,
            np.asarray(
                [x["action_sdk_stamp"] for x in self.buffers],
                dtype=np.float64,
            ),
        )
        self._append(
            self.ds_qpos_sdk_ts,
            np.asarray(
                [x["qpos_sdk_stamp"] for x in self.buffers],
                dtype=np.float64,
            ),
        )
        self._append(
            self.ds_action_age,
            np.asarray(
                [x["action_age_s"] for x in self.buffers],
                dtype=np.float64,
            ),
        )
        self._append(
            self.ds_action_updated,
            np.asarray(
                [1 if x["action_updated"] else 0 for x in self.buffers],
                dtype=np.uint8,
            ),
        )

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
            "Check Windows agx_cando channel, CAN wiring, Follower mode, "
            "CAN termination, and robot power."
        )

    raise RuntimeError(
        "Follower is online, but no valid Leader action was observed. "
        "Move the Leader arm during startup so at least one control "
        "command can be cached."
    )


def depth_to_colormap(depth_mm, depth_min_mm, depth_max_mm):
    depth_mm = np.asarray(depth_mm, dtype=np.uint16)
    valid = depth_mm > 0
    vis = np.zeros(depth_mm.shape, dtype=np.uint8)
    if np.any(valid):
        clipped = np.clip(
            depth_mm.astype(np.float32),
            float(depth_min_mm),
            float(depth_max_mm),
        )
        norm = (
            (clipped - float(depth_min_mm))
            / max(1.0, float(depth_max_mm) - float(depth_min_mm))
        )
        vis[valid] = np.clip(norm[valid] * 255.0, 0, 255).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)


def draw_preview(
    rgb,
    depth_mm,
    state,
    sample_count=0,
    qpos_hz=0.0,
    action_state="HOLD",
    image_age_ms=0.0,
    depth_min_mm=150.0,
    depth_max_mm=1500.0,
):
    """Draw synchronized RGB and metric-depth preview."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    depth_vis = depth_to_colormap(depth_mm, depth_min_mm, depth_max_mm)

    if depth_vis.shape[:2] != bgr.shape[:2]:
        depth_vis = cv2.resize(
            depth_vis,
            (bgr.shape[1], bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    canvas = np.hstack([bgr, depth_vis])

    if state == "READY":
        label = "READY | S: start recording | Q/ESC: quit"
        color = (0, 220, 255)
    else:
        label = "REC | Q/ESC: stop & save"
        color = (0, 0, 255)

    cv2.rectangle(
        canvas,
        (0, 0),
        (canvas.shape[1], 78),
        (0, 0, 0),
        thickness=-1,
    )

    valid_ratio = (
        float(np.count_nonzero(depth_mm)) / float(depth_mm.size)
        if depth_mm.size
        else 0.0
    )

    cv2.putText(
        canvas,
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
        f"rgbd_age={image_age_ms:.0f}ms  "
        f"depth_valid={valid_ratio * 100.0:.0f}%"
    )

    cv2.putText(
        canvas,
        status,
        (12, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        "RGB",
        (12, canvas.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "DEPTH (metric mm, aligned to RGB)",
        (bgr.shape[1] + 12, canvas.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return canvas



def draw_dual_preview(
    wrist_rgb,
    wrist_depth,
    external_rgb,
    external_depth,
    state,
    sample_count,
    qpos_hz,
    action_state,
    wrist_age_ms,
    external_age_ms,
    sync_delta_ms,
    wrist_depth_min_mm,
    wrist_depth_max_mm,
    external_depth_min_mm,
    external_depth_max_mm,
):
    wrist_panel = draw_preview(
        wrist_rgb,
        wrist_depth,
        state=state,
        sample_count=sample_count,
        qpos_hz=qpos_hz,
        action_state=action_state,
        image_age_ms=wrist_age_ms,
        depth_min_mm=wrist_depth_min_mm,
        depth_max_mm=wrist_depth_max_mm,
    )
    external_panel = draw_preview(
        external_rgb,
        external_depth,
        state=state,
        sample_count=sample_count,
        qpos_hz=qpos_hz,
        action_state=action_state,
        image_age_ms=external_age_ms,
        depth_min_mm=external_depth_min_mm,
        depth_max_mm=external_depth_max_mm,
    )

    target_w = min(wrist_panel.shape[1], external_panel.shape[1])
    if wrist_panel.shape[1] != target_w:
        wrist_panel = cv2.resize(
            wrist_panel,
            (
                target_w,
                int(wrist_panel.shape[0] * target_w / wrist_panel.shape[1]),
            ),
        )
    if external_panel.shape[1] != target_w:
        external_panel = cv2.resize(
            external_panel,
            (
                target_w,
                int(external_panel.shape[0] * target_w / external_panel.shape[1]),
            ),
        )

    cv2.putText(
        wrist_panel,
        "WRIST / ORBBEC",
        (12, 104),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        external_panel,
        f"EXTERNAL / REALSENSE | sync={sync_delta_ms:.1f}ms",
        (12, 104),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return np.vstack([wrist_panel, external_panel])


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Record PiPER-X Leader/Follower + wrist Orbbec RGB-D + "
            "external RealSense RGB-D for dual-camera ACT."
        )
    )

    parser.add_argument("--can-channel", default="0")
    parser.add_argument("--can-backend", default="agx_cando")
    parser.add_argument(
        "--judge-flag",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    parser.add_argument("--wrist-width", type=int, default=640)
    parser.add_argument("--wrist-height", type=int, default=480)
    parser.add_argument("--wrist-fps", type=int, default=30)
    parser.add_argument(
        "--wrist-depth-align",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--wrist-depth-min-mm", type=float, default=150.0)
    parser.add_argument("--wrist-depth-max-mm", type=float, default=1500.0)

    parser.add_argument("--external-width", type=int, default=640)
    parser.add_argument("--external-height", type=int, default=480)
    parser.add_argument("--external-fps", type=int, default=30)
    parser.add_argument(
        "--external-serial",
        default="",
        help="Optional RealSense serial number when more than one device is attached.",
    )
    parser.add_argument("--external-depth-min-mm", type=float, default=250.0)
    parser.add_argument("--external-depth-max-mm", type=float, default=3000.0)

    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Dataset sampling rate.",
    )
    parser.add_argument(
        "--max-image-age",
        type=float,
        default=0.10,
        help="Skip a sample if either camera frame is older than this.",
    )
    parser.add_argument(
        "--max-camera-sync-delta",
        type=float,
        default=0.05,
        help="Maximum allowed host timestamp difference between wrist and external frames.",
    )

    parser.add_argument(
        "--dataset-dir",
        default="../Dataset/dataset_stand_bottle_dual_rgbd",
    )
    parser.add_argument("--episode-index", type=int, default=None)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--startup-timeout", type=float, default=10.0)
    parser.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.max_image_age <= 0:
        raise ValueError("--max-image-age must be > 0")
    if args.max_camera_sync_delta <= 0:
        raise ValueError("--max-camera-sync-delta must be > 0")

    for name, low, high in (
        ("wrist", args.wrist_depth_min_mm, args.wrist_depth_max_mm),
        ("external", args.external_depth_min_mm, args.external_depth_max_mm),
    ):
        if low < 0 or high <= low:
            raise ValueError(
                f"{name}: require 0 <= depth_min_mm < depth_max_mm"
            )

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_path = next_episode_path(
        dataset_dir,
        args.episode_index,
    )
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing file: {output_path}"
        )

    wrist_camera = None
    external_camera = None
    robot = None
    writer = None

    try:
        print("[INFO] Starting wrist Orbbec DaBai DC1...")
        wrist_camera = OrbbecCamera(
            width=args.wrist_width,
            height=args.wrist_height,
            fps=args.wrist_fps,
            depth_align=args.wrist_depth_align,
            depth_min_mm=args.wrist_depth_min_mm,
            depth_max_mm=args.wrist_depth_max_mm,
        )
        wrist_camera.wait_until_ready(args.startup_timeout)

        print("[INFO] Starting external Intel RealSense...")
        external_camera = RealSenseCamera(
            width=args.external_width,
            height=args.external_height,
            fps=args.external_fps,
            depth_min_mm=args.external_depth_min_mm,
            depth_max_mm=args.external_depth_max_mm,
            serial=args.external_serial,
        )
        external_camera.wait_until_ready(args.startup_timeout)

        # ACT stacks camera tensors. Keep both cameras at identical HxW.
        wrist_pack = wrist_camera.get_latest()
        external_pack = external_camera.get_latest()
        if wrist_pack is None or external_pack is None:
            raise RuntimeError("Both cameras must be ready.")
        if wrist_pack[0].shape[:2] != external_pack[0].shape[:2]:
            raise RuntimeError(
                "Wrist and external camera RGB resolutions must match. "
                f"wrist={wrist_pack[0].shape[:2]}, "
                f"external={external_pack[0].shape[:2]}"
            )

        print(
            f"[INFO] Connecting PiPER on Windows: "
            f"{args.can_backend}:{args.can_channel} ..."
        )
        robot = PiperReader(
            can_channel=args.can_channel,
            can_backend=args.can_backend,
            judge_flag=args.judge_flag,
        )
        wait_for_robot(
            robot,
            args.startup_timeout,
        )

        print("\nDataset mapping:")
        print("  /observations/qpos             = Follower actual [q1..q6, gripper]")
        print("  /action                        = Leader target  [q1..q6, gripper]")
        print("  /observations/images/wrist     = Orbbec RGB uint8")
        print("  /observations/depth/wrist      = Orbbec aligned depth uint16 mm")
        print("  /observations/images/external  = RealSense RGB uint8")
        print("  /observations/depth/external   = RealSense aligned depth uint16 mm")
        print(f"  sampling                       = {args.fps:.1f} Hz")
        print(f"  output                         = {output_path}\n")

        latest_robot = robot.read()

        if args.preview:
            window_name = "PiPER ACT - Dual RGB-D Cameras"
            cv2.namedWindow(
                window_name,
                cv2.WINDOW_NORMAL,
            )
            print(
                "[READY] Press S to START recording. "
                "Press Q or ESC to quit."
            )

            while True:
                latest_robot = robot.read()
                wpack = wrist_camera.get_latest()
                epack = external_camera.get_latest()

                if wpack is not None and epack is not None:
                    w_rgb, w_depth, w_stamp = wpack
                    e_rgb, e_depth, e_stamp = epack
                    now = time.monotonic()

                    preview = draw_dual_preview(
                        w_rgb,
                        w_depth,
                        e_rgb,
                        e_depth,
                        state="READY",
                        sample_count=0,
                        qpos_hz=latest_robot.get("qpos_hz", 0.0),
                        action_state=(
                            "NEW"
                            if latest_robot.get("action_updated", False)
                            else "HOLD"
                        ),
                        wrist_age_ms=max(0.0, now - w_stamp) * 1000.0,
                        external_age_ms=max(0.0, now - e_stamp) * 1000.0,
                        sync_delta_ms=abs(w_stamp - e_stamp) * 1000.0,
                        wrist_depth_min_mm=args.wrist_depth_min_mm,
                        wrist_depth_max_mm=args.wrist_depth_max_mm,
                        external_depth_min_mm=args.external_depth_min_mm,
                        external_depth_max_mm=args.external_depth_max_mm,
                    )
                    cv2.imshow(
                        window_name,
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

        writer = HDF5EpisodeWriter(
            output_path,
            sample_rate_hz=args.fps,
            can_channel=args.can_channel,
            can_backend=args.can_backend,
            wrist_config={
                "backend": "pyorbbecsdk",
                "width_requested": int(args.wrist_width),
                "height_requested": int(args.wrist_height),
                "fps_requested": int(args.wrist_fps),
                "depth_min_mm": float(args.wrist_depth_min_mm),
                "depth_max_mm": float(args.wrist_depth_max_mm),
                "depth_aligned_to_color": bool(args.wrist_depth_align),
            },
            external_config={
                "backend": "pyrealsense2",
                "serial": str(getattr(external_camera, "device_serial", "")),
                "width_requested": int(args.external_width),
                "height_requested": int(args.external_height),
                "fps_requested": int(args.external_fps),
                "depth_min_mm": float(args.external_depth_min_mm),
                "depth_max_mm": float(args.external_depth_max_mm),
                "depth_aligned_to_color": True,
            },
        )

        print("[REC] Dual-camera recording started.")

        stop_event = threading.Event()
        if not args.preview and args.duration <= 0:
            def wait_enter():
                try:
                    input("\n[REC] Press Enter to STOP and save.\n")
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

        last_w = None
        last_e = None
        latest_robot = robot.read()

        skipped_no_camera = 0
        skipped_stale = 0
        skipped_sync = 0
        skipped_robot = 0
        last_print_count = 0

        window_name = "PiPER ACT - Dual RGB-D Cameras"

        while not stop_event.is_set():
            now = time.monotonic()

            if args.duration > 0 and now - start_time >= args.duration:
                break

            wpack = wrist_camera.get_latest()
            epack = external_camera.get_latest()
            if wpack is not None:
                last_w = wpack
            if epack is not None:
                last_e = epack

            if args.preview and last_w is not None and last_e is not None:
                w_rgb, w_depth, w_stamp = last_w
                e_rgb, e_depth, e_stamp = last_e

                current_count = writer.count + len(writer.buffers)

                preview = draw_dual_preview(
                    w_rgb,
                    w_depth,
                    e_rgb,
                    e_depth,
                    state="REC",
                    sample_count=current_count,
                    qpos_hz=latest_robot.get("qpos_hz", 0.0),
                    action_state=(
                        "NEW"
                        if latest_robot.get("action_updated", False)
                        else "HOLD"
                    ),
                    wrist_age_ms=max(0.0, now - w_stamp) * 1000.0,
                    external_age_ms=max(0.0, now - e_stamp) * 1000.0,
                    sync_delta_ms=abs(w_stamp - e_stamp) * 1000.0,
                    wrist_depth_min_mm=args.wrist_depth_min_mm,
                    wrist_depth_max_mm=args.wrist_depth_max_mm,
                    external_depth_min_mm=args.external_depth_min_mm,
                    external_depth_max_mm=args.external_depth_max_mm,
                )
                cv2.imshow(
                    window_name,
                    preview,
                )

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break

            if now < next_tick:
                time.sleep(
                    min(
                        next_tick - now,
                        0.002,
                    )
                )
                continue

            if now - next_tick > 2.0 * period:
                next_tick = now

            sample_stamp = next_tick
            next_tick += period

            if last_w is None or last_e is None:
                skipped_no_camera += 1
                continue

            w_rgb, w_depth, w_stamp = last_w
            e_rgb, e_depth, e_stamp = last_e

            w_age = now - w_stamp
            e_age = now - e_stamp

            if (
                w_age > args.max_image_age
                or e_age > args.max_image_age
            ):
                skipped_stale += 1
                continue

            sync_delta = abs(w_stamp - e_stamp)
            if sync_delta > args.max_camera_sync_delta:
                skipped_sync += 1
                continue

            latest_robot = robot.read()

            if (
                latest_robot["qpos_hz"] <= 0.0
                or not latest_robot["ready"]
            ):
                skipped_robot += 1
                continue

            writer.add(
                {
                    "wrist_image": w_rgb,
                    "wrist_depth": w_depth,
                    "external_image": e_rgb,
                    "external_depth": e_depth,
                    "qpos": latest_robot["qpos"],
                    "action": latest_robot["action"],
                    "qpos_raw": latest_robot["qpos_raw"],
                    "action_raw": latest_robot["action_raw"],
                    "sample_monotonic": sample_stamp,
                    "wrist_camera_host_stamp": w_stamp,
                    "external_camera_host_stamp": e_stamp,
                    "wrist_image_age_s": w_age,
                    "external_image_age_s": e_age,
                    "camera_sync_delta_s": sync_delta,
                    "action_sdk_stamp": latest_robot["action_sdk_stamp"],
                    "qpos_sdk_stamp": latest_robot["qpos_sdk_stamp"],
                    "action_age_s": latest_robot["action_age_s"],
                    "action_updated": latest_robot["action_updated"],
                }
            )

            current_count = writer.count + len(writer.buffers)

            if (
                current_count
                >= last_print_count + max(1, int(args.fps))
            ):
                last_print_count = current_count
                print(
                    f"\r[REC] {current_count:5d} samples | "
                    f"wrist_age={w_age*1000:5.1f}ms | "
                    f"external_age={e_age*1000:5.1f}ms | "
                    f"sync={sync_delta*1000:5.1f}ms | "
                    f"skip no/stale/sync/robot="
                    f"{skipped_no_camera}/{skipped_stale}/"
                    f"{skipped_sync}/{skipped_robot}",
                    end="",
                    flush=True,
                )

        print("\n[INFO] Stopping recording...")

    except KeyboardInterrupt:
        print("\n[INFO] Ctrl+C received; saving current episode...")

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
        if wrist_camera is not None:
            wrist_camera.close()
        if external_camera is not None:
            external_camera.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
