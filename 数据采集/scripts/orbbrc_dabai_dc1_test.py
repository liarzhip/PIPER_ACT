#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Orbbec DaBai DC1 Test
Ubuntu 22.04 + Python 3.10 + legacy pyorbbecsdk v1.x

功能：
1. 检查 pyorbbecsdk
2. 打印 SDK 版本
3. 检测 Orbbec 设备
4. 打印设备和 Sensor 信息
5. 启动 Color Stream
6. 使用 OpenCV 实时显示 RGB 图像

按 q 或 ESC 退出
"""

import sys
from pathlib import Path

import cv2
import numpy as np


# ============================================================
# 1. 指定你自己编译的 pyorbbecsdk
# ============================================================

ORBBEC_LIB = Path.home() / "piper_x/pyorbbecsdk/install/lib"

if not ORBBEC_LIB.exists():
    raise RuntimeError(
        f"找不到 Orbbec SDK 路径：{ORBBEC_LIB}"
    )

# 让 Python 能找到 pyorbbecsdk*.so
sys.path.insert(0, str(ORBBEC_LIB))


# ============================================================
# 2. 导入 Orbbec SDK
# ============================================================

try:
    from pyorbbecsdk import (
        Context,
        Config,
        Pipeline,
        OBSensorType,
        OBFormat,
        OBError,
        VideoStreamProfile,
        get_version,
    )

except ImportError as e:
    print("\n[ERROR] 无法导入 pyorbbecsdk")
    print(e)

    print("\n请先执行：")
    print(
        "export PYTHONPATH="
        "$HOME/piper_x/pyorbbecsdk/install/lib:$PYTHONPATH"
    )
    print(
        "export LD_LIBRARY_PATH="
        "$HOME/piper_x/pyorbbecsdk/install/lib:$LD_LIBRARY_PATH"
    )

    sys.exit(1)


ESC_KEY = 27


# ============================================================
# 3. ColorFrame -> OpenCV BGR
#
# 参考 Orbbec 官方 legacy examples/utils.py
# ============================================================

def frame_to_bgr_image(frame):

    width = frame.get_width()
    height = frame.get_height()
    color_format = frame.get_format()

    data = np.asanyarray(frame.get_data())

    # --------------------------------------------------------
    # RGB888
    # --------------------------------------------------------
    if color_format == OBFormat.RGB:

        image = np.resize(
            data,
            (height, width, 3),
        )

        return cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR,
        )

    # --------------------------------------------------------
    # BGR888
    # --------------------------------------------------------
    elif color_format == OBFormat.BGR:

        return np.resize(
            data,
            (height, width, 3),
        )

    # --------------------------------------------------------
    # MJPEG
    # --------------------------------------------------------
    elif color_format == OBFormat.MJPG:

        return cv2.imdecode(
            data,
            cv2.IMREAD_COLOR,
        )

    # --------------------------------------------------------
    # YUYV
    # --------------------------------------------------------
    elif color_format == OBFormat.YUYV:

        image = np.resize(
            data,
            (height, width, 2),
        )

        return cv2.cvtColor(
            image,
            cv2.COLOR_YUV2BGR_YUY2,
        )

    # --------------------------------------------------------
    # UYVY
    # --------------------------------------------------------
    elif color_format == OBFormat.UYVY:

        image = np.resize(
            data,
            (height, width, 2),
        )

        return cv2.cvtColor(
            image,
            cv2.COLOR_YUV2BGR_UYVY,
        )

    # --------------------------------------------------------
    # I420
    # --------------------------------------------------------
    elif color_format == OBFormat.I420:

        image = np.resize(
            data,
            (height * 3 // 2, width),
        )

        return cv2.cvtColor(
            image,
            cv2.COLOR_YUV2BGR_I420,
        )

    # --------------------------------------------------------
    # NV12
    # --------------------------------------------------------
    elif color_format == OBFormat.NV12:

        image = np.resize(
            data,
            (height * 3 // 2, width),
        )

        return cv2.cvtColor(
            image,
            cv2.COLOR_YUV2BGR_NV12,
        )

    # --------------------------------------------------------
    # NV21
    # --------------------------------------------------------
    elif color_format == OBFormat.NV21:

        image = np.resize(
            data,
            (height * 3 // 2, width),
        )

        return cv2.cvtColor(
            image,
            cv2.COLOR_YUV2BGR_NV21,
        )

    print(
        "[WARN] Unsupported color format:",
        color_format,
    )

    return None


# ============================================================
# 4. 检测设备
# ============================================================

def check_device():

    print("=" * 60)
    print("Orbbec DaBai DC1 Test")
    print("=" * 60)

    print(f"\n[INFO] SDK version: {get_version()}")

    context = Context()

    device_list = context.query_devices()

    count = device_list.get_count()

    print(f"[INFO] Device count: {count}")

    if count == 0:

        print("\n[ERROR] 没有检测到 Orbbec 设备。")

        print("\n请检查：")
        print("  1. lsusb | grep 2bc5")
        print("  2. VMware USB 是否连接给 Ubuntu")
        print("  3. udev rules 是否安装")
        print("  4. USB 3.0 / 3.1 是否启用")

        return False

    device = device_list.get_device_by_index(0)

    info = device.get_device_info()

    print("\n[INFO] Device info:")
    print(info)

    print("\n[INFO] Sensor list:")

    sensor_list = device.get_sensor_list()

    for i in range(sensor_list.get_count()):

        sensor = sensor_list.get_sensor_by_index(i)

        print(
            f"  Sensor {i}:",
            sensor.get_type(),
        )

    return True


# ============================================================
# 5. Color Viewer
# ============================================================

def run_color_camera():

    print("\n" + "=" * 60)
    print("Starting Color Camera")
    print("=" * 60)

    config = Config()
    pipeline = Pipeline()

    # --------------------------------------------------------
    # 获取 Color Sensor 支持的 profile
    # --------------------------------------------------------

    try:

        profile_list = pipeline.get_stream_profile_list(
            OBSensorType.COLOR_SENSOR
        )

        if profile_list is None:

            print("[ERROR] 找不到 COLOR_SENSOR")
            return

        # ----------------------------------------------------
        # 和官方 color_viewer.py 一样：
        # 优先尝试 RGB / 30FPS
        # ----------------------------------------------------

        try:

            color_profile: VideoStreamProfile = (
                profile_list.get_video_stream_profile(
                    640,
                    0,
                    OBFormat.RGB,
                    30,
                )
            )

            print(
                "[INFO] Using requested color profile:"
            )
            print(color_profile)

        except OBError as e:

            print(
                "\n[WARN] "
                "RGB 640 @ 30 FPS 不受支持："
            )

            print(e)

            print(
                "\n[INFO] Falling back "
                "to default color profile..."
            )

            color_profile = (
                profile_list
                .get_default_video_stream_profile()
            )

            print(
                "[INFO] Default color profile:"
            )
            print(color_profile)

        config.enable_stream(color_profile)

    except Exception as e:

        print("\n[ERROR] Color stream configuration failed:")
        print(e)

        return

    # --------------------------------------------------------
    # 启动相机
    # --------------------------------------------------------

    try:

        pipeline.start(config)

    except Exception as e:

        print("\n[ERROR] Pipeline start failed:")
        print(e)

        return

    print("\n[OK] Color stream started!")

    print("\nControls:")
    print("  Q   -> Quit")
    print("  ESC -> Quit")

    cv2.namedWindow(
        "DaBai DC1 Color",
        cv2.WINDOW_NORMAL,
    )

    frame_count = 0

    try:

        while True:

            # ------------------------------------------------
            # 官方 color_viewer 使用 100 ms timeout
            # ------------------------------------------------

            frames = pipeline.wait_for_frames(100)

            if frames is None:
                continue

            color_frame = frames.get_color_frame()

            if color_frame is None:
                continue

            # ------------------------------------------------
            # Orbbec frame -> OpenCV
            # ------------------------------------------------

            image = frame_to_bgr_image(
                color_frame
            )

            if image is None:

                print(
                    "[WARN] Failed to convert "
                    "color frame."
                )

                continue

            frame_count += 1

            # ------------------------------------------------
            # 在画面上显示基本信息
            # ------------------------------------------------

            h, w = image.shape[:2]

            text = (
                f"DaBai DC1 | "
                f"{w}x{h} | "
                f"frame {frame_count}"
            )

            cv2.putText(
                image,
                text,
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow(
                "DaBai DC1 Color",
                image,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == ESC_KEY:
                break

    except KeyboardInterrupt:

        print("\n[INFO] Ctrl+C")

    finally:

        print("\n[INFO] Stopping pipeline...")

        pipeline.stop()

        cv2.destroyAllWindows()

        print("[OK] Camera stopped.")


# ============================================================
# main
# ============================================================

def main():

    if not check_device():
        return

    run_color_camera()


if __name__ == "__main__":
    main()