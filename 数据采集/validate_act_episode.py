#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Validate one PiPER ACT HDF5 episode.

Checks:
  1. Required ACT paths and matching lengths
  2. qpos/action shape, dtype, NaN/Inf
  3. Image shape/dtype and basic visual statistics
  4. Sample timestamps and measured sampling frequency
  5. Camera image age / synchronization
  6. Leader NEW/HOLD statistics
  7. qpos-action tracking error
  8. Optional RGB playback
  9. Optional qpos/action plots saved as PNG files
"""

import argparse
from pathlib import Path
import sys
import time

import cv2
import h5py
import numpy as np


NAMES = ["J1", "J2", "J3", "J4", "J5", "J6", "Gripper"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", help="Path to episode_xxxxxx.hdf5")
    parser.add_argument(
        "--play",
        action="store_true",
        help="Play recorded wrist-camera RGB frames.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Save qpos/action curves as separate PNG files.",
    )
    parser.add_argument(
        "--max-image-age",
        type=float,
        default=0.10,
        help="Warn when image age exceeds this value, default 0.10 s.",
    )
    return parser.parse_args()


def ok(text):
    print(f"[ OK ] {text}")


def warn(text):
    print(f"[WARN] {text}")


def fail(text):
    print(f"[FAIL] {text}")


def main():
    args = parse_args()
    path = Path(args.episode).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(path)

    errors = 0
    warnings = 0

    with h5py.File(path, "r") as f:
        print("=" * 72)
        print(f"Episode: {path}")
        print("=" * 72)

        required = [
            "observations/qpos",
            "action",
        ]

        for item in required:
            if item not in f:
                fail(f"Missing required dataset_stand_bottle: /{item}")
                errors += 1

        if "observations/images" not in f:
            fail("Missing /observations/images")
            errors += 1

        if errors:
            print(f"\nValidation stopped: {errors} structural error(s).")
            sys.exit(2)

        image_group = f["observations/images"]
        camera_names = list(image_group.keys())

        if not camera_names:
            fail("No camera dataset_stand_bottle under /observations/images")
            sys.exit(2)

        camera_name = f.attrs.get(
            "camera_name",
            camera_names[0],
        )

        if isinstance(camera_name, bytes):
            camera_name = camera_name.decode()

        if camera_name not in image_group:
            camera_name = camera_names[0]

        qpos = f["observations/qpos"]
        action = f["action"]
        images = image_group[camera_name]

        n_qpos = qpos.shape[0]
        n_action = action.shape[0]
        n_image = images.shape[0]

        print("\n[1] Structure")
        print(f"  qpos   : {qpos.shape}  {qpos.dtype}")
        print(f"  action : {action.shape}  {action.dtype}")
        print(f"  image  : {images.shape}  {images.dtype}")
        print(f"  camera : {camera_name}")

        if n_qpos == n_action == n_image and n_qpos > 0:
            ok(f"All main streams have the same length: T={n_qpos}")
        else:
            fail(
                f"Length mismatch: qpos={n_qpos}, "
                f"action={n_action}, image={n_image}"
            )
            errors += 1

        if qpos.ndim == 2 and qpos.shape[1] == 7:
            ok("qpos shape is [T, 7]")
        else:
            fail(f"Unexpected qpos shape: {qpos.shape}")
            errors += 1

        if action.ndim == 2 and action.shape[1] == 7:
            ok("action shape is [T, 7]")
        else:
            fail(f"Unexpected action shape: {action.shape}")
            errors += 1

        if images.ndim == 4 and images.shape[-1] == 3:
            ok("image shape is [T, H, W, 3]")
        else:
            fail(f"Unexpected image shape: {images.shape}")
            errors += 1

        q = qpos[:]
        a = action[:]

        print("\n[2] Numeric integrity")

        if np.isfinite(q).all():
            ok("qpos contains no NaN/Inf")
        else:
            fail("qpos contains NaN/Inf")
            errors += 1

        if np.isfinite(a).all():
            ok("action contains no NaN/Inf")
        else:
            fail("action contains NaN/Inf")
            errors += 1

        print("\n  Range:")
        for i, name in enumerate(NAMES):
            unit = "m" if i == 6 else "rad"
            print(
                f"  {name:7s} "
                f"qpos=[{q[:, i].min(): .5f}, {q[:, i].max(): .5f}] {unit}  "
                f"action=[{a[:, i].min(): .5f}, {a[:, i].max(): .5f}] {unit}"
            )

        print("\n[3] Sampling / timestamps")

        expected_hz = float(f.attrs.get("sample_rate_hz", 0.0))
        print(f"  requested sample rate: {expected_hz:.3f} Hz")

        if "timestamps/sample_monotonic" in f:
            ts = f["timestamps/sample_monotonic"][:]

            if len(ts) == n_qpos:
                ok("sample timestamp length matches T")
            else:
                fail(
                    f"timestamp length {len(ts)} != T {n_qpos}"
                )
                errors += 1

            if len(ts) > 1:
                dt = np.diff(ts)

                if np.all(dt > 0):
                    ok("sample timestamps are strictly increasing")
                else:
                    fail("sample timestamps are not strictly increasing")
                    errors += 1

                measured_hz = 1.0 / np.mean(dt)
                print(
                    f"  measured sample rate : "
                    f"{measured_hz:.3f} Hz"
                )
                print(
                    f"  dt mean/std/max      : "
                    f"{np.mean(dt)*1000:.2f} / "
                    f"{np.std(dt)*1000:.2f} / "
                    f"{np.max(dt)*1000:.2f} ms"
                )

                if expected_hz > 0:
                    relative_error = abs(
                        measured_hz - expected_hz
                    ) / expected_hz

                    if relative_error <= 0.10:
                        ok("measured sample rate is close to requested rate")
                    else:
                        warn(
                            "measured sample rate differs by more than 10%"
                        )
                        warnings += 1
        else:
            warn("No /timestamps/sample_monotonic")
            warnings += 1

        print("\n[4] Camera data / synchronization")

        if "timestamps/image_age_s" in f:
            image_age = f["timestamps/image_age_s"][:]

            print(
                f"  image age mean/p95/max: "
                f"{np.mean(image_age)*1000:.1f} / "
                f"{np.percentile(image_age, 95)*1000:.1f} / "
                f"{np.max(image_age)*1000:.1f} ms"
            )

            stale = int(
                np.sum(image_age > args.max_image_age)
            )

            if stale == 0:
                ok(
                    f"No saved image exceeds "
                    f"{args.max_image_age*1000:.0f} ms age"
                )
            else:
                warn(
                    f"{stale}/{len(image_age)} saved frames exceed "
                    f"{args.max_image_age*1000:.0f} ms"
                )
                warnings += 1
        else:
            warn("No /timestamps/image_age_s")
            warnings += 1

        # Sample image quality instead of loading the entire image dataset_stand_bottle.
        sample_ids = np.linspace(
            0,
            max(0, n_image - 1),
            num=min(20, n_image),
            dtype=int,
        )

        means = []
        stds = []

        for idx in sample_ids:
            img = images[idx]
            means.append(float(np.mean(img)))
            stds.append(float(np.std(img)))

        if means:
            print(
                f"  sampled image mean intensity: "
                f"{np.mean(means):.2f}"
            )
            print(
                f"  sampled image std          : "
                f"{np.mean(stds):.2f}"
            )

            if np.mean(stds) > 3.0:
                ok("sampled RGB frames are not visually blank/constant")
            else:
                warn("sampled RGB frames have very low variation")
                warnings += 1

        print("\n[5] Leader action HOLD behavior")

        if "action_updated" in f:
            updated = f["action_updated"][:].astype(bool)
            new_count = int(updated.sum())
            hold_count = int(len(updated) - new_count)
            new_ratio = (
                new_count / len(updated)
                if len(updated)
                else 0.0
            )

            print(f"  NEW samples : {new_count}")
            print(f"  HOLD samples: {hold_count}")
            print(f"  NEW ratio   : {new_ratio*100:.1f}%")
            ok("action_updated diagnostic exists")
        else:
            warn("No /action_updated diagnostic")
            warnings += 1

        if "timestamps/action_age_s" in f:
            action_age = f["timestamps/action_age_s"][:]
            finite_age = action_age[np.isfinite(action_age)]

            if len(finite_age):
                print(
                    f"  action age mean/p95/max: "
                    f"{np.mean(finite_age)*1000:.1f} / "
                    f"{np.percentile(finite_age,95)*1000:.1f} / "
                    f"{np.max(finite_age)*1000:.1f} ms"
                )

        print("\n[6] Follower tracking relative to Leader target")

        error = q - a
        abs_error = np.abs(error)

        for i, name in enumerate(NAMES):
            unit = "mm" if i == 6 else "deg"

            if i == 6:
                mae = np.mean(abs_error[:, i]) * 1000.0
                p95 = np.percentile(abs_error[:, i], 95) * 1000.0
                mx = np.max(abs_error[:, i]) * 1000.0
            else:
                scale = 180.0 / np.pi
                mae = np.mean(abs_error[:, i]) * scale
                p95 = np.percentile(abs_error[:, i], 95) * scale
                mx = np.max(abs_error[:, i]) * scale

            print(
                f"  {name:7s} MAE/P95/MAX: "
                f"{mae:7.3f} / {p95:7.3f} / {mx:7.3f} {unit}"
            )

        # ----------------------------------------------------
        # Optional playback
        # ----------------------------------------------------
        if args.play:
            hz = expected_hz if expected_hz > 0 else 20.0
            delay_ms = max(1, int(round(1000.0 / hz)))

            cv2.namedWindow(
                f"ACT Episode - {camera_name}",
                cv2.WINDOW_NORMAL,
            )

            paused = False
            i = 0

            print(
                "\n[PLAY] SPACE pause/resume | "
                "A/D previous/next when paused | Q/ESC quit"
            )

            while i < n_image:
                rgb = images[i]
                bgr = cv2.cvtColor(
                    rgb,
                    cv2.COLOR_RGB2BGR,
                )

                cv2.putText(
                    bgr,
                    f"{i+1}/{n_image}",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                cv2.imshow(
                    f"ACT Episode - {camera_name}",
                    bgr,
                )

                key = cv2.waitKey(
                    0 if paused else delay_ms
                ) & 0xFF

                if key in (ord("q"), 27):
                    break

                if key == ord(" "):
                    paused = not paused
                    continue

                if paused and key in (ord("a"), ord("A")):
                    i = max(0, i - 1)
                    continue

                if paused and key in (ord("d"), ord("D")):
                    i = min(n_image - 1, i + 1)
                    continue

                if not paused:
                    i += 1

            cv2.destroyAllWindows()

        # ----------------------------------------------------
        # Optional plots
        # ----------------------------------------------------
        if args.plots:
            import matplotlib.pyplot as plt

            plot_dir = path.parent / (
                path.stem + "_validation"
            )
            plot_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            x = np.arange(n_qpos)
            if "timestamps/sample_monotonic" in f:
                ts = f["timestamps/sample_monotonic"][:]
                if len(ts) == n_qpos:
                    x = ts - ts[0]

            xlabel = (
                "Time (s)"
                if not np.array_equal(
                    x,
                    np.arange(n_qpos),
                )
                else "Sample"
            )

            for i, name in enumerate(NAMES):
                fig = plt.figure(figsize=(10, 4))
                ax = fig.add_subplot(111)

                if i == 6:
                    ax.plot(x, q[:, i] * 1000.0, label="qpos")
                    ax.plot(x, a[:, i] * 1000.0, label="action")
                    ax.set_ylabel("Gripper (mm)")
                else:
                    scale = 180.0 / np.pi
                    ax.plot(x, q[:, i] * scale, label="qpos")
                    ax.plot(x, a[:, i] * scale, label="action")
                    ax.set_ylabel(f"{name} (deg)")

                ax.set_xlabel(xlabel)
                ax.set_title(
                    f"{path.stem}: {name} qpos vs action"
                )
                ax.grid(True, alpha=0.3)
                ax.legend()
                fig.tight_layout()

                fig.savefig(
                    plot_dir / f"{name.lower()}_qpos_action.png",
                    dpi=160,
                )
                plt.close(fig)

            print(f"\n[PLOTS] Saved to: {plot_dir}")

        print("\n" + "=" * 72)

        if errors == 0 and warnings == 0:
            print("RESULT: PASS - no structural/synchronization warning found")
        elif errors == 0:
            print(
                f"RESULT: USABLE WITH REVIEW - "
                f"{warnings} warning(s), 0 structural errors"
            )
        else:
            print(
                f"RESULT: FAIL - "
                f"{errors} error(s), {warnings} warning(s)"
            )

        print("=" * 72)

        if errors:
            sys.exit(2)


if __name__ == "__main__":
    main()
