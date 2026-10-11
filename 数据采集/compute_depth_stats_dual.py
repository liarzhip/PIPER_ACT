#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np

_EPISODE_RE = re.compile(r"episode_(\d+)\.hdf5$", re.IGNORECASE)


def discover_episode_files(dataset_dir: Path) -> list[Path]:
    dataset_dir = dataset_dir.expanduser().resolve()
    files = [
        p for p in dataset_dir.glob("episode_*.hdf5")
        if _EPISODE_RE.fullmatch(p.name)
    ]
    files.sort(key=lambda p: int(_EPISODE_RE.fullmatch(p.name).group(1)))
    if not files:
        raise FileNotFoundError(f"No episode_*.hdf5 in {dataset_dir}")
    return files


def split_episode_files(files, train_ratio, seed):
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")
    if len(files) < 2:
        raise ValueError("At least 2 episodes are required.")
    idx = np.random.default_rng(seed).permutation(len(files))
    n_train = max(1, min(len(files) - 1, int(round(len(files) * train_ratio))))
    return [files[i] for i in idx[:n_train]], [files[i] for i in idx[n_train:]]


class RunningStats:
    def __init__(self):
        self.count = 0
        self.sum = 0.0
        self.sum_sq = 0.0

    def update(self, x):
        x = np.asarray(x, np.float64).reshape(-1)
        if x.size == 0:
            return
        self.count += int(x.size)
        self.sum += float(x.sum())
        self.sum_sq += float(np.square(x).sum())

    def finish(self):
        if self.count == 0:
            raise RuntimeError("No valid depth pixels were found.")
        mean = self.sum / self.count
        var = max(self.sum_sq / self.count - mean * mean, 0.0)
        return float(mean), float(np.sqrt(var))


def compute_one_camera(
    files,
    camera_name,
    depth_min_mm,
    depth_max_mm,
    frame_chunk=8,
):
    key = f"/observations/depth/{camera_name}"
    stats = RunningStats()
    total = invalid = valid = low = high = frames_total = 0
    scale = float(depth_max_mm - depth_min_mm)

    for i, path in enumerate(files, 1):
        with h5py.File(path, "r") as root:
            if key not in root:
                raise KeyError(f"{path.name}: missing {key}")
            ds = root[key]
            if ds.ndim != 3:
                raise ValueError(f"{path.name}: {key} must be [T,H,W], got {ds.shape}")
            frames_total += int(ds.shape[0])
            print(f"[{camera_name}] [{i:03d}/{len(files):03d}] {path.name}: {ds.shape[0]} frames")

            for start in range(0, int(ds.shape[0]), frame_chunk):
                depth = np.asarray(
                    ds[start:min(start + frame_chunk, ds.shape[0])],
                    np.float32,
                )
                total += int(depth.size)
                mask = np.isfinite(depth) & (depth > 0.0)
                invalid += int((~mask).sum())
                if not np.any(mask):
                    continue
                raw = depth[mask]
                valid += int(raw.size)
                low += int((raw < depth_min_mm).sum())
                high += int((raw > depth_max_mm).sum())

                clipped = np.clip(raw, depth_min_mm, depth_max_mm)
                normalized = (clipped - depth_min_mm) / scale
                stats.update(normalized)

    mean, std = stats.finish()
    return {
        "camera_name": camera_name,
        "depth_key": key,
        "depth_mean": mean,
        "depth_std": std,
        "depth_min_mm": float(depth_min_mm),
        "depth_max_mm": float(depth_max_mm),
        "num_files": len(files),
        "num_frames": int(frames_total),
        "num_total_pixels": int(total),
        "num_valid_pixels": int(valid),
        "valid_ratio": float(valid / total if total else 0.0),
        "invalid_ratio": float(invalid / total if total else 0.0),
        "clipped_low_ratio_of_valid": float(low / valid if valid else 0.0),
        "clipped_high_ratio_of_valid": float(high / valid if valid else 0.0),
    }


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute per-camera depth statistics for dual RGB-D ACT."
    )
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--camera-names", nargs="+", default=["wrist", "external"])

    p.add_argument("--wrist-depth-min-mm", type=float, default=150.0)
    p.add_argument("--wrist-depth-max-mm", type=float, default=1500.0)
    p.add_argument("--external-depth-min-mm", type=float, default=250.0)
    p.add_argument("--external-depth-max-mm", type=float, default=3000.0)

    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frame-chunk", type=int, default=8)
    p.add_argument("--use-all", action="store_true")
    p.add_argument("--output-json", default="")
    return p.parse_args()


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    files = discover_episode_files(dataset_dir)

    if args.use_all:
        selected = files
        val_files = []
    else:
        selected, val_files = split_episode_files(
            files,
            args.train_ratio,
            args.seed,
        )

    ranges = {
        "wrist": (
            args.wrist_depth_min_mm,
            args.wrist_depth_max_mm,
        ),
        "external": (
            args.external_depth_min_mm,
            args.external_depth_max_mm,
        ),
    }

    camera_stats = {}
    for cam in args.camera_names:
        if cam not in ranges:
            raise ValueError(
                f"No CLI depth range defined for camera {cam!r}. "
                "Use camera names wrist/external or extend this script."
            )
        dmin, dmax = ranges[cam]
        if dmax <= dmin:
            raise ValueError(f"{cam}: max depth must exceed min depth")
        camera_stats[cam] = compute_one_camera(
            selected,
            cam,
            dmin,
            dmax,
            args.frame_chunk,
        )

    result = {
        "format": "dual_rgbd_depth_stats_v1",
        "camera_names": list(args.camera_names),
        "train_ratio": None if args.use_all else float(args.train_ratio),
        "seed": None if args.use_all else int(args.seed),
        "files_used": [str(p) for p in selected],
        "validation_files": [str(p) for p in val_files],
        "cameras": camera_stats,
    }

    output = (
        Path(args.output_json).expanduser().resolve()
        if args.output_json
        else dataset_dir / "depth_stats_dual.json"
    )
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n================ RESULT ================")
    for cam, st in camera_stats.items():
        print(
            f"{cam:>8}: mean={st['depth_mean']:.8f} "
            f"std={st['depth_std']:.8f} "
            f"range={st['depth_min_mm']:.0f}~{st['depth_max_mm']:.0f}mm "
            f"valid={st['valid_ratio']*100:.1f}%"
        )
    print(f"[OK] Saved -> {output}")


if __name__ == "__main__":
    main()
