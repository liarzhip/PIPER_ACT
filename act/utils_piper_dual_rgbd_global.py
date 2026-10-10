#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PiPER-X dual RGB-D ACT dataset utilities with GLOBAL recurrent history.

Each training sample contains:
    images: [num_cameras, 4, H, W]
    qpos[t]: [7]
    global_history: [(qpos[0], action[0]), ..., (qpos[t-1], action[t-1])]
    action[t:t+chunk_size]: [chunk_size, 7]

Camera order is explicit and must remain identical in training and deployment.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Sequence, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_EPISODE_RE = re.compile(r"episode_(\d+)\.hdf5$", re.IGNORECASE)


def _episode_number(path: Path) -> int:
    m = _EPISODE_RE.search(path.name)
    return int(m.group(1)) if m else 10**18


def discover_episode_files(dataset_dir: str | Path) -> List[Path]:
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(dataset_dir)

    files = [
        p
        for p in dataset_dir.glob("episode_*.hdf5")
        if _EPISODE_RE.fullmatch(p.name)
    ]
    files.sort(key=lambda p: (_episode_number(p), p.name))
    if not files:
        raise FileNotFoundError(f"No episode_*.hdf5 in {dataset_dir}")
    return files


def validate_episode_file(
    path: str | Path,
    camera_names: Sequence[str],
    state_dim: int,
) -> int:
    path = Path(path)

    with h5py.File(path, "r") as root:
        for key in ("/observations/qpos", "/action"):
            if key not in root:
                raise KeyError(f"{path.name}: missing {key}")

        qpos = root["/observations/qpos"]
        action = root["/action"]

        if qpos.ndim != 2 or qpos.shape[1] != state_dim:
            raise ValueError(
                f"{path.name}: qpos shape {qpos.shape}, expected [T,{state_dim}]"
            )
        if action.ndim != 2 or action.shape[1] != state_dim:
            raise ValueError(
                f"{path.name}: action shape {action.shape}, expected [T,{state_dim}]"
            )
        if qpos.shape[0] != action.shape[0]:
            raise ValueError(f"{path.name}: qpos/action length mismatch")

        T = int(action.shape[0])
        if T <= 0:
            raise ValueError(f"{path.name}: empty episode")

        common_hw = None
        for cam in camera_names:
            rgb_key = f"/observations/images/{cam}"
            dep_key = f"/observations/depth/{cam}"

            if rgb_key not in root or dep_key not in root:
                raise KeyError(
                    f"{path.name}: missing {rgb_key} or {dep_key}"
                )

            rgb = root[rgb_key]
            dep = root[dep_key]

            if rgb.shape[0] != T or dep.shape[0] != T:
                raise ValueError(
                    f"{path.name}: camera {cam} length mismatch"
                )

            if rgb.ndim != 4 or rgb.shape[-1] != 3:
                raise ValueError(
                    f"{path.name}: {cam} RGB shape {rgb.shape}, "
                    "expected [T,H,W,3]"
                )

            if dep.ndim != 3:
                raise ValueError(
                    f"{path.name}: {cam} depth shape {dep.shape}, "
                    "expected [T,H,W]"
                )

            if tuple(dep.shape[1:]) != tuple(rgb.shape[1:3]):
                raise ValueError(
                    f"{path.name}: {cam} RGB/depth size mismatch: "
                    f"rgb={rgb.shape}, depth={dep.shape}"
                )

            hw = tuple(rgb.shape[1:3])
            if common_hw is None:
                common_hw = hw
            elif hw != common_hw:
                raise ValueError(
                    f"{path.name}: all cameras must share the same HxW "
                    f"for ACT stacking. Expected {common_hw}, got {cam}={hw}."
                )

    return T


def get_norm_stats(
    episode_files: Sequence[str | Path],
    state_dim: int = 7,
):
    all_qpos = []
    all_action = []

    for path in episode_files:
        with h5py.File(path, "r") as root:
            all_qpos.append(
                torch.from_numpy(
                    np.asarray(
                        root["/observations/qpos"][()],
                        np.float32,
                    )
                )
            )
            all_action.append(
                torch.from_numpy(
                    np.asarray(
                        root["/action"][()],
                        np.float32,
                    )
                )
            )

    q = torch.cat(all_qpos, dim=0)
    a = torch.cat(all_action, dim=0)

    if q.shape[1] != state_dim or a.shape[1] != state_dim:
        raise ValueError("state_dim mismatch")

    q_mean = q.mean(0)
    q_std = torch.clamp(q.std(0), min=1e-2)
    a_mean = a.mean(0)
    a_std = torch.clamp(a.std(0), min=1e-2)

    return {
        "qpos_mean": q_mean.numpy(),
        "qpos_std": q_std.numpy(),
        "action_mean": a_mean.numpy(),
        "action_std": a_std.numpy(),
        "state_dim": int(state_dim),
        "num_normalization_samples": int(q.shape[0]),
    }


def split_episode_files(
    episode_files: Sequence[Path],
    train_ratio: float = 0.8,
    seed: int = 0,
) -> Tuple[List[Path], List[Path]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")
    if len(episode_files) < 2:
        raise ValueError("At least 2 episodes are required")

    idx = np.random.default_rng(seed).permutation(len(episode_files))
    n_train = max(
        1,
        min(
            len(episode_files) - 1,
            int(round(len(episode_files) * train_ratio)),
        ),
    )

    return (
        [episode_files[i] for i in idx[:n_train]],
        [episode_files[i] for i in idx[n_train:]],
    )


class PiperDualRGBDGlobalDataset(Dataset):
    def __init__(
        self,
        episode_files,
        camera_names,
        norm_stats,
        camera_depth_stats,
        chunk_size,
        state_dim=7,
        sample_stride=2,
        global_history_stride=1,
    ):
        super().__init__()

        self.episode_files = [Path(p) for p in episode_files]
        self.camera_names = list(camera_names)
        self.norm_stats = norm_stats
        self.camera_depth_stats = dict(camera_depth_stats)

        self.chunk_size = int(chunk_size)
        self.state_dim = int(state_dim)
        self.sample_stride = int(sample_stride)
        self.global_history_stride = int(global_history_stride)

        if (
            self.chunk_size <= 0
            or self.sample_stride <= 0
            or self.global_history_stride <= 0
        ):
            raise ValueError(
                "chunk_size/sample_stride/global_history_stride must be > 0"
            )

        self.episode_lengths = [
            validate_episode_file(
                p,
                self.camera_names,
                self.state_dim,
            )
            for p in self.episode_files
        ]

        self.sample_index = [
            (episode_index, t)
            for episode_index, T in enumerate(self.episode_lengths)
            for t in range(0, T, self.sample_stride)
        ]
        if not self.sample_index:
            raise RuntimeError("No timestep samples generated")

        self.q_mean = np.asarray(
            norm_stats["qpos_mean"],
            np.float32,
        )
        self.q_std = np.asarray(
            norm_stats["qpos_std"],
            np.float32,
        )
        self.a_mean = np.asarray(
            norm_stats["action_mean"],
            np.float32,
        )
        self.a_std = np.asarray(
            norm_stats["action_std"],
            np.float32,
        )

        for cam in self.camera_names:
            if cam not in self.camera_depth_stats:
                raise KeyError(
                    f"Missing depth stats for camera {cam!r}"
                )

        self._h5_cache = {}

    def __len__(self):
        return len(self.sample_index)

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_h5_cache"] = {}
        return d

    def _get_h5(self, path):
        key = str(path)
        root = self._h5_cache.get(key)
        if root is None or not root.id.valid:
            root = h5py.File(path, "r")
            self._h5_cache[key] = root
        return root

    def close(self):
        for root in self._h5_cache.values():
            try:
                root.close()
            except Exception:
                pass
        self._h5_cache.clear()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _make_rgbd(self, rgb, depth_mm, cam_name):
        stats = self.camera_depth_stats[cam_name]
        depth_mean = float(stats["depth_mean"])
        depth_min_mm = float(stats["depth_min_mm"])
        depth_max_mm = float(stats["depth_max_mm"])

        rgb_f = np.asarray(rgb, np.uint8).astype(np.float32) / 255.0
        depth_f = np.asarray(depth_mm, np.float32)

        valid = np.isfinite(depth_f) & (depth_f > 0.0)

        # Fill invalid depth with the training mean in [0,1].
        # After policy standardization this becomes exactly zero.
        depth_norm = np.full_like(
            depth_f,
            depth_mean,
            dtype=np.float32,
        )

        if np.any(valid):
            clipped = np.clip(
                depth_f[valid],
                depth_min_mm,
                depth_max_mm,
            )
            depth_norm[valid] = (
                clipped - depth_min_mm
            ) / (
                depth_max_mm - depth_min_mm
            )

        return np.concatenate(
            [rgb_f, depth_norm[..., None]],
            axis=-1,
        ).astype(np.float32)

    def __getitem__(self, index):
        episode_index, t = self.sample_index[int(index)]
        path = self.episode_files[episode_index]
        root = self._get_h5(path)
        T = self.episode_lengths[episode_index]

        qpos = np.asarray(
            root["/observations/qpos"][t],
            np.float32,
        )

        all_cam_images = []
        expected_hw = None

        for cam in self.camera_names:
            rgb = np.asarray(
                root[f"/observations/images/{cam}"][t],
                np.uint8,
            )
            depth_mm = np.asarray(
                root[f"/observations/depth/{cam}"][t],
                np.uint16,
            )

            if tuple(depth_mm.shape) != tuple(rgb.shape[:2]):
                raise RuntimeError(
                    f"{path.name}: {cam} RGB/depth mismatch"
                )

            if expected_hw is None:
                expected_hw = tuple(rgb.shape[:2])
            elif tuple(rgb.shape[:2]) != expected_hw:
                raise RuntimeError(
                    f"{path.name}: camera sizes differ at t={t}: "
                    f"expected={expected_hw}, {cam}={rgb.shape[:2]}"
                )

            all_cam_images.append(
                self._make_rgbd(
                    rgb,
                    depth_mm,
                    cam,
                )
            )

        hist_idx = np.arange(
            0,
            int(t),
            self.global_history_stride,
            dtype=np.int64,
        )

        if hist_idx.size:
            q_hist = np.asarray(
                root["/observations/qpos"][hist_idx],
                np.float32,
            )
            a_hist = np.asarray(
                root["/action"][hist_idx],
                np.float32,
            )
            q_hist = (q_hist - self.q_mean) / self.q_std
            a_hist = (a_hist - self.a_mean) / self.a_std
            global_history = np.concatenate(
                [q_hist, a_hist],
                axis=-1,
            ).astype(np.float32)
        else:
            global_history = np.zeros(
                (0, self.state_dim * 2),
                dtype=np.float32,
            )

        end = min(
            t + self.chunk_size,
            T,
        )

        target = np.asarray(
            root["/action"][t:end],
            np.float32,
        )
        valid_len = int(target.shape[0])

        padded = np.zeros(
            (self.chunk_size, self.state_dim),
            np.float32,
        )
        padded[:valid_len] = target

        is_pad = np.ones(
            self.chunk_size,
            np.bool_,
        )
        is_pad[:valid_len] = False

        image = (
            torch.from_numpy(np.stack(all_cam_images))
            .permute(0, 3, 1, 2)
            .contiguous()
            .float()
        )

        qpos_t = torch.from_numpy(
            (
                (qpos - self.q_mean)
                / self.q_std
            ).astype(np.float32)
        )

        hist_t = torch.from_numpy(
            global_history
        ).float()

        action_t = torch.from_numpy(
            (
                (padded - self.a_mean)
                / self.a_std
            ).astype(np.float32)
        )

        pad_t = torch.from_numpy(is_pad)

        return (
            image,
            qpos_t,
            hist_t,
            action_t,
            pad_t,
        )


def global_history_collate(batch):
    images, qpos, histories, actions, is_pad = zip(*batch)

    lengths = torch.tensor(
        [h.shape[0] for h in histories],
        dtype=torch.long,
    )

    hist_dim = histories[0].shape[-1] if histories else 14
    max_len = max(
        1,
        int(lengths.max().item()) if len(lengths) else 1,
    )

    padded_history = torch.zeros(
        (len(histories), max_len, hist_dim),
        dtype=torch.float32,
    )

    for i, h in enumerate(histories):
        if h.shape[0] > 0:
            padded_history[i, : h.shape[0]] = h

    return (
        torch.stack(images, 0),
        torch.stack(qpos, 0),
        padded_history,
        lengths,
        torch.stack(actions, 0),
        torch.stack(is_pad, 0),
    )


def load_data(
    dataset_dir,
    camera_names,
    batch_size_train,
    batch_size_val,
    chunk_size,
    state_dim=7,
    train_ratio=0.8,
    seed=0,
    num_workers=0,
    max_episodes=None,
    depth_stats_json=None,
    sample_stride=2,
    global_history_stride=1,
):
    dataset_dir = Path(dataset_dir).expanduser().resolve()

    depth_stats_path = (
        Path(depth_stats_json).expanduser().resolve()
        if depth_stats_json
        else dataset_dir / "depth_stats_dual.json"
    )

    if not depth_stats_path.exists():
        raise FileNotFoundError(
            f"Depth statistics not found: {depth_stats_path}. "
            "Run compute_depth_stats_dual.py first."
        )

    depth_document = json.loads(
        depth_stats_path.read_text(encoding="utf-8")
    )

    if "cameras" not in depth_document:
        raise KeyError(
            "depth_stats_dual.json must contain a 'cameras' dictionary"
        )

    if depth_document.get("train_ratio") is not None:
        if abs(
            float(depth_document["train_ratio"])
            - float(train_ratio)
        ) > 1e-12:
            raise ValueError(
                "Depth stats train_ratio does not match training."
            )

    if depth_document.get("seed") is not None:
        if int(depth_document["seed"]) != int(seed):
            raise ValueError(
                "Depth stats seed does not match training."
            )

    camera_depth_stats = {}
    for cam in camera_names:
        if cam not in depth_document["cameras"]:
            raise KeyError(
                f"depth_stats_dual.json has no statistics for {cam!r}"
            )

        stats = dict(depth_document["cameras"][cam])
        for key in (
            "depth_mean",
            "depth_std",
            "depth_min_mm",
            "depth_max_mm",
        ):
            if key not in stats:
                raise KeyError(
                    f"{cam} depth statistics missing {key}"
                )
        if float(stats["depth_std"]) <= 0:
            raise ValueError(
                f"{cam}: depth_std must be > 0"
            )
        camera_depth_stats[cam] = stats

    all_files = discover_episode_files(dataset_dir)
    if max_episodes is not None:
        all_files = all_files[: int(max_episodes)]

    episode_lengths = [
        validate_episode_file(
            p,
            camera_names,
            state_dim,
        )
        for p in all_files
    ]

    train_files, val_files = split_episode_files(
        all_files,
        train_ratio,
        seed,
    )

    norm_stats = get_norm_stats(
        train_files,
        state_dim,
    )

    norm_stats.update(
        {
            "camera_names": list(camera_names),
            "num_cameras": len(camera_names),
            "image_channels": 4,
            "camera_depth_stats": camera_depth_stats,
            "depth_stats_json": str(depth_stats_path),
            "sample_stride": int(sample_stride),
            "global_history_stride": int(global_history_stride),
            "global_history_input_dim": int(state_dim * 2),
        }
    )

    train_dataset = PiperDualRGBDGlobalDataset(
        train_files,
        camera_names,
        norm_stats,
        camera_depth_stats,
        chunk_size,
        state_dim=state_dim,
        sample_stride=sample_stride,
        global_history_stride=global_history_stride,
    )

    val_dataset = PiperDualRGBDGlobalDataset(
        val_files,
        camera_names,
        norm_stats,
        camera_depth_stats,
        chunk_size,
        state_dim=state_dim,
        sample_stride=sample_stride,
        global_history_stride=global_history_stride,
    )

    loader_kwargs = {
        "pin_memory": torch.cuda.is_available(),
        "num_workers": int(num_workers),
        "collate_fn": global_history_collate,
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=False,
        **loader_kwargs,
    )

    info = {
        "all_files": [str(p) for p in all_files],
        "train_files": [str(p) for p in train_files],
        "val_files": [str(p) for p in val_files],
        "episode_lengths": episode_lengths,
        "min_episode_len": int(min(episode_lengths)),
        "max_episode_len": int(max(episode_lengths)),
        "mean_episode_len": float(np.mean(episode_lengths)),
        "num_episodes": len(all_files),
        "num_train": len(train_files),
        "num_val": len(val_files),
        "num_train_samples": len(train_dataset),
        "num_val_samples": len(val_dataset),
        "camera_names": list(camera_names),
        "num_cameras": len(camera_names),
        "image_channels": 4,
        "camera_depth_stats": camera_depth_stats,
        "max_global_history_steps": int(
            max(episode_lengths) // global_history_stride
        ),
    }

    return (
        train_loader,
        val_loader,
        norm_stats,
        info,
    )


def compute_dict_mean(epoch_dicts):
    if not epoch_dicts:
        raise ValueError("epoch_dicts is empty")
    return {
        key: torch.stack(
            [d[key] for d in epoch_dicts]
        ).mean()
        for key in epoch_dicts[0]
    }


def detach_dict(data):
    return {
        key: value.detach()
        for key, value in data.items()
    }


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
