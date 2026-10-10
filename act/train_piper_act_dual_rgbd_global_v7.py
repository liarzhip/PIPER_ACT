#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train dual-camera RGB-D ACT + GLOBAL GRU memory on PiPER-X."""
from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from copy import deepcopy
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from policy_piper_dual_rgbd_global import ACTPolicy
from utils_piper_dual_rgbd_global import (
    compute_dict_mean,
    detach_dict,
    load_data,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--ckpt_dir", required=True)

    p.add_argument(
        "--camera_names",
        nargs="+",
        default=["wrist", "external"],
        help="Camera order used by ACT. Keep identical at deployment.",
    )

    p.add_argument("--state_dim", type=int, default=7)
    p.add_argument("--depth_stats_json", default=None)

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_epochs", type=int, default=50)

    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--lr_backbone", type=float, default=1e-5)

    p.add_argument(
        "--chunk_size",
        type=int,
        default=20,
        help="20 steps at 20 Hz = 1 second future action chunk.",
    )

    p.add_argument(
        "--sample_stride",
        type=int,
        default=2,
        help="Use every Nth raw timestep as an ACT training target.",
    )

    p.add_argument(
        "--global_history_stride",
        type=int,
        default=1,
        help="Use every Nth strictly-past qpos+action from episode start.",
    )

    p.add_argument(
        "--global_hidden_dim",
        type=int,
        default=128,
    )

    p.add_argument(
        "--global_residual_scale",
        type=float,
        default=1.0,
    )

    p.add_argument("--kl_weight", type=float, default=10.0)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--dim_feedforward", type=int, default=3200)

    p.add_argument("--enc_layers", type=int, default=4)
    p.add_argument("--dec_layers", type=int, default=7)
    p.add_argument("--nheads", type=int, default=8)
    p.add_argument("--backbone", default="resnet18")

    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_episodes", type=int, default=None)

    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--plot_every", type=int, default=10)
    return p.parse_args()


def forward_pass(data, policy):
    (
        image,
        qpos,
        global_history,
        global_lengths,
        action,
        is_pad,
    ) = data

    image = image.cuda(non_blocking=True)
    qpos = qpos.cuda(non_blocking=True)
    global_history = global_history.cuda(non_blocking=True)
    global_lengths = global_lengths.cuda(non_blocking=True)
    action = action.cuda(non_blocking=True)
    is_pad = is_pad.cuda(non_blocking=True)

    return policy(
        qpos,
        image,
        action,
        is_pad,
        global_history=global_history,
        global_history_lengths=global_lengths,
    )


def save_curves(train_hist, val_hist, ckpt_dir):
    if not train_hist:
        return

    for key in train_hist[0]:
        plt.figure(figsize=(8, 4))
        plt.plot(
            [x[key] for x in train_hist],
            label="train",
        )
        plt.plot(
            [x[key] for x in val_hist],
            label="val",
        )
        plt.xlabel("Epoch")
        plt.ylabel(key)
        plt.title(key)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            ckpt_dir / f"train_val_{key}.png",
            dpi=160,
        )
        plt.close()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    for name in (
        "chunk_size",
        "sample_stride",
        "global_history_stride",
        "global_hidden_dim",
        "num_epochs",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name} must be > 0")

    if args.state_dim != 7:
        raise ValueError("This PiPER setup expects state_dim=7")

    if len(args.camera_names) != 2:
        raise ValueError(
            "V7 is designed for exactly two cameras: wrist external."
        )

    set_seed(args.seed)

    ckpt_dir = Path(
        args.ckpt_dir
    ).expanduser().resolve()
    ckpt_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    architecture_config = {
        "camera_names": list(args.camera_names),
        "num_cameras": len(args.camera_names),
        "state_dim": int(args.state_dim),
        "chunk_size": int(args.chunk_size),
        "kl_weight": float(args.kl_weight),
        "hidden_dim": int(args.hidden_dim),
        "dim_feedforward": int(args.dim_feedforward),
        "enc_layers": int(args.enc_layers),
        "dec_layers": int(args.dec_layers),
        "nheads": int(args.nheads),
        "backbone": args.backbone,
        "image_channels": 4,
        "global_history_stride": int(args.global_history_stride),
        "global_hidden_dim": int(args.global_hidden_dim),
        "global_residual_scale": float(args.global_residual_scale),
    }

    config_path = ckpt_dir / "model_config.json"

    if config_path.exists():
        old_config = json.loads(
            config_path.read_text(encoding="utf-8")
        )

        changed = {
            key: (
                old_config.get(key),
                value,
            )
            for key, value in architecture_config.items()
            if old_config.get(key) != value
        }

        if changed:
            raise RuntimeError(
                "Checkpoint directory contains a different model configuration: "
                f"{changed}. Use a NEW --ckpt_dir."
            )

    existing_best = ckpt_dir / "policy_best.ckpt"
    if existing_best.exists() and not config_path.exists():
        old_state = torch.load(
            existing_best,
            map_location="cpu",
        )
        qkey = next(
            (
                k
                for k in old_state
                if k == "model.query_embed.weight"
                or k.endswith(".query_embed.weight")
            ),
            None,
        )
        if qkey is not None:
            old_chunk = int(
                old_state[qkey].shape[0]
            )
            if old_chunk != int(args.chunk_size):
                raise RuntimeError(
                    f"Existing checkpoint uses chunk_size={old_chunk}, "
                    f"but requested {args.chunk_size}. Use a NEW --ckpt_dir."
                )

    config_path.write_text(
        json.dumps(
            architecture_config,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    (
        train_loader,
        val_loader,
        stats,
        info,
    ) = load_data(
        dataset_dir=args.dataset_dir,
        camera_names=args.camera_names,
        batch_size_train=args.batch_size,
        batch_size_val=args.batch_size,
        chunk_size=args.chunk_size,
        state_dim=args.state_dim,
        train_ratio=args.train_ratio,
        seed=args.seed,
        num_workers=args.num_workers,
        max_episodes=args.max_episodes,
        depth_stats_json=args.depth_stats_json,
        sample_stride=args.sample_stride,
        global_history_stride=args.global_history_stride,
    )

    camera_depth_stats = stats["camera_depth_stats"]
    depth_means = [
        float(camera_depth_stats[cam]["depth_mean"])
        for cam in args.camera_names
    ]
    depth_stds = [
        float(camera_depth_stats[cam]["depth_std"])
        for cam in args.camera_names
    ]

    print(
        "\n========== PiPER DUAL RGB-D ACT + GLOBAL MEMORY =========="
    )
    print(
        f"Episodes: {info['num_episodes']} "
        f"(train={info['num_train']}, val={info['num_val']})"
    )
    print(
        f"Episode length: min={info['min_episode_len']} "
        f"mean={info['mean_episode_len']:.1f} "
        f"max={info['max_episode_len']}"
    )
    print(
        f"Training samples: train={info['num_train_samples']} "
        f"val={info['num_val_samples']}"
    )
    print(
        f"Cameras/order: {args.camera_names} | "
        f"image tensor=[B,{len(args.camera_names)},4,H,W]"
    )
    print(
        f"sample_stride={args.sample_stride} | "
        f"chunk_size={args.chunk_size}"
    )
    print(
        "GLOBAL past: episode start -> t-1 | "
        f"history_stride={args.global_history_stride} | "
        f"max_steps≈{info['max_global_history_steps']}"
    )
    print(
        f"Global GRU hidden={args.global_hidden_dim} | "
        f"residual_scale={args.global_residual_scale}"
    )
    for cam in args.camera_names:
        st = camera_depth_stats[cam]
        print(
            f"{cam:>8} depth: "
            f"{st['depth_min_mm']:.0f}~{st['depth_max_mm']:.0f} mm | "
            f"mean={st['depth_mean']:.8f} | "
            f"std={st['depth_std']:.8f}"
        )
    print(
        f"Batches/epoch: train={len(train_loader)} "
        f"val={len(val_loader)} | "
        f"GPU={torch.cuda.get_device_name(0)}"
    )
    print(
        "=========================================================\n"
    )

    stats["global_history_stride"] = int(args.global_history_stride)
    stats["global_hidden_dim"] = int(args.global_hidden_dim)
    stats["global_residual_scale"] = float(args.global_residual_scale)
    stats["global_memory"] = (
        "qpos+action_from_episode_start_strictly_past"
    )

    with open(
        ckpt_dir / "dataset_stats.pkl",
        "wb",
    ) as f:
        pickle.dump(stats, f)

    with open(
        ckpt_dir / "dataset_split.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            info,
            f,
            indent=2,
            ensure_ascii=False,
        )

    policy_config = {
        "lr": args.lr,
        "lr_backbone": args.lr_backbone,
        "num_queries": args.chunk_size,
        "kl_weight": args.kl_weight,
        "hidden_dim": args.hidden_dim,
        "dim_feedforward": args.dim_feedforward,
        "backbone": args.backbone,
        "enc_layers": args.enc_layers,
        "dec_layers": args.dec_layers,
        "nheads": args.nheads,
        "camera_names": list(args.camera_names),
        "state_dim": 7,
        "image_channels": 4,
        "depth_means": depth_means,
        "depth_stds": depth_stds,
        "global_history_stride": args.global_history_stride,
        "global_hidden_dim": args.global_hidden_dim,
        "global_residual_scale": args.global_residual_scale,
    }

    policy = ACTPolicy(
        policy_config
    ).cuda()

    optimizer = policy.configure_optimizers()

    train_hist = []
    val_hist = []
    min_val = math.inf
    best_epoch = -1
    best_state = None

    for epoch in tqdm(
        range(args.num_epochs),
        desc="Training ACT Dual RGB-D Global",
    ):
        policy.eval()
        val_dicts = []

        with torch.inference_mode():
            for data in val_loader:
                val_dicts.append(
                    detach_dict(
                        forward_pass(
                            data,
                            policy,
                        )
                    )
                )

        val_summary = compute_dict_mean(
            val_dicts
        )
        val_hist.append(
            {
                k: float(v.item())
                for k, v in val_summary.items()
            }
        )
        val_loss = float(
            val_summary["loss"].item()
        )

        if val_loss < min_val:
            min_val = val_loss
            best_epoch = epoch
            best_state = deepcopy(
                policy.state_dict()
            )
            torch.save(
                best_state,
                ckpt_dir / "policy_best.ckpt",
            )

        policy.train()
        train_dicts = []

        for data in train_loader:
            optimizer.zero_grad(
                set_to_none=True
            )

            output = forward_pass(
                data,
                policy,
            )
            output["loss"].backward()
            optimizer.step()

            train_dicts.append(
                detach_dict(output)
            )

        train_summary = compute_dict_mean(
            train_dicts
        )

        train_hist.append(
            {
                k: float(v.item())
                for k, v in train_summary.items()
            }
        )

        print(
            f"\nEpoch {epoch:04d} | "
            f"train={float(train_summary['loss']):.6f} | "
            f"val={val_loss:.6f} | "
            f"best={min_val:.6f}@{best_epoch}"
        )
        print(
            "  train: "
            + " ".join(
                f"{k}={v.item():.4f}"
                for k, v in train_summary.items()
            )
        )
        print(
            "  val:   "
            + " ".join(
                f"{k}={v.item():.4f}"
                for k, v in val_summary.items()
            )
        )

        print(
            "WEB_METRICS "
            + json.dumps(
                {
                    "epoch": epoch + 1,
                    "total_epochs": args.num_epochs,
                    "train": {
                        k: float(v.item())
                        for k, v in train_summary.items()
                    },
                    "val": {
                        k: float(v.item())
                        for k, v in val_summary.items()
                    },
                    "best_loss": min_val,
                    "best_epoch": best_epoch + 1,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        if (
            args.save_every > 0
            and (epoch + 1) % args.save_every == 0
        ):
            torch.save(
                policy.state_dict(),
                ckpt_dir
                / f"policy_epoch_{epoch+1:04d}.ckpt",
            )

        if (
            args.plot_every > 0
            and (
                (epoch + 1) % args.plot_every == 0
                or epoch == args.num_epochs - 1
            )
        ):
            save_curves(
                train_hist,
                val_hist,
                ckpt_dir,
            )

    torch.save(
        policy.state_dict(),
        ckpt_dir / "policy_last.ckpt",
    )

    if best_state is not None:
        torch.save(
            best_state,
            ckpt_dir / "policy_best.ckpt",
        )

    save_curves(
        train_hist,
        val_hist,
        ckpt_dir,
    )

    history = {
        "train": train_hist,
        "val": val_hist,
        "best_epoch": best_epoch,
        "best_val_loss": min_val,
        "args": vars(args),
    }

    with open(
        ckpt_dir / "training_history.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            history,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"\n[OK] Best epoch={best_epoch}, "
        f"best val={min_val:.6f}"
    )
    print(
        f"[OK] Best checkpoint: "
        f"{ckpt_dir / 'policy_best.ckpt'}"
    )


if __name__ == "__main__":
    main()
