#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import torch
import torch.nn as nn
from torch.nn import functional as F

from detr.main import get_args_parser
from detr.models import build_ACT_model


def _set_submodule(root: nn.Module, dotted_name: str, new_module: nn.Module):
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    leaf = parts[-1]
    if leaf.isdigit():
        parent[int(leaf)] = new_module
    else:
        setattr(parent, leaf, new_module)


def _convert_backbone_rgb_to_rgbd(model: nn.Module, image_channels: int = 4):
    """Patch every ACT camera backbone first conv from RGB(3) to RGB-D(4)."""
    if image_channels == 3:
        return []

    candidates = [
        (name, module)
        for name, module in model.named_modules()
        if "backbone" in name.lower()
        and isinstance(module, nn.Conv2d)
        and module.in_channels == 3
    ]
    if not candidates:
        raise RuntimeError("Could not find a 3-channel Conv2d inside ACT camera backbones.")

    replaced = []
    for name, old in candidates:
        new = nn.Conv2d(
            image_channels,
            old.out_channels,
            old.kernel_size,
            old.stride,
            old.padding,
            old.dilation,
            old.groups,
            old.bias is not None,
            old.padding_mode,
        )
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :3].copy_(old.weight)
            mean_kernel = old.weight.mean(dim=1, keepdim=True)
            for c in range(3, image_channels):
                new.weight[:, c : c + 1].copy_(mean_kernel)
            if old.bias is not None:
                new.bias.copy_(old.bias)

        _set_submodule(model, name, new)
        replaced.append(name)

    return replaced


class GlobalHistoryEncoder(nn.Module):
    """GRU memory over the whole strictly-past episode.

    Each history step is:
        [normalized qpos(7), normalized action(7)] -> 14D
    The final recurrent hidden state summarizes the whole past from episode start.
    """

    def __init__(self, state_dim=7, hidden_dim=128, residual_scale=1.0):
        super().__init__()
        self.state_dim = int(state_dim)
        self.input_dim = self.state_dim * 2
        self.hidden_dim = int(hidden_dim)
        self.residual_scale = float(residual_scale)

        self.gru = nn.GRU(
            self.input_dim,
            self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.state_dim)

        nn.init.normal_(self.out.weight, 0.0, 1e-3)
        nn.init.zeros_(self.out.bias)

    def zero_hidden(self, batch_size, device, dtype=torch.float32):
        return torch.zeros(
            1,
            int(batch_size),
            self.hidden_dim,
            device=device,
            dtype=dtype,
        )

    def encode_sequence(self, history, lengths):
        if history.ndim != 3 or history.shape[-1] != self.input_dim:
            raise RuntimeError(
                f"global_history must be [B,L,{self.input_dim}], "
                f"got {tuple(history.shape)}"
            )

        lengths = lengths.to(
            device=history.device,
            dtype=torch.long,
        ).reshape(-1)

        if lengths.shape[0] != history.shape[0]:
            raise RuntimeError("global_history_lengths batch mismatch")

        safe_lengths = lengths.clamp(min=1, max=history.shape[1])
        packed = nn.utils.rnn.pack_padded_sequence(
            history,
            safe_lengths.detach().cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, h = self.gru(packed)
        valid = (lengths > 0).to(h.dtype).view(1, -1, 1)
        return h * valid

    def step(self, history_step, hidden):
        if history_step.ndim != 2 or history_step.shape[-1] != self.input_dim:
            raise RuntimeError(
                f"history_step must be [B,{self.input_dim}], "
                f"got {tuple(history_step.shape)}"
            )
        _, h = self.gru(history_step.unsqueeze(1), hidden)
        return h

    def residual_from_hidden(self, hidden):
        h = self.norm(hidden[-1])
        return torch.tanh(self.out(h)) * self.residual_scale


def build_piper_act_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser(
        "ACT model defaults",
        parents=[get_args_parser()],
        add_help=False,
    )
    base_args = [
        "--ckpt_dir", "__unused__",
        "--policy_class", "ACT",
        "--task_name", "piper_pick",
        "--seed", "0",
        "--num_epochs", "1",
    ]
    args = parser.parse_args(base_args)

    for key, value in args_override.items():
        setattr(args, key, value)

    model = build_ACT_model(args)

    image_channels = int(args_override.get("image_channels", 4))
    replaced = _convert_backbone_rgb_to_rgbd(
        model,
        image_channels=image_channels,
    )

    print(
        f"[INFO] ACT visual backbone input channels: {image_channels}; "
        f"patched={replaced}"
    )

    model.cuda()

    param_dicts = [
        {
            "params": [
                p
                for name, p in model.named_parameters()
                if "backbone" not in name and p.requires_grad
            ]
        },
        {
            "params": [
                p
                for name, p in model.named_parameters()
                if "backbone" in name and p.requires_grad
            ],
            "lr": args.lr_backbone,
        },
    ]

    optimizer = torch.optim.AdamW(
        param_dicts,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    return model, optimizer


class ACTPolicy(nn.Module):
    """Dual-camera RGB-D ACT + persistent global GRU memory."""

    def __init__(self, args_override):
        super().__init__()

        model, optimizer = build_piper_act_model_and_optimizer(args_override)
        self.model = model

        self.kl_weight = float(args_override["kl_weight"])
        self.state_dim = int(args_override.get("state_dim", 7))
        self.image_channels = int(args_override.get("image_channels", 4))
        self.camera_names = list(args_override.get("camera_names", ["wrist", "external"]))

        self.global_hidden_dim = int(args_override.get("global_hidden_dim", 128))
        self.global_residual_scale = float(
            args_override.get("global_residual_scale", 1.0)
        )
        self.global_history_stride = int(
            args_override.get("global_history_stride", 1)
        )

        self.global_encoder = GlobalHistoryEncoder(
            self.state_dim,
            self.global_hidden_dim,
            self.global_residual_scale,
        ).cuda()

        optimizer.add_param_group(
            {
                "params": self.global_encoder.parameters(),
                "lr": float(args_override["lr"]),
            }
        )
        self.optimizer = optimizer

        depth_means = args_override.get("depth_means", None)
        depth_stds = args_override.get("depth_stds", None)

        if depth_means is None:
            depth_means = [float(args_override.get("depth_mean", 0.5))] * len(self.camera_names)
        if depth_stds is None:
            depth_stds = [float(args_override.get("depth_std", 0.5))] * len(self.camera_names)

        if len(depth_means) != len(self.camera_names):
            raise ValueError("depth_means length must match camera_names")
        if len(depth_stds) != len(self.camera_names):
            raise ValueError("depth_stds length must match camera_names")
        if any(float(x) <= 0 for x in depth_stds):
            raise ValueError("Every camera depth_std must be > 0")

        self.register_buffer(
            "_depth_means",
            torch.tensor(depth_means, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_depth_stds",
            torch.tensor(depth_stds, dtype=torch.float32),
            persistent=False,
        )

        print(f"KL Weight {self.kl_weight}")
        print(f"[INFO] cameras={self.camera_names}")
        print(f"[INFO] RGB-D image channels per camera: {self.image_channels}")
        print(
            f"[INFO] GLOBAL GRU memory: hidden={self.global_hidden_dim}, "
            f"history_stride={self.global_history_stride}, "
            f"residual_scale={self.global_residual_scale:.2f}"
        )
        for idx, cam in enumerate(self.camera_names):
            print(
                f"[INFO] {cam} depth normalize: "
                f"mean={float(self._depth_means[idx]):.8f}, "
                f"std={float(self._depth_stds[idx]):.8f}"
            )

    def init_memory(self, batch_size=1, device=None, dtype=torch.float32):
        if device is None:
            device = next(self.parameters()).device
        return self.global_encoder.zero_hidden(
            batch_size,
            device,
            dtype,
        )

    def advance_memory(
        self,
        normalized_qpos,
        normalized_action,
        memory_state,
    ):
        step = torch.cat(
            [normalized_qpos, normalized_action],
            dim=-1,
        )
        return self.global_encoder.step(
            step,
            memory_state,
        )

    def _memory_for_forward(
        self,
        qpos,
        global_history,
        global_history_lengths,
        memory_state,
    ):
        if memory_state is not None:
            return memory_state

        if global_history is None or global_history_lengths is None:
            raise RuntimeError(
                "Provide global_history+lengths for training "
                "or memory_state for runtime."
            )

        return self.global_encoder.encode_sequence(
            global_history,
            global_history_lengths,
        )

    def _normalize_multi_camera_rgbd(self, image):
        """Normalize [B,K,4,H,W] with per-camera depth statistics."""
        if image.ndim != 5:
            raise RuntimeError(
                f"image must be [B,K,C,H,W], got {tuple(image.shape)}"
            )
        if image.shape[1] != len(self.camera_names):
            raise RuntimeError(
                f"Expected {len(self.camera_names)} cameras {self.camera_names}, "
                f"got image shape={tuple(image.shape)}"
            )
        if image.shape[2] != self.image_channels:
            raise RuntimeError(
                f"Expected {self.image_channels} channels per camera, "
                f"got image shape={tuple(image.shape)}"
            )

        if self.image_channels != 4:
            raise RuntimeError(
                "This V7 policy expects RGB-D (4 channels) for every camera."
            )

        means = []
        stds = []
        for cam_idx in range(len(self.camera_names)):
            means.append(
                [
                    0.485,
                    0.456,
                    0.406,
                    float(self._depth_means[cam_idx]),
                ]
            )
            stds.append(
                [
                    0.229,
                    0.224,
                    0.225,
                    float(self._depth_stds[cam_idx]),
                ]
            )

        mean = torch.tensor(
            means,
            dtype=image.dtype,
            device=image.device,
        ).view(1, len(self.camera_names), 4, 1, 1)

        std = torch.tensor(
            stds,
            dtype=image.dtype,
            device=image.device,
        ).view(1, len(self.camera_names), 4, 1, 1)

        return (image - mean) / std

    def forward(
        self,
        qpos,
        image,
        actions=None,
        is_pad=None,
        global_history=None,
        global_history_lengths=None,
        memory_state=None,
    ):
        env_state = None

        memory = self._memory_for_forward(
            qpos,
            global_history,
            global_history_lengths,
            memory_state,
        )

        qpos_conditioned = (
            qpos
            + self.global_encoder.residual_from_hidden(memory)
        )

        image = self._normalize_multi_camera_rgbd(image)

        if actions is not None:
            actions = actions[:, : self.model.num_queries]
            is_pad = is_pad[:, : self.model.num_queries]

            a_hat, _, (mu, logvar) = self.model(
                qpos_conditioned,
                image,
                env_state,
                actions,
                is_pad,
            )

            total_kld, _, _ = kl_divergence(mu, logvar)

            all_l1 = F.l1_loss(
                actions,
                a_hat,
                reduction="none",
            )

            valid_mask = (
                (~is_pad)
                .unsqueeze(-1)
                .expand_as(all_l1)
            )

            l1 = (
                all_l1 * valid_mask
            ).sum() / valid_mask.sum().clamp_min(1)

            result = {
                "l1": l1,
                "kl": total_kld[0],
            }
            result["loss"] = (
                result["l1"]
                + result["kl"] * self.kl_weight
            )
            return result

        a_hat, _, (_, _) = self.model(
            qpos_conditioned,
            image,
            env_state,
        )
        return a_hat

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0

    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))

    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (
        1
        + logvar
        - mu.pow(2)
        - logvar.exp()
    )

    return (
        klds.sum(1).mean(0, True),
        klds.mean(0),
        klds.mean(1).mean(0, True),
    )
