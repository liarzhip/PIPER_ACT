#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Patch the official tonyzhaozh/act repository for PiPER-X state_dim=7.

Run from the ACT repository root:
    python patch_act_model_for_piper.py

The script:
- backs up detr/models/detr_vae.py
- replaces ACT hardcoded 14D robot/action projections with `state_dim`
- makes build()/build_cnnmlp() read args.state_dim (default 14)

It is intentionally narrow so the rest of the official repository remains
untouched.
"""

from pathlib import Path
import shutil
import sys


def replace_once_or_fail(text, old, new, label):
    count = text.count(old)

    if count == 0:
        raise RuntimeError(
            f"Could not find expected upstream code for: {label}\n"
            f"Expected text:\n{old}"
        )

    return text.replace(old, new)


def main():
    repo_root = Path.cwd()
    target = (
        repo_root
        / "detr"
        / "models"
        / "detr_vae.py"
    )

    if not target.exists():
        raise FileNotFoundError(
            "Run this script from the ACT repository root.\n"
            f"Missing: {target}"
        )

    backup = target.with_suffix(
        target.suffix + ".before_piper.bak"
    )

    if not backup.exists():
        shutil.copy2(
            target,
            backup,
        )
        print(f"[OK] Backup: {backup}")
    else:
        print(
            f"[INFO] Backup already exists: {backup}"
        )

    text = target.read_text(
        encoding="utf-8"
    )

    # ACT / DETRVAE path.
    text = replace_once_or_fail(
        text,
        "self.input_proj_robot_state = nn.Linear(14, hidden_dim)",
        "self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)",
        "robot-state projection",
    )

    text = replace_once_or_fail(
        text,
        "self.encoder_action_proj = nn.Linear(14, hidden_dim)",
        "self.encoder_action_proj = nn.Linear(state_dim, hidden_dim)",
        "CVAE action projection",
    )

    text = replace_once_or_fail(
        text,
        "self.encoder_joint_proj = nn.Linear(14, hidden_dim)",
        "self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)",
        "CVAE joint projection",
    )

    # Upstream has two build functions with this hardcode.
    text = text.replace(
        "state_dim = 14 # TODO hardcode",
        'state_dim = getattr(args, "state_dim", 14)',
    )
    text = text.replace(
        "state_dim = 14  # TODO hardcode",
        'state_dim = getattr(args, "state_dim", 14)',
    )

    # CNNMLP compatibility too, although the PiPER trainer uses ACT.
    text = text.replace(
        "mlp_in_dim = 768 * len(backbones) + 14",
        "mlp_in_dim = 768 * len(backbones) + state_dim",
    )
    text = text.replace(
        "output_dim=14",
        "output_dim=state_dim",
    )

    target.write_text(
        text,
        encoding="utf-8",
    )

    # Basic post-patch checks.
    checks = [
        "nn.Linear(state_dim, hidden_dim)",
        'getattr(args, "state_dim", 14)',
    ]

    patched = target.read_text(
        encoding="utf-8"
    )

    for item in checks:
        if item not in patched:
            raise RuntimeError(
                f"Patch verification failed: {item}"
            )

    print(
        "[OK] PiPER state_dim patch applied to "
        f"{target}"
    )
    print(
        "[INFO] Original file is preserved at "
        f"{backup}"
    )
    print(
        "[NEXT] Copy utils_piper.py, policy_piper.py and "
        "train_piper_act.py into the ACT repository root."
    )


if __name__ == "__main__":
    main()
