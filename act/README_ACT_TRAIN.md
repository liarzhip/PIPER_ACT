# PiPER-X ACT 训练适配包

适用于当前数据格式：

```text
episode_000000.hdf5
├── /observations/qpos          [T, 7]
├── /observations/images/wrist  [T, H, W, 3]
└── /action                     [T, 7]
```

不同 Episode 的 `T` 可以不同。

## 这版与官方 ACT 的关键差异

官方 `utils.py` 会把完整 Episode 用 `torch.stack` 堆起来，因此不同长度的
Episode 会报 shape 错误；同时官方 loader 还读取 `/observations/qvel`。

本包使用“变长 Episode + action chunk 局部 padding”：

```text
Episode A: T=183
Episode B: T=251
Episode C: T=309
```

都可以直接训练。

假设 `chunk_size=40`，随机采到 Episode A 的 t=170：

```text
真实 action: action[170:183]  -> 13 帧
padding:                         27 帧
最终输出:                      [40, 7]
is_pad: 前 13 个 False，后 27 个 True
```

因此 DataLoader 中每个 batch 的 action shape 始终一致，但无需把每个
Episode 录成相同长度。

归一化统计也使用所有真实帧 `torch.cat`，padding 不参与统计。

## 1. 放置文件

把下面 4 个文件放到官方 ACT 仓库根目录：

```text
act/
├── train_piper_act.py
├── utils_piper.py
├── policy_piper.py
├── patch_act_model_for_piper.py
├── policy.py
├── utils.py
├── detr/
└── ...
```

## 2. 修改官方模型的 14D 写死

在 ACT 根目录执行：

```powershell
python patch_act_model_for_piper.py
```

脚本会自动备份：

```text
detr/models/detr_vae.py.before_piper.bak
```

并把官方模型内部写死的 14 维改成可配置 `state_dim`。

## 3. 确保 DETR 已安装

在 ACT 根目录：

```powershell
cd detr
python -m pip install -e .
cd ..
```

## 4. 先做 5~20 epoch 冒烟测试

Windows PowerShell：

```shell
python train_piper_act_rgbd_global_v4.py `
  --dataset_dir "D:\piper_act\Dataset\desktop_tidy" `
  --ckpt_dir "D:\piper_act\checkpoints\desktop_tidy" `
  --num_epochs 100 `
  --batch_size 10 `
  --chunk_size 20 `
  --sample_stride 10 `
  --global_history_stride 1 `
  --global_hidden_dim 128 `
  --global_residual_scale 1.0 `
  --kl_weight 10 `
  --hidden_dim 512 `
  --dim_feedforward 3200 `
  --lr 1e-5 `
  --lr_backbone 1e-5 `
  --train_ratio 0.8 `
  --seed 0 `
  --num_workers 0
```


确认 CUDA、数据加载、loss 和 checkpoint 都正常以后再正式训练。

## 5. 正式第一版训练

```shell
python train_piper_act_dual_rgbd_global_v7.py `
  --dataset_dir "D:\piper_act\Dataset\dataset_stand_bottle_dual_rgbd" `
  --ckpt_dir "D:\piper_act\checkpoints\piper_dual_rgbd_global_v7" `
  --camera_names wrist external `
  --num_epochs 50 `
  --batch_size 8 `
  --chunk_size 20 `
  --sample_stride 2 `
  --global_history_stride 1 `
  --global_hidden_dim 128 `
  --global_residual_scale 1.0 `
  --kl_weight 10 `
  --hidden_dim 512 `
  --dim_feedforward 3200 `
  --lr 1e-5 `
  --lr_backbone 1e-5 `
  --train_ratio 0.8 `
  --seed 0 `
  --num_workers 0
```

第一版建议 `chunk_size=20`。你的采样率约 20 Hz，因此 20 步约等于未来 1 秒。

## 6. 输出

训练目录会生成：

```text
policy_best.ckpt
policy_last.ckpt
dataset_stats.pkl
dataset_split.json
training_history.json
train_val_loss.png
train_val_l1.png
train_val_kl.png
...
```

其中实机部署至少需要：

```text
policy_best.ckpt
dataset_stats.pkl
```

## Windows 提醒

先用：

```text
--num_workers 0
```

这是默认值。HDF5 + Windows 多进程在第一轮调试时没必要增加复杂度。

## 对齐说明

本采集器在同一个采样 tick 保存：

```text
image[t]
qpos[t]
action[t]
```

所以此 loader 使用：

```text
qpos[t] -> action[t:t+chunk_size]
```

没有沿用官方真实 ALOHA loader 中 `start_ts - 1` 的对齐 hack。

## 常见问题解决
如果遇到
```text
OMP: Error #15: Initializing libomp.dll, but found libiomp5md.dll already initialized.
```
请在 Windows PowerShell 里执行：
```powershell
setx KMP_DUPLICATE_LIB_OK TRUE
```
或者在 Python 训练程序的最上段加上：
```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```