### 采集数据指令
```shell
python piper_act_recorder_windows_dual_rgbd_v7.py `
  --dataset-dir "D:\piper_act\Dataset\dataset_stand_bottle_dual_rgbd" `
  --fps 20 `
  --wrist-width 640 `
  --wrist-height 480 `
  --wrist-fps 30 `
  --wrist-depth-min-mm 150 `
  --wrist-depth-max-mm 1500 `
  --external-width 640 `
  --external-height 480 `
  --external-fps 30 `
  --external-depth-min-mm 250 `
  --external-depth-max-mm 3000 `
  --max-camera-sync-delta 0.05
```

### 训练act模型指令
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

### 执行训练模型
```shell
python run_piper_act_windows_smooth_cpv_v7_dual_rgbd_global.py `
  --ckpt_dir "D:\piper_act\checkpoints\piper_dual_rgbd_global_v7" `
  --control `
  --cpv_cv 0.30 `
  --speed_percent 15 `
  --control_hz 20 `
  --cpv_hz 50 `
  --wrist_width 640 `
  --wrist_height 480 `
  --wrist_fps 30 `
  --external_width 640 `
  --external_height 480 `
  --external_fps 30 `
  --joint_alpha 0.20 `
  --gripper_alpha 0.15 `
  --joint_deadband_deg 0 `
  --max_joint_vel_deg_s 15 `
  --max_joint_acc_deg_s2 60 `
  --max_tracking_error_deg 5 `
  --max_joint_step_deg 1.0 `
  --target_timeout 0.30 `
  --max_camera_sync_delta 0.05 `
  --max_seconds 0
```