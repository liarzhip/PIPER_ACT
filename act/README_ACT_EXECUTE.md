# ACT 执行说明
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