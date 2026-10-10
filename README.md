### 采集数据指令
```shell
cd ./数据采集
python piper_act_recorder_windows.py `
>>   --dataset-dir "D:\piper_act\dataset_stand_bottle" `
>>   --can-channel 0 `
>>   --fps 20 `
>>   --camera-width 640 `
>>   --camera-height 480 `
>>   --camera-fps 30
```

### 训练act模型指令
```shell
python train_piper_act.py `
>>   --dataset_dir "D:\piper_act\dataset_stand_bottle" `
>>   --ckpt_dir "D:\piper_act\checkpoints\piper_stand_bottle" `
>>   --num_epochs 2000 `
>>   --batch_size 8 `
>>   --chunk_size 40 `
>>   --kl_weight 10 `
>>   --hidden_dim 512 `
>>   --dim_feedforward 3200 `
>>   --lr 1e-5
```

### 执行训练模型
```shell
python run_piper_act_windows_complete.py `
>>   --ckpt_dir "D:\piper_act\checkpoints\piper_bottle_in_holder" `
>>   --control `                                                   
>>   --speed_percent 15 `
>>   --max_joint_step_deg 2.0 `
>>   --max_seconds 0 


python run_piper_act_windows_smooth_move_j.py `
  --ckpt_dir "D:\piper_act\checkpoints\piper_pick" `
  --control `
  --speed_percent 15 `
  --max_joint_step_deg 1.0 `
  --joint_alpha 0.20 `
  --gripper_alpha 0.15 `
  --joint_deadband_deg 0.08 `
  --gripper_deadband_mm 0.5 `
  --max_seconds 0
```