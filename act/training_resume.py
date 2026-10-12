"""Epoch-boundary recovery for trusted, locally generated ACT checkpoints."""
import json
import math
import os
from pathlib import Path
import pickle
import random
import re
import shutil
import time

import numpy as np
import torch


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else {}


def atomic_json(value, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)


def atomic_torch_save(value, path):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


def select_checkpoint(folder):
    folder = Path(folder)
    if (folder/'training_state.ckpt').is_file():
        return folder/'training_state.ckpt', None
    history = read_json(folder/'training_history.json')
    candidates = []
    for path in folder.glob('policy_epoch_*.ckpt'):
        match = re.fullmatch(r'policy_epoch_(\d+)\.ckpt', path.name)
        if match: candidates.append((int(match[1]), path))
    if (folder/'policy_last.ckpt').is_file() and history.get('train'):
        candidates.append((history.get('completed_epochs',len(history['train'])),folder/'policy_last.ckpt'))
    if candidates:
        epoch, path = max(candidates,key=lambda item:item[0])
        return path, epoch
    if (folder/'policy_best.ckpt').is_file():
        return folder/'policy_best.ckpt', max(0,int(history.get('best_epoch',0)))
    raise FileNotFoundError('没有可恢复的训练断点，请检查模型目录')


def prepare_resume(args):
    folder = Path(args.ckpt_dir).expanduser().resolve()
    if args.additional_epochs < 0:
        raise ValueError('additional_epochs 必须为非负整数')
    if not args.resume:
        if args.additional_epochs:
            raise ValueError('追加轮次需要 --resume')
        if any(folder.glob('*.ckpt')):
            raise ValueError('此目录已有模型，请选择继续训练或使用新的模型目录')
        return None
    path, completed = select_checkpoint(folder)
    payload = torch.load(path,map_location='cpu',weights_only=False)
    full = path.name == 'training_state.ckpt'
    if full and payload.get('format') != 'piper-training-state-v1':
        raise ValueError('完整断点格式无效')
    history = payload['history'] if full else read_json(folder/'training_history.json')
    completed = int(payload['completed_epochs']) if full else completed
    saved_args = payload['args'] if full else history.get('args',{})
    # Structural and data parameters are restored, not silently replaced by UI defaults.
    for key,value in saved_args.items():
        if key not in ('num_epochs','additional_epochs','resume','ckpt_dir','dataset_dir','depth_stats_json') and hasattr(args,key):
            setattr(args,key,value)
    for key,value in read_json(folder/'model_config.json').items():
        if hasattr(args,key): setattr(args,key,value)
    split = read_json(folder/'dataset_split.json')
    stats_path = folder/'dataset_stats.pkl'
    if not stats_path.is_file() or not split:
        raise ValueError('续训需要原目录中的 dataset_stats.pkl 和 dataset_split.json')
    with stats_path.open('rb') as f: old_stats = pickle.load(f)
    for key in ('sample_stride','global_history_stride','global_hidden_dim','global_residual_scale'):
        if key in old_stats: setattr(args,key,old_stats[key])
    total = completed + args.additional_epochs if args.additional_epochs else args.num_epochs
    if total <= completed:
        raise ValueError(f'已完成 {completed} 轮；目标总轮数需大于它，或选择追加轮次')
    args.num_epochs = total
    # Use exactly the previously saved normalization for both cameras.
    depth = dict(cameras=old_stats['camera_depth_stats'],camera_names=args.camera_names)
    depth_path = folder/'resume_depth_stats.json'
    atomic_json(depth,depth_path)
    args.depth_stats_json = str(depth_path)
    offset = int(history.get('history_start_epoch',0))
    usable = max(0,completed-offset)
    train = history.get('train',[])[:usable]
    val = history.get('val',[])[:usable]
    if len(train) != usable or len(val) != usable:
        train=[];val=[];offset=completed
    best_loss = float(history.get('best_val_loss',math.inf))
    best_epoch = int(history.get('best_epoch',-1))
    best_state = payload.get('best_state') if full else None
    best_path = folder/'policy_best.ckpt'
    if best_state is None and best_path.is_file():
        best_state=torch.load(best_path,map_location='cpu',weights_only=False)
    print(f'[RESUME] {path.name}: completed={completed}; target={total}',flush=True)
    if not full:
        print('[WARN] 旧断点仅包含权重：优化器重新初始化；未保存的轮次无法恢复。',flush=True)
    return dict(full=full,payload=payload,path=path,completed=completed,train=train,val=val,
                offset=offset,min_val=best_loss,best_epoch=best_epoch,best_state=best_state,
                old_stats=old_stats,split=split)


def verify_resume_data(resume,stats,info):
    if resume is None: return
    for key in ('train_files','val_files'):
        if info[key] != resume['split'][key]:
            raise ValueError('训练/验证数据划分与原训练不一致，请恢复原数据、seed、train_ratio 和 max_episodes')
    old=resume['old_stats']
    for key in ('qpos_mean','qpos_std','action_mean','action_std'):
        if key not in old or not np.allclose(old[key],stats[key],rtol=1e-6,atol=1e-8):
            raise ValueError(f'原始数据统计已变化：{key}，请勿在原目录续训')


def restore_training(resume,policy,optimizer):
    if resume is None: return
    payload=resume['payload']
    policy.load_state_dict(payload['model'] if resume['full'] else payload,strict=True)
    if resume['full']:
        optimizer.load_state_dict(payload['optimizer'])
        # Optimizer implementations do not all relocate CPU state automatically.
        device=next(policy.parameters()).device
        for state in optimizer.state.values():
            for key,value in state.items():
                if torch.is_tensor(value) and key!='step':state[key]=value.to(device)
        random.setstate(payload['rng_python']);np.random.set_state(payload['rng_numpy'])
        torch.set_rng_state(payload['rng_torch'])
        if torch.cuda.is_available() and payload.get('rng_cuda'):
            torch.cuda.set_rng_state_all(payload['rng_cuda'])
    folder=resume['path'].parent
    backup=folder/'resume_backups'/time.strftime('%Y%m%d-%H%M%S')
    backup.mkdir(parents=True,exist_ok=False)
    for name in ('policy_best.ckpt','policy_last.ckpt','training_history.json','model_config.json','dataset_stats.pkl','dataset_split.json'):
        if (folder/name).is_file():shutil.copy2(folder/name,backup/name)


def save_training_state(folder,args,completed,policy,optimizer,train,val,min_val,best_epoch,best_state,offset):
    folder=Path(folder)
    history=dict(train=train,val=val,best_epoch=best_epoch,best_val_loss=min_val,args=vars(args),
                 completed_epochs=completed,history_start_epoch=offset)
    state=dict(format='piper-training-state-v1',completed_epochs=completed,model=policy.state_dict(),
               optimizer=optimizer.state_dict(),args=vars(args),history=history,best_state=best_state,
               rng_python=random.getstate(),rng_numpy=np.random.get_state(),rng_torch=torch.get_rng_state(),
               rng_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    atomic_torch_save(state,folder/'training_state.ckpt')
    atomic_json(history,folder/'training_history.json')
