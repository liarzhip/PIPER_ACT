"""Prepare training-only per-camera depth statistics before the V7 trainer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = ROOT / 'act/train_piper_act_dual_rgbd_global_v7.py'


def prepare_depth_stats(args):
    if args.depth_stats_json:
        return Path(args.depth_stats_json).expanduser().resolve()
    import h5py
    sys.path.insert(0, str(ROOT / 'act'))
    sys.path.insert(0, str(ROOT / '数据采集'))
    from utils_piper_dual_rgbd_global import discover_episode_files, split_episode_files
    from compute_depth_stats_dual import compute_one_camera
    dataset = Path(args.dataset_dir).expanduser().resolve()
    existing = dataset / 'depth_stats_dual.json'
    if existing.is_file():
        return existing
    files = discover_episode_files(dataset)
    if args.max_episodes is not None:
        files = files[:args.max_episodes]
    train, val = split_episode_files(files, args.train_ratio, args.seed)
    ranges = {}
    for cam in args.camera_names:
        observed = set()
        defaults = (150,1500) if cam == 'wrist' else (250,3000)
        for path in train:
            with h5py.File(path, 'r') as f:
                if f'observations/depth/{cam}' not in f:
                    raise ValueError(f'{path.name} 缺少 {cam} 深度数据；V7 需要 wrist 和 external 双相机数据')
                observed.add(tuple(float(f.attrs.get(f'{cam}_depth_{bound}_mm', default)) for bound,default in zip(('min','max'),defaults)))
        if len(observed) != 1:
            raise ValueError(f'{cam} 训练 Episode 的深度范围不一致')
        ranges[cam] = observed.pop()
        if ranges[cam][0] >= ranges[cam][1]:
            raise ValueError(f'{cam} 深度范围无效')
    identity = dict(cameras=args.camera_names, seed=args.seed, train_ratio=args.train_ratio,
                    ranges=ranges, train=[str(p) for p in train], val=[str(p) for p in val],
                    files=[(str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in files])
    digest = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    cache = dataset / '.depth_stats-cache' / f'dual-{digest}.json'
    if cache.is_file():
        print(f'[PREP] Reusing dual depth statistics: {cache}', flush=True)
        return cache
    print(f'[PREP] Computing dual depth statistics from {len(train)} training episodes; excluding {len(val)} validation episodes.', flush=True)
    cameras = {cam:compute_one_camera(train,cam,*ranges[cam]) for cam in args.camera_names}
    stats = dict(format='dual_rgbd_depth_stats_v1',camera_names=args.camera_names,cameras=cameras,
                 train_ratio=args.train_ratio,seed=args.seed,files_used=[str(p) for p in train],
                 validation_files=[str(p) for p in val],source_signature=identity)
    cache.parent.mkdir(parents=True,exist_ok=True)
    temporary=cache.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(cache)
    return cache


def main():
    original = sys.argv[1:]
    if '--help' not in original and '-h' not in original:
        parser=argparse.ArgumentParser(add_help=False)
        parser.add_argument('--dataset_dir',required=True)
        parser.add_argument('--depth_stats_json')
        parser.add_argument('--camera_names',nargs='+',default=['wrist','external'])
        parser.add_argument('--train_ratio',type=float,default=.8)
        parser.add_argument('--seed',type=int,default=0)
        parser.add_argument('--max_episodes',type=int)
        args,_=parser.parse_known_args(original)
        if not args.depth_stats_json:
            original += ['--depth_stats_json',str(prepare_depth_stats(args))]
    sys.path.insert(0,str(TRAIN_SCRIPT.parent))
    sys.argv=[str(TRAIN_SCRIPT),*original]
    runpy.run_path(str(TRAIN_SCRIPT),run_name='__main__')


if __name__=='__main__':
    main()
