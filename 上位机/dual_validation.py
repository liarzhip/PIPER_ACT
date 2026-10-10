"""Additional validation for dual RGB-D episodes; legacy episodes remain readable."""
import h5py
import numpy as np


def validate_dual(path, max_age=.1, max_sync=.05):
    if not np.isfinite(max_sync) or not .001 <= max_sync <= 5:
        raise ValueError('双相机同步阈值必须在 0.001–5 秒之间')
    lines=[]; failed=False; warning=False
    with h5py.File(path,'r') as f:
        group=f.get('observations/images')
        if group is None or ('external' not in group and 'camera_names' not in f.attrs):
            return 'review','\n[WARN] 单相机旧数据：可回看，但不能直接用于 V7 双相机训练。\n'
        n=len(f['action']) if 'action' in f else 0
        shapes=[]
        for cam in ('wrist','external'):
            for kind in ('images','depth'):
                key=f'observations/{kind}/{cam}'
                ds=f.get(key)
                expected_ndim=4 if kind=='images' else 3
                valid=(ds is not None and ds.ndim==expected_ndim and len(ds)==n and n>0)
                if valid:
                    valid=(ds.dtype==np.dtype('uint8') and ds.shape[-1]==3) if kind=='images' else ds.dtype==np.dtype('uint16')
                    shapes.append(ds.shape[1:3])
                if not valid: failed=True
                lines.append(f'[{"OK" if valid else "FAIL"}] {key}: '+str(ds.shape if ds is not None else 'missing'))
            key=f'timestamps/{cam}_image_age_s'
            if key not in f or len(f[key])!=n:
                failed=True;lines.append(f'[FAIL] Missing or mismatched {key}')
            else:
                ages=f[key][:]
                valid=np.all(np.isfinite(ages)) and np.all(ages>=0)
                if not valid: failed=True
                elif np.any(ages>max_age): warning=True
                lines.append(f'[{"FAIL" if not valid else "WARN" if np.any(ages>max_age) else "OK"}] {cam} frame age')
        if len(set(shapes))>1:
            failed=True;lines.append('[FAIL] Camera RGB/depth dimensions differ')
        key='timestamps/camera_sync_delta_s'
        if key not in f or len(f[key])!=n:
            failed=True;lines.append('[FAIL] Missing or mismatched dual camera sync timestamps')
        else:
            values=f[key][:]
            if not np.all(np.isfinite(values)) or np.any(values<0):
                failed=True;lines.append('[FAIL] Invalid camera sync values')
            elif np.any(values>max_sync):
                warning=True;lines.append('[WARN] Camera sync exceeds threshold')
            else: lines.append('[OK] Dual camera synchronization')
    return ('fail' if failed else 'review' if warning else 'pass'),'\n[V7 双相机检查]\n'+'\n'.join(lines)+'\n'
