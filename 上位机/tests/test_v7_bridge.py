import argparse
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np
from training_entry import prepare_depth_stats
from dual_validation import validate_dual
from workflow import parameter_values, parameter_arguments
from web_app import settings


class DualBridgeTests(unittest.TestCase):
    def test_camera_order_and_dimensions(self):
        self.assertEqual(parameter_values('training',{'camera_names':'wrist external'})['camera_names'],['wrist','external'])
        argv=parameter_arguments('training',{})
        index=argv.index('--camera_names')
        self.assertEqual(argv[index+1:index+3],['wrist','external'])
        with self.assertRaises(ValueError):parameter_values('training',{'camera_names':'external wrist'})
        with self.assertRaises(ValueError):parameter_values('execution',{'external_width':320})
        with self.assertRaises(ValueError):settings({'external_width':320})

    def test_per_camera_training_only_stats_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            for i in range(4):
                with h5py.File(Path(folder)/f'episode_{i:06}.hdf5','w') as f:
                    for cam,offset in [('wrist',300),('external',900)]:
                        f.create_dataset(f'observations/depth/{cam}',data=np.array([[[offset+i*100,offset+i*100+30]]],dtype=np.uint16))
            args=argparse.Namespace(dataset_dir=folder,depth_stats_json=None,camera_names=['wrist','external'],max_episodes=3,train_ratio=.5,seed=0)
            path=prepare_depth_stats(args);result=json.loads(path.read_text())
            self.assertEqual(len(result['files_used'])+len(result['validation_files']),3)
            self.assertFalse(set(result['files_used'])&set(result['validation_files']))
            self.assertNotEqual(result['cameras']['wrist']['depth_mean'],result['cameras']['external']['depth_mean'])
            expected=[]
            for name in result['files_used']:
                with h5py.File(name) as f: expected.extend((f['observations/depth/wrist'][:].astype(float).flatten()-150)/1350)
            self.assertAlmostEqual(result['cameras']['wrist']['depth_mean'],np.mean(expected),places=6)
            self.assertEqual(prepare_depth_stats(args),path)
            self.assertFalse((Path(folder)/'depth_stats_dual.json').exists())

    def test_dual_validation_detects_missing_external_depth(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'episode_000000.hdf5'
            with h5py.File(path,'w') as f:
                f.create_dataset('action',data=np.zeros((2,7)))
                for cam in ('wrist','external'):
                    f.create_dataset(f'observations/images/{cam}',data=np.zeros((2,2,2,3),dtype=np.uint8))
                    f.create_dataset(f'observations/depth/{cam}',data=np.ones((2,2,2),dtype=np.uint16))
                    f.create_dataset(f'timestamps/{cam}_image_age_s',data=[.01,.02])
                f.create_dataset('timestamps/camera_sync_delta_s',data=[.01,.01])
            self.assertEqual(validate_dual(path)[0],'pass')
            with h5py.File(path,'a') as f:del f['observations/depth/external']
            self.assertEqual(validate_dual(path)[0],'fail')

if __name__=='__main__':unittest.main()
