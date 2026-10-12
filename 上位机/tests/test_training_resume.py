import argparse
import copy
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest

sys.path.insert(0,'D:/piper_act/act')
sys.path.insert(0,'D:/piper_act/上位机')
import torch
import numpy as np
from training_resume import prepare_resume,restore_training,save_training_state,select_checkpoint,verify_resume_data
from workflow import training_command

class ResumeTests(unittest.TestCase):
    def args(self,folder,**kw):
        values=dict(ckpt_dir=str(folder),dataset_dir=str(folder),resume=True,additional_epochs=3,num_epochs=10,
                    camera_names=['wrist','external'],chunk_size=20,sample_stride=10)
        values.update(kw)
        return argparse.Namespace(**values)

    def metadata(self,p):
        stats={k:np.array([1.,2.]) for k in ('qpos_mean','qpos_std','action_mean','action_std')}
        stats.update(camera_depth_stats={},sample_stride=5)
        (p/'dataset_stats.pkl').write_bytes(pickle.dumps(stats))
        (p/'dataset_split.json').write_text(json.dumps(dict(train_files=['a'],val_files=['b'])))
        (p/'model_config.json').write_text(json.dumps(dict(chunk_size=40)))
        return stats

    def test_full_optimizer_continuation_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);stats=self.metadata(p)
            model=torch.nn.Linear(2,1);opt=torch.optim.AdamW(model.parameters(),lr=.01)
            def step(m,o):
                o.zero_grad();m(torch.tensor([[1.,2.]])).square().mean().backward();o.step()
            step(model,opt)
            saved_args=self.args(p,chunk_size=40,sample_stride=5)
            save_training_state(p,saved_args,1,model,opt,[{'loss':1}],[{'loss':2}],2,0,copy.deepcopy(model.state_dict()),0)
            args=self.args(p)
            restored=prepare_resume(args)
            self.assertEqual(args.num_epochs,4);self.assertEqual(args.chunk_size,40);self.assertEqual(args.sample_stride,5)
            verify_resume_data(restored,stats,dict(train_files=['a'],val_files=['b']))
            with self.assertRaises(ValueError):verify_resume_data(restored,stats,dict(train_files=['x'],val_files=['b']))
            new=torch.nn.Linear(2,1);newopt=torch.optim.AdamW(new.parameters(),lr=.5)
            restore_training(restored,new,newopt)
            step(model,opt);step(new,newopt)
            for a,b in zip(model.parameters(),new.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
            self.assertEqual(len(restored['train']),1)

    def test_legacy_and_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);self.metadata(p)
            torch.save({'weight':torch.ones(1)},p/'policy_epoch_0030.ckpt')
            torch.save({'weight':torch.ones(1)},p/'policy_epoch_0020.ckpt')
            self.assertEqual(select_checkpoint(p)[1],30)
            args=self.args(p);state=prepare_resume(args)
            self.assertEqual(args.num_epochs,33);self.assertEqual(state['offset'],30)
            self.assertFalse(state['full']);self.assertEqual(state['train'],[])
            with self.assertRaises(ValueError):prepare_resume(self.args(p,additional_epochs=0,num_epochs=30))
            (p/'episode_000001.hdf5').touch()
            task=dict(dataset_dir=str(p),ckpt_dir=str(p))
            cmd=training_command(task,dict(training_mode='extend',additional_epochs=50))
            self.assertIn('--resume',cmd);self.assertEqual(cmd[-2:],['--additional_epochs','50'])
            with self.assertRaises(ValueError):training_command(task,dict(training_mode='new'))
            with self.assertRaises(ValueError):training_command(task,dict(training_mode='extend',additional_epochs=1.5))

if __name__=='__main__':unittest.main()
