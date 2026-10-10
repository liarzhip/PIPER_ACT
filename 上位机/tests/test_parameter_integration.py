"""Verify CLI parity without importing SDKs or starting training/motion."""
import argparse
import ast
from html.parser import HTMLParser
from pathlib import Path
import sys
import tempfile
import unittest

from workflow import TRAIN_SCRIPT, RUN_SCRIPT, PARAMETER_SPECS, training_command, execution_command, parameter_values
from web_app import create_app, settings, ROOT, COLLECTION_ROOT


def cli_parser(path):
    module = ast.parse(path.read_text(encoding='utf-8'))
    fn = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'parse_args')
    ret = next(n for n in ast.walk(fn) if isinstance(n, ast.Return))
    ret.value = ret.value.func.value
    namespace = {'argparse': argparse}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(path), 'exec'), namespace)
    return namespace['parse_args']()


class Forms(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.forms = {}
        self.current = None
        self.feed(source)

    def handle_starttag(self, tag, attrs):
        d = dict(attrs)
        if tag == 'form':
            self.current = d.get('id')
            self.forms[self.current] = {}
        if tag in ('input','select') and self.current and d.get('name'):
            self.forms[self.current][d['name']] = d

    def handle_endtag(self, tag):
        if tag == 'form': self.current = None


class ParametersTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        (root/'data').mkdir()
        (root/'models').mkdir()
        (root/'data/episode_000000.hdf5').touch()
        (root/'models/policy_best.ckpt').touch()
        self.task = dict(dataset_dir=str(root/'data'), ckpt_dir=str(root/'models'))

    def tearDown(self):
        self.temp.cleanup()

    def test_screenshot_training_command_and_all_options(self):
        data=dict(num_epochs=100,batch_size=10,chunk_size=20,sample_stride=10,
                  global_history_stride=1,global_hidden_dim=128,global_residual_scale=1,
                  kl_weight=10,hidden_dim=512,dim_feedforward=3200,lr=1e-5,
                  lr_backbone=1e-5,train_ratio=.8,seed=0,num_workers=0)
        cmd=training_command(self.task,data)
        self.assertEqual(Path(cmd[2]).name,'training_entry.py')
        parsed=vars(cli_parser(TRAIN_SCRIPT).parse_args(cmd[3:]))
        for k,v in data.items(): self.assertEqual(parsed[k],v,k)
        for spec in PARAMETER_SPECS['training']:
            if spec['optional']: continue
            self.assertIn('--'+spec['name'],cmd)

    def test_execution_values_booleans_and_control_gate(self):
        data=dict(cpv_cv=.30,speed_percent=15,control_hz=20,cpv_hz=50,joint_alpha=.20,
                  gripper_alpha=.15,joint_deadband_deg=0,max_joint_vel_deg_s=15,
                  max_joint_acc_deg_s2=60,max_tracking_error_deg=5,max_joint_step_deg=1,
                  target_timeout=.30,max_seconds=0,wrist_depth_align=False,configure_cpv_cv=False,temporal_agg=False)
        cmd,stop,preview=execution_command(self.task,data,Path(self.temp.name)/'runtime')
        parsed=vars(cli_parser(RUN_SCRIPT).parse_args(cmd[3:]))
        for k,v in data.items(): self.assertEqual(parsed[k],v,k)
        self.assertFalse(parsed['control'])
        self.assertFalse(parsed['web_authorized'])
        self.assertTrue(parsed['headless'] and parsed['auto_start'])
        self.assertEqual(parsed['web_stop_file'],str(stop))
        with self.assertRaises(ValueError): execution_command(self.task,{'control':True},stop.parent)
        cmd,_,_=execution_command(self.task,{'control':True,'motion_confirmed':True},stop.parent)
        self.assertIn('--web_authorized',cmd)
        with self.assertRaises(ValueError): execution_command(self.task,{'control':'false'},stop.parent)

    def test_form_fields_cover_real_cli_parameters(self):
        forms=Forms((ROOT/'frontend/index.html').read_text(encoding='utf-8')).forms
        for phase,form in [('training','trainingForm'),('execution','executionForm')]:
            for spec in PARAMETER_SPECS[phase]: self.assertIn(spec['name'],forms[form])
        for k in ('wrist_depth_min_mm','wrist_depth_max_mm','wrist_depth_align','dataset_dir','episode_index','can_backend'):
            self.assertIn(k,forms['configForm'])

    def test_invalid_values_and_capture_metadata_settings(self):
        for phase,data in [('training',{'sample_stride':0}),('training',{'batch_size':1.5}),
                           ('training',{'hidden_dim':513}),('training',{'train_ratio':1}),
                           ('execution',{'target_timeout':0}),('execution',{'joint_alpha':1.1}),
                           ('execution',{'max_tracking_error_deg':float('nan')}),
                           ('execution',{'wrist_depth_align':'false'}),('execution',{'gripper_min_m':.1,'gripper_max_m':0})]:
            with self.assertRaises(ValueError,msg=str(data)): parameter_values(phase,data)
        s=settings(dict(wrist_depth_min_mm=200,wrist_depth_max_mm=1800,wrist_depth_align=False,episode_index=12))
        self.assertEqual((s['wrist_depth_min_mm'],s['wrist_depth_max_mm'],s['wrist_depth_align'],s['episode_index']),(200,1800,False,12))
        with self.assertRaises(ValueError): settings({'wrist_depth_min_mm':2000,'wrist_depth_max_mm':1500})
        with self.assertRaises(ValueError): settings({'episode_index':-1})

    def test_task_path_api_persists_without_moving_data(self):
        app=create_app(self.task['dataset_dir'])
        client=app.test_client()
        task=client.get('/api/tasks').json[0]
        destination=str(Path(self.temp.name)/'alternate_data')
        response=client.post('/api/tasks/'+task['id']+'/paths',json={'dataset_dir':destination},headers={'X-Console-Request':'1'})
        self.assertEqual(response.status_code,200,response.json)
        self.assertEqual(app.extensions['recorder'].dataset,Path(destination))
        self.assertTrue((Path(self.task['dataset_dir'])/'episode_000000.hdf5').is_file())
        self.assertFalse(Path(destination).exists())
        app.extensions['recorder'].disconnect()

    def test_stop_bridge_and_original_cli_defaults(self):
        args=cli_parser(RUN_SCRIPT).parse_args(['--ckpt_dir',self.task['ckpt_dir']])
        self.assertFalse(args.headless or args.auto_start or args.web_authorized or args.control)
        # Compilation catches accidental syntax damage without hardware imports.
        source=RUN_SCRIPT.read_text(encoding='utf-8')
        compile(source,str(RUN_SCRIPT),'exec')
        self.assertIn('Path(args.web_stop_file).exists()',source)
        self.assertIn('robot.hold_current()',source)


if __name__=='__main__': unittest.main()
