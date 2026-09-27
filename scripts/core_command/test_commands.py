import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import launch
import aggregate
import importlib.util

def load_source(name, relative):
    spec = importlib.util.spec_from_file_location(name, launch.ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class CommandTests(unittest.TestCase):
    def test_resume_rejects_changed_eval_config(self):
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'config.yaml'
            data = launch.read_config(launch.ROOT/'config/eval.yaml')
            source.write_text(yaml.safe_dump(data))
            argv = ['launch','groot_quantvla','quant_original','eval','--suite','libero_goal',
                    '--results-root',str(root/'result'),'--config',str(source)]
            with patch.object(launch,'run_worker'), contextlib.redirect_stdout(io.StringIO()):
                with patch.object(sys,'argv',argv):
                    launch.main()
                with patch.object(sys,'argv',argv+['--resume','--gpu','3','--port','9911']):
                    launch.main()
                data['models']['libero_goal'] = '/different/checkpoint'
                source.write_text(yaml.safe_dump(data))
                with patch.object(sys,'argv',argv+['--resume']), self.assertRaisesRegex(ValueError,'protocol'):
                    launch.main()

    def test_ablation_paths_match_training_yaml(self):
        import yaml
        for name, directory in launch.ABLATION_DIRS.items():
            cfg = yaml.safe_load((launch.ROOT/f'config/{name}.yaml').read_text())
            self.assertEqual(Path(cfg['paths']['results_dir']).parent.name, directory)
            out = self.plan('groot_quantvla','pivot_q','eval','--suite','libero_goal',
                            '--ablation',name,'--adapter-path','/example/adapter')
            self.assertIn(f'ablations/{directory}/eval/',out)

    def test_training_protocol_guards(self):
        from env.protocol import check_training_output, write_training_protocol
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = dict(resume=True, seed=0, lambda_anchor=0.1, env_ports=[1,2,3,4])
            saved = check_training_output(root,cfg)
            write_training_protocol(root,saved)
            check_training_output(root,{**cfg,'env_ports':[5,6,7,8]})
            for changed in ({'resume':False},{'lambda_anchor':0.2},{'seed':1}):
                with self.assertRaises(ValueError):
                    check_training_output(root,{**cfg,**changed})
            checkpoint = root/'checkpoints/step-000025'
            checkpoint.mkdir(parents=True)
            marker = checkpoint/'complete.json'
            marker.write_text(json.dumps({'protocol':saved}))
            check_training_output(root,cfg)
            marker.write_text('{}')
            with self.assertRaisesRegex(ValueError,'Legacy checkpoint'):
                check_training_output(root,cfg)
            (root/'training_protocol.json').unlink()
            with self.assertRaisesRegex(ValueError,'Legacy training'):
                check_training_output(root,cfg)

    def test_documented_eval_to_seven_category_report(self):
        module = load_source('seven_categories', 'scripts/aggregate_seven_categories.py')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {'initial_state_ids':[0], 'task_ids_by_suite':
                        {suite:list(range(140)) for suite in launch.SUITES}}
            # Fake only the GPU worker; exercise the real launcher/receipt and report.
            def worker(command, **kwargs):
                directory = Path(command[command.index('--output-dir') + 1])
                suite = command[command.index('--suite') + 1]
                metrics = directory / 'metrics'
                metrics.mkdir(parents=True)
                rows = [{'task_id':i, 'initial_state_id':0, 'success':True,
                         'suite':suite, 'category':module.CATEGORIES[i // 20]}
                        for i in range(140)]
                (metrics/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                (metrics/'summary.json').write_text(json.dumps({'episodes':140, 'errors':0,
                    'categories':{c:{'episodes':20,'successes':20} for c in module.CATEGORIES}}))
            for ablation in (None, 'random_sparse'):
                method = 'pivot_q' if ablation is None else 'ablations/random_sparse'
                result = root/'quantvla/groot_n1_5'/method
                argv = ['launch','groot_quantvla','pivot_q','eval','--suite','all',
                        '--seed','2027','--adapter-path',str(root/'adapter/{suite}'),
                        '--results-root',str(result)]
                if ablation:
                    argv += ['--ablation',ablation]
                with patch.object(sys,'argv',argv), patch.object(launch,'run_worker',worker), \
                        contextlib.redirect_stdout(io.StringIO()):
                    launch.main()
            rows, audit = module.collect(root, manifest)
            self.assertEqual(len(rows),2)
            self.assertTrue(all(r['Status']=='complete' and r['Avg.']==100 for r in rows))
            self.assertTrue(all(not r['issues'] for r in audit))

    def test_documented_eval_commands_parse(self):
        import re, shlex
        count = 0
        for path in (launch.ROOT/'docs').glob('commands-*.md'):
            for block in re.findall(r'```bash\n(.*?)```', path.read_text(), re.S):
                text = block.replace('\\\n',' ')
                for line in text.splitlines():
                    prefix = 'python scripts/core_command/launch.py '
                    if line.startswith(prefix):
                        self.plan(*shlex.split(line[len(prefix):]))
                        count += 1
        self.assertEqual(count, 96)

    def test_seven_category_identity(self):
        module = load_source('seven_categories', 'scripts/aggregate_seven_categories.py')
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            receipt = directory / 'core_command.json'
            data = dict(family='groot_quantvla', setting='quant_original', seed=2026,
                        suite='libero_goal', benchmark='libero-plus')
            manifest = {'task_ids_by_suite': {'libero_goal': [1, 2]}}
            rows = [{'task_id': i, 'initial_state_id': 0} for i in (1, 2)]
            def validate():
                module.validate_identity(directory, 'quantvla', 'groot_n1_5',
                                         'quantized', 2026, 'libero_goal', rows, manifest)
            with self.assertRaises(ValueError):
                validate()
            receipt.write_text(json.dumps(data))
            validate()
            for field, bad in [('seed',2027), ('family','groot_holoqvla'),
                               ('setting','full_distill'), ('suite','libero_10')]:
                receipt.write_text(json.dumps({**data, field: bad}))
                with self.assertRaises(ValueError):
                    validate()
            receipt.write_text(json.dumps(data))
            rows[1]['task_id'] = 3
            with self.assertRaises(ValueError):
                validate()
            rows[1]['task_id'] = 2
            (directory / 'run.json').write_text('{"eval_seed": 2028}')
            with self.assertRaises(ValueError):
                validate()

    def test_single_initial_restart(self):
        module = load_source('single_training', 'pi05_quantvla/pivot_q/train_single.py')
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            values = dict(resume=True, seed=0, learning_rate=5e-5)
            with self.assertRaises(RuntimeError):
                module.validate_initial_restart(out, values)
            (out / 'run.json').write_text(json.dumps({'format':'pivot_q_single_v1', 'config':values}))
            module.validate_initial_restart(out, values)
            for update in ({'resume':False}, {'seed':1}, {'learning_rate':1e-4}):
                with self.assertRaises(RuntimeError):
                    module.validate_initial_restart(out, {**values, **update})
            (out / 'final_adapter').symlink_to('missing')
            with self.assertRaises(RuntimeError):
                module.validate_initial_restart(out, values)

    def plan(self, *args):
        stream = io.StringIO()
        with patch.object(sys, "argv", ["launch", *args, "--dry-run"]), contextlib.redirect_stdout(stream):
            launch.main()
        return stream.getvalue()

    def test_seed_reaches_legacy_eval_config(self):
        out = self.plan("groot_quantvla", "fp_original", "eval", "--seed", "2027", "--suite", "libero_goal")
        self.assertIn('"policy_seed": 2027', out)
        self.assertIn("seed-2027/libero_goal", out)
        self.assertIn("CUDA_VISIBLE_DEVICES=0", out)

    def test_seed_reaches_oft_config(self):
        out = self.plan("openvla_oft_qvla", "pivot_q", "eval", "--suite", "libero_goal",
                        "--seed", "2028", "--adapter-path", "/tmp/a.pt")
        self.assertIn("evaluation.seed=2028", out)
        self.assertIn("seed-2028/libero_goal", out)
        self.assertIn("paths.adapter=/tmp/a.pt", out)

    def test_pi05_joint_training_seed(self):
        out = self.plan("pi05_holoqvla", "pivot_q", "train", "--seed", "17")
        self.assertIn('"seed": 17', out)
        self.assertIn("--gpus 0 1 2 3", out)

    def test_complete_multi_seed_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = "libero_spatial"
            manifest = root/"manifest.json"
            manifest.write_text(json.dumps({"initial_state_ids":[0], "task_ids_by_suite":{suite:[1,2]}}))
            for seed, successes in [(2026,[True,False]),(2027,[True,True])]:
                directory = launch.eval_directory(root, "groot_quantvla", "fp_original", seed, suite)
                (directory/"metrics").mkdir(parents=True)
                (directory/"core_command.json").write_text(json.dumps({
                    "family":"groot_quantvla", "setting":"fp_original", "seed":seed, "suite":suite}))
                (directory/"metrics/episodes.jsonl").write_text("".join(json.dumps({
                    "task_index":i,"success":ok,"error":None})+"\\n" for i,ok in zip([1,2],successes)).replace("\\n","\n"))
            argv = ["aggregate","groot_quantvla","fp_original","--seeds","2026","2027",
                    "--suite",suite,"--results-root",str(root),"--manifest",str(manifest)]
            with patch.object(sys,"argv",argv), contextlib.redirect_stdout(io.StringIO()):
                aggregate.main()
            result=json.loads((root/"aggregate/seeds-2026-2027"/suite/"summary.json").read_text())
            self.assertEqual(result["across_seeds"]["overall"]["mean_pct"],75.)
            self.assertAlmostEqual(result["across_seeds"]["overall"]["std_pct"],35.3553390593)
            path=launch.eval_directory(root, "groot_quantvla", "fp_original", 2027, suite)/"metrics/episodes.jsonl"
            path.write_text(json.dumps({"task_index":1,"success":True})+"\n")
            with self.assertRaises(ValueError), patch.object(sys,"argv",argv):
                aggregate.main()

    def test_legacy_seed_uses_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            suite="libero_spatial"
            directory=base/"eval/seed-000"/suite
            (directory/"metrics").mkdir(parents=True)
            (directory/"metrics/episodes.jsonl").write_text('{"task_index":1,"success":true}\\n')
            (directory/"run.json").write_text(json.dumps({"eval_seed":2026,"train_seed":0}))
            self.assertEqual(aggregate.locate(base,"groot_quantvla","pivot_q",2026,suite),directory)
            with self.assertRaises(ValueError):
                aggregate.locate(base,"groot_quantvla","pivot_q",0,suite)
            (directory/"run.json").unlink()
            with self.assertRaises(ValueError):
                aggregate.locate(base,"groot_quantvla","pivot_q",2026,suite)

    def test_duplicate_attempt_and_error_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp); (directory/"eval").mkdir()
            path=directory/"eval/episodes.jsonl"
            path.write_text(json.dumps({"task_id":1,"initial_state_id":0,"success":False})+"\n"+
                            json.dumps({"task_id":1,"initial_state_id":0,"success":True})+"\n")
            with self.assertRaisesRegex(ValueError, "duplicate episode"):
                aggregate.summarize(directory,"openvla_oft_qvla","libero_goal",[1])
            path.write_text(json.dumps({"task_id":1,"success":False,"error":"model error"})+"\n")
            with self.assertRaises(ValueError):
                aggregate.summarize(directory,"openvla_oft_qvla","libero_goal",[1])

    def test_oft_resume_reaches_evaluator(self):
        out = self.plan("openvla_oft_qvla", "quant_original", "eval", "--suite", "libero_goal", "--resume")
        self.assertIn("evaluation.evaluate", out)
        self.assertNotIn("--resume", out)  # the evaluator resumes its prefix automatically

if __name__=="__main__":
    unittest.main()
