"""Benchmark schedules and LeRobot's official LIBERO evaluation entry."""
import json
import os
from pathlib import Path
import sys
from . import discrepancy

def episode_specs(cfg, benchmark, suite):
    settings = cfg['evaluation'][benchmark]
    if benchmark == 'libero':
        ids = settings['task_ids'] if settings['task_ids'] is not None else range(10)
        initial = range(settings['initial_state_start'], settings['initial_state_start'] + settings['episodes_per_task'])
        return [dict(suite=suite, task_id=int(task), initial_state_id=int(init), category='clean') for task in ids for init in initial]
    manifest = json.loads(Path(cfg['benchmark']['task_file']).read_text())
    metadata = {int(row['task_id']): row for row in manifest.get('task_metadata_by_suite', {}).get(suite, [])}
    return [dict(metadata.get(int(task), {}), suite=suite, task_id=int(task), initial_state_id=settings['initial_state_id']) for task in manifest['task_ids_by_suite'][suite]]

def summarize(rows, expected):

    def stats(group):
        n = len(group)
        return dict(episodes=n, successes=sum((bool(r['success']) for r in group)), errors=sum((bool(r.get('error')) for r in group)), success_rate=sum((bool(r['success']) for r in group)) / n if n else None)
    suites = {s: stats([r for r in rows if r['suite'] == s]) for s in expected}
    categories = {c: stats([r for r in rows if r['category'] == c]) for c in sorted({r['category'] for r in rows if r['category'] != 'clean'})}
    tasks = {f'{s}/{t}': stats([r for r in rows if r['suite'] == s and r['task_id'] == t]) for (s, t) in sorted({(r['suite'], r['task_id']) for r in rows})}
    return dict(complete=all((suites[s]['episodes'] == expected[s] for s in expected)) and (not any((r.get('error') for r in rows))), suites=suites, categories=categories, tasks=tasks, overall=stats(rows))

def setup_environment(cfg, benchmark):
    root = Path(__file__).resolve().parents[1]
    source = cfg['paths']['libero' if benchmark == 'libero' else 'libero_plus']
    os.environ['LIBERO_CONFIG_PATH'] = cfg['paths']['clean_config' if benchmark == 'libero' else 'plus_config']
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    sys.path.insert(0, source)
    sys.path.insert(0, str(root / 'third_party/lerobot/src'))

def evaluate_policy(policy, preprocessor, postprocessor, cfg, out, seed, benchmark, resume, suites=None, flat=False):
    setup_environment(cfg, benchmark)
    import warnings
    from unittest.mock import patch
    from tqdm.auto import tqdm
    from lerobot.envs import close_envs, make_env, make_env_pre_post_processors
    from lerobot.envs.configs import LiberoEnv, LiberoPlusEnv
    from lerobot.scripts.lerobot_eval import eval_policy
    from lerobot.utils.random_utils import set_seed
    from .pi05_state import write_status
    suites = suites or cfg['benchmark']['suites']
    settings = cfg['evaluation']
    schedules = {s: episode_specs(cfg, benchmark, s) for s in suites}
    expected = {s: len(rows) for (s, rows) in schedules.items()}
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    env_class = LiberoEnv if benchmark == 'libero' else LiberoPlusEnv
    for (suite, specs) in schedules.items():
        path = (out if flat else out / suite) / 'metrics/episodes.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        prior = []
        if path.is_file():
            with path.open('r+b') as stream:
                while (line := stream.readline()):
                    if not line.endswith(b'\n'):
                        stream.truncate(stream.tell() - len(line))
                        break
                    prior.append(json.loads(line))
        if prior and (not resume):
            raise FileExistsError(path)
        done = {(r['task_id'], r['initial_state_id']): r for r in prior}
        requested = {(r['task_id'], r['initial_state_id']) for r in specs}
        if len(done) != len(prior) or not set(done) <= requested or any((r['seed'] != seed for r in prior)):
            raise ValueError('Existing episodes differ from this task or seed schedule')
        rows.extend(prior)
        with tqdm(total=len(specs), initial=len(prior), desc=f'{suite} seed={seed}', unit='episode', dynamic_ncols=True) as bar:
            for task_id in dict.fromkeys((spec['task_id'] for spec in specs)):
                pending = [r for r in specs if r['task_id'] == task_id and (task_id, r['initial_state_id']) not in done]
                if not pending:
                    continue
                env_cfg = env_class(task=suite, task_ids=[task_id], control_mode=settings['control_mode'], hard_reset=settings['hard_reset'], episode_length=settings['max_steps'][suite], observation_height=256, observation_width=256)
                (env_pre, env_post) = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=policy.config)
                envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
                env = envs[suite][task_id]
                try:
                    env.set_attr('num_steps_wait', settings['settle_steps'])
                    for spec in pending:
                        init = spec['initial_state_id']
                        suite_index = cfg['benchmark']['suites'].index(suite)
                        policy_seed = (seed * 1000003 + suite_index * 100003 + task_id * 1009 + init) % 2 ** 32
                        set_seed(policy_seed)
                        if hasattr(policy, 'episode_seed'):
                            policy.episode_seed = policy_seed
                        env.set_attr('init_state_id', init)
                        for processor in (preprocessor, postprocessor, env_pre, env_post):
                            processor.reset()
                        with discrepancy.capture(policy, dict(spec, seed=seed), policy_seed), warnings.catch_warnings(), patch('lerobot.scripts.lerobot_eval.trange', lambda *a, **kw: __import__('tqdm').trange(*a, **kw | {'disable': True})):
                            warnings.simplefilter('ignore', DeprecationWarning)
                            info = eval_policy(env=env, policy=policy, env_preprocessor=env_pre, env_postprocessor=env_post, preprocessor=preprocessor, postprocessor=postprocessor, n_episodes=1, start_seed=seed + init, max_episodes_rendered=0)
                        result = info['per_episode'][0]
                        row = dict(spec, seed=seed, policy_seed=policy_seed, env_seed=seed + init, success=bool(result['success']), error=None, sum_reward=result['sum_reward'], duration_seconds=info['aggregated']['eval_s'])
                        with path.open('a') as stream:
                            stream.write(json.dumps(row) + '\n')
                            stream.flush()
                        rows.append(row)
                        write_status(out / 'summary.json', summarize(rows, expected))
                        bar.update(1)
                finally:
                    close_envs(envs)
    write_status(out / 'summary.json', summarize(rows, expected))

def evaluate(cfg, run, out, seed, adapter, resume, benchmark):
    setup_environment(cfg, benchmark)
    from types import SimpleNamespace
    from .cost import measured
    loader = __import__('pivot_q.holoq_policy' if run['backend'] == 'holoq' else 'pivot_q.quantvla_policy', fromlist=['load_bundle']).load_bundle
    bundle = loader(SimpleNamespace(**run), quantized=run['method_name'] != 'original', adapter=adapter, recovery=run['method_name'] in {'pivot_q', 'full_distill'}, trainable=False)
    bundle.policy.config.n_action_steps = cfg['evaluation']['action_steps']
    predict = bundle.policy.predict_action_chunk
    bundle.policy.predict_action_chunk = lambda *a, **kw: measured('policy_forward', predict, *a, **kw)
    if discrepancy.cache() is not None and run['method_name'] != 'original':
        for suite in cfg['benchmark']['suites']:
            discrepancy.replay(bundle.policy, suite, run['method_name'], bundle.policy.config.device)
        return
    evaluate_policy(bundle.policy, bundle.preprocessor, bundle.postprocessor, cfg, out, seed, benchmark, resume)

def evaluate_remote(payload):
    (cfg, plan) = (payload['experiment'], payload['plan'])
    setup_environment(cfg, payload['benchmark'])
    import torch
    from collections import deque
    from lerobot.policies.pretrained import PreTrainedPolicy
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from .pi05_state import EnvClient
    client = EnvClient(plan['port'])
    client.wait()

    class RemotePolicy(PreTrainedPolicy):
        config_class = PI05Config
        name = 'pivot_q_remote'

        def __init__(self):
            super().__init__(PI05Config(device='cpu'))
            self.queue = deque()
            self.episode_seed = plan['seed']

        def get_optim_params(self):
            return {}

        def reset(self):
            self.queue.clear()
            client.call('reset_policy', seed=self.episode_seed)

        def forward(self, batch):
            raise RuntimeError('Evaluation adapter has no training forward')

        def predict_action_chunk(self, batch, **kwargs):
            observation = {key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value for (key, value) in batch.items()}
            return torch.as_tensor(client.call('predict', observation=observation), dtype=torch.float32).unsqueeze(0)

        def diagnostic_prediction(self, batch, seed):
            observation = {key: value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            actions, env_actions = client.call('diagnostic_predict', observation=observation, seed=seed)
            return tuple(torch.as_tensor(value).unsqueeze(0) for value in (actions, env_actions))

        def select_action(self, batch, **kwargs):
            if not self.queue:
                chunk = self.predict_action_chunk(batch)[:, :cfg['evaluation']['action_steps']]
                self.queue.extend(chunk.transpose(0, 1))
            return self.queue.popleft()

    class IdentityProcessor:

        def reset(self):
            pass

        def __call__(self, value):
            return value
    policy = RemotePolicy()
    try:
        if discrepancy.cache() is not None and plan['method'] != 'original':
            discrepancy.replay(policy, plan['suite'], plan['method'], 'cpu')
            return
        evaluate_policy(policy, IdentityProcessor(), IdentityProcessor(), cfg, Path(plan['output']), plan['seed'], payload['benchmark'], plan['resume'], suites=[plan['suite']], flat=True)
    finally:
        client.close()
if __name__ == '__main__':
    evaluate_remote(json.loads(sys.argv[1]))
