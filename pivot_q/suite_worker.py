"""Isolate model dependencies while sharing the LeRobot evaluation loop."""
import json
import os
from pathlib import Path
import signal
import sys
from .cost import measured, record_compute

def main():
    payload = json.loads(sys.argv[1])
    (app, plan) = (payload['experiment'], payload['plan'])
    cfg = payload['model']

    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    with record_compute(plan['output'], app['train' if plan['phase'] == 'train' else 'evaluation']['estimate_flops'], resume=plan['resume'] and plan['phase'] == 'eval'):
        if app['backbone'] == 'openvla_oft':
            if plan['phase'] == 'train':
                from .oft_runtime.training.train_single_gpu import run
                run(cfg)
                return
            from .oft_runtime.integration.policy import OFTPolicy
            from .oft_runtime.training.checkpoint import read
            policy = OFTPolicy(cfg, quantized=plan['method'] != 'original', trainable=bool(plan['adapter']))
            if plan['adapter']:
                read(Path(plan['adapter']), policy, cfg)

            def predict(observation, diagnostic=False):
                import numpy as np
                from .oft_runtime.integration.observations import libero_action
                normalized = policy.predict(oft_observation(observation), observation['task'][0])
                actions = policy.actions(normalized)
                env_actions = np.asarray([libero_action(action) for action in actions], dtype=np.float32)
                return (normalized.detach().float().cpu().numpy().reshape(8, 7), env_actions) if diagnostic else env_actions
        else:
            from .groot_runtime import train as core
            from .groot_runtime import selector, full_distill, train_single_gpu
            if app['quantizer'] == 'holoq':
                from .groot_runtime.holoq import install_hooks
                install_hooks(core)
            config = core.TrainConfig(**cfg)
            if plan['phase'] == 'train':
                core.build_pivot_q_targets = selector.build_targets
                train_single_gpu.install(core)
                core.train(config, full_distill.select_states if plan['method'] == 'full_distill' else selector.select_states)
                return
            policy = core._load_policy(config, quantized=plan['method'] != 'original', repo_root=Path(os.environ['PIVOT_Q_QUANTVLA_ROOT']))
            if plan['adapter']:
                policy.model = core.PeftModel.from_pretrained(policy.model, plan['adapter'], is_trainable=False)
                policy.model.eval()

            def predict(observation, diagnostic=False):
                import numpy as np
                import torch
                state = np.asarray(observation['observation.state'])[0]
                inputs = {'video.image': image(observation['observation.images.image'])[None], 'video.wrist_image': image(observation['observation.images.image2'])[None], 'annotation.human.action.task_description': [observation['task'][0]]}
                for (index, key) in enumerate(('x', 'y', 'z', 'roll', 'pitch', 'yaw')):
                    inputs['state.' + key] = np.asarray([[state[index]]])
                inputs['state.gripper'] = state[6:8][None]
                normalized = core._normalize_observation(policy, inputs)
                base = core._base_model(policy)
                (backbone, action_input) = core._prepare_backbone(base, normalized)
                noise = torch.randn(1, base.action_head.config.action_horizon, base.action_head.config.action_dim)
                actions = core._sample_from_cached(base, backbone, action_input, noise, with_grad=False)
                env_actions = np.asarray([core._normalized_to_libero_action(policy, actions[:, i:i + 1]) for i in range(actions.shape[1])])
                return (actions[0].detach().float().cpu().numpy(), env_actions) if diagnostic else env_actions
        serve(predict, plan['port'])

def image(value):
    import numpy as np
    return np.clip(np.asarray(value)[0].transpose(1, 2, 0) * 255, 0, 255).round().astype(np.uint8)

def oft_observation(observation):
    return dict(full_image=image(observation['observation.images.image']), wrist_image=image(observation['observation.images.image2']), state=observation['observation.state'][0])

def serve(predict, port):
    import pickle
    import zmq
    import torch
    import numpy as np
    import random
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f'tcp://127.0.0.1:{port}')
    try:
        while True:
            request = pickle.loads(socket.recv())
            try:
                (endpoint, data) = (request['endpoint'], request.get('data', {}))
                if endpoint == 'predict':
                    result = measured('policy_forward', predict, data['observation'])
                elif endpoint == 'diagnostic_predict':
                    torch.manual_seed(data['seed'])
                    result = predict(data['observation'], diagnostic=True)
                elif endpoint == 'reset_policy':
                    random.seed(data['seed'])
                    np.random.seed(data['seed'])
                    torch.manual_seed(data['seed'])
                    result = True
                elif endpoint == 'ping':
                    result = dict(status='ok')
                else:
                    raise ValueError(endpoint)
                reply = {'result': result}
            except Exception as error:
                reply = {'error': repr(error)}
            socket.send(pickle.dumps(reply))
    finally:
        socket.close(linger=0)
        context.term()
if __name__ == '__main__':
    main()
