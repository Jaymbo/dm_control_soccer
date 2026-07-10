"""Test / visualize a trained PPO agent on a dm_control environment.

Usage:
    python test_ppo.py --domain cartpole --task balance
    python test_ppo.py --domain cartpole --task balance --no-viewer   # just print rewards
    python test_ppo.py --checkpoint checkpoints/ppo_cartpole_balance.pt

If no --checkpoint is given, the script auto-discovers the best available:
  1. Standard checkpoint: checkpoints/ppo_<domain>_<task>.pt
  2. Best HPO trial via Optuna DB (optuna.db)
  3. Best HPO trial by comparing best_eval in all trial checkpoints
"""
import argparse
import glob
import os
import re
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.ppo import PPO


def get_obs_dim(env):
    spec = env.observation_spec()
    return int(sum(np.prod(v.shape) for v in spec.values()))


def get_act_dim(env):
    spec = env.action_spec()
    return int(np.prod(spec.shape))


def get_act_limit(env):
    spec = env.action_spec()
    return float(np.max(np.abs(spec.maximum)))


def make_env(domain, task):
    if domain in ('cartpole', 'cheetah', 'hopper', 'walker', 'pendulum', 'fish',
                  'humanoid', 'point_mass', 'reacher', 'finger', 'manipulator',
                  'acrobot', 'ball_in_cup', 'dog', 'humanoid_CMU', 'lqr'):
        from dm_control import suite
        return suite.load(domain, task)
    else:
        import environments.suite as suite
        return suite.load(domain, task)


def run_episode(env, agent, deterministic=True, render=False):
    """Run one episode and optionally launch the viewer."""
    time_step = env.reset()
    total_reward = 0.0
    steps = 0

    if render:
        from dm_control import viewer
        def policy(time_step):
            action, _, _ = agent.get_action(time_step.observation, deterministic=deterministic)
            return action
        viewer.launch(env, policy=policy)
        return 0.0, 0

    while not time_step.last():
        action, _, _ = agent.get_action(time_step.observation, deterministic=deterministic)
        action = np.clip(action, env.action_spec().minimum, env.action_spec().maximum)
        time_step = env.step(action)
        total_reward += float(time_step.reward) if time_step.reward is not None else 0.0
        steps += 1
    return total_reward, steps


def _find_best_checkpoint(domain, task):
    """Auto-discover the best PPO checkpoint for a domain/task."""
    save_dir = 'checkpoints'
    base_name = f'ppo_{domain}_{task}'
    standard = os.path.join(save_dir, f'{base_name}.pt')

    if os.path.exists(standard):
        return standard

    # Best trial via Optuna DB
    optuna_db = 'sqlite:///optuna.db'
    if os.path.exists('optuna.db'):
        try:
            import optuna
            storage = optuna.storages.RDBStorage(
                optuna_db,
                engine_kwargs={'connect_args': {'timeout': 10}},
            )
            study = optuna.load_study(study_name='ppo_hpo', storage=storage)
            best_trial = study.best_trial
            best_path = os.path.join(save_dir, f'{base_name}_trial{best_trial.number}.pt')
            steps = best_trial.user_attrs.get('steps', '?')
            if os.path.exists(best_path):
                print(f"Auto-selected best HPO trial {best_trial.number} "
                      f"(final_eval={best_trial.value:.3f}, steps={steps}) from Optuna DB")
                return best_path
        except Exception:
            pass

    # Filesystem scan
    pattern = os.path.join(save_dir, f'{base_name}_trial*.pt')
    trial_files = glob.glob(pattern)
    if not trial_files:
        return standard

    best_eval = -1e9
    best_path = None
    best_steps = '?'
    for f in trial_files:
        try:
            ckpt = torch.load(f, map_location='cpu', weights_only=False)
            eval_val = ckpt.get('final_eval', ckpt.get('best_eval', -1e9))
            if eval_val > best_eval:
                best_eval = eval_val
                best_path = f
                best_steps = ckpt.get('total_steps', '?')
        except Exception:
            continue

    if best_path:
        m = re.search(r'trial(\d+)', os.path.basename(best_path))
        trial_num = m.group(1) if m else '?'
        print(f"Auto-selected trial {trial_num} (final_eval={best_eval:.3f}, "
              f"steps={best_steps}) from {len(trial_files)} checkpoints")
        return best_path

    return standard


def _infer_domain_task(checkpoint_path):
    """Infer domain/task from checkpoint filename: ppo_<domain>_<task>.pt or ppo_<domain>_<task>_trialN.pt."""
    name = os.path.basename(checkpoint_path)
    name = name.replace('.pt', '')
    name = re.sub(r'_trial\d+$', '', name)
    prefix = 'ppo_'
    if name.startswith(prefix):
        name = name[len(prefix):]
    known_domains = ['cartpole_ball', 'one_joint_ball', 'cartpole', 'cheetah', 'hopper',
                    'walker', 'pendulum', 'fish', 'humanoid', 'point_mass', 'reacher',
                    'finger', 'manipulator', 'acrobot', 'ball_in_cup', 'humanoid_CMU', 'lqr']
    for d in known_domains:
        if name.startswith(d + '_'):
            task = name[len(d) + 1:]
            return d, task
    parts = name.rsplit('_', 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default=None)
    parser.add_argument('--task', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to .pt file. Default: checkpoints/ppo_<domain>_<task>.pt')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--deterministic', action='store_true', default=True)
    parser.add_argument('--stochastic', action='store_true',
                        help='Use stochastic actions instead of deterministic')
    parser.add_argument('--no-viewer', action='store_true',
                        help='Do not launch the dm_control viewer; just print rewards')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    if args.checkpoint is None:
        if args.domain is None or args.task is None:
            print("ERROR: Please specify either --checkpoint or both --domain and --task.")
            sys.exit(1)
        args.checkpoint = _find_best_checkpoint(args.domain, args.task)

    # Infer domain/task from checkpoint if not explicitly provided
    if args.domain is None or args.task is None:
        inferred_domain, inferred_task = _infer_domain_task(args.checkpoint)
        if args.domain is None:
            args.domain = inferred_domain
        if args.task is None:
            args.task = inferred_task
        if args.domain is None or args.task is None:
            print("ERROR: Could not infer domain/task. Please specify --domain and --task explicitly.")
            sys.exit(1)

    env = make_env(args.domain, args.task)

    obs_dim = get_obs_dim(env)
    act_dim = get_act_dim(env)
    act_limit = get_act_limit(env)

    agent = PPO(obs_dim, act_dim, act_limit=act_limit, device=args.device)

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: No checkpoint found at {args.checkpoint}")
        print("Train the agent first:  python train.py --algo ppo --domain {} --task {}".format(args.domain, args.task))
        sys.exit(1)

    agent.load(args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    total_steps = ckpt.get('total_steps', '?')
    final_eval = ckpt.get('final_eval', ckpt.get('best_eval', '?'))
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  Steps: {total_steps} | Final eval: {final_eval} | Best eval: {ckpt.get('best_eval', '?')}")

    deterministic = not args.stochastic

    if not args.no_viewer:
        print("Launching dm_control viewer (close window to exit)...")
        run_episode(env, agent, deterministic=deterministic, render=True)
    else:
        rewards = []
        for i in range(args.episodes):
            r, s = run_episode(env, agent, deterministic=deterministic, render=False)
            rewards.append(r)
            print(f"Episode {i+1}: reward={r:.3f}, steps={s}")
        print(f"\nMean reward over {args.episodes} episodes: {np.mean(rewards):.3f} +/- {np.std(rewards):.3f}")


if __name__ == '__main__':
    main()
