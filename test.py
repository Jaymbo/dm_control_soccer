"""Test / visualize a trained MPO agent on a dm_control environment.

Usage:
    python test.py --domain cartpole --task balance
    python test.py --domain cartpole --task balance --no-viewer   # just print rewards
    python test.py --domain cartpole_ball --task kick             # custom env
"""
import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.mpo import MPO


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
        # viewer.launch calls the policy on every step including the first.
        # On FIRST step, observation is available so we can compute an action.
        def policy(time_step):
            return agent.get_action(time_step.observation, deterministic=deterministic)
        viewer.launch(env, policy=policy)
        return 0.0, 0

    while not time_step.last():
        action = agent.get_action(time_step.observation, deterministic=deterministic)
        action = np.clip(action, env.action_spec().minimum, env.action_spec().maximum)
        time_step = env.step(action)
        total_reward += float(time_step.reward) if time_step.reward is not None else 0.0
        steps += 1
    return total_reward, steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='cartpole')
    parser.add_argument('--task', type=str, default='balance')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to .pt file. Default: checkpoints/mpo_<domain>_<task>.pt')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--deterministic', action='store_true', default=True)
    parser.add_argument('--stochastic', action='store_true',
                        help='Use stochastic actions instead of deterministic')
    parser.add_argument('--no-viewer', action='store_true',
                        help='Do not launch the dm_control viewer; just print rewards')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    env = make_env(args.domain, args.task)

    obs_dim = get_obs_dim(env)
    act_dim = get_act_dim(env)
    act_limit = get_act_limit(env)

    agent = MPO(obs_dim, act_dim, act_limit=act_limit, device=args.device)

    if args.checkpoint is None:
        args.checkpoint = os.path.join('checkpoints', f'mpo_{args.domain}_{args.task}.pt')

    if not os.path.exists(args.checkpoint):
        print(f"ERROR: No checkpoint found at {args.checkpoint}")
        print("Train the agent first:  python train.py --domain {} --task {}".format(args.domain, args.task))
        sys.exit(1)

    agent.load(args.checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint}")

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
