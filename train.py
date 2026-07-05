"""Train MPO on dm_control CartPole (balance task).

Usage:
    python train.py --task balance --steps 100000
    python train.py --task swingup   # harder task

The environment can be swapped easily:
    --domain cartpole --task balance   (original dm_control)
    --domain cartpole_ball --task kick  (custom env)
"""
import argparse
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.mpo import MPO


def get_obs_dim(env):
    """Compute flat observation dimension from dm_control observation spec."""
    spec = env.observation_spec()
    dim = sum(np.prod(v.shape) for v in spec.values())
    return int(dim)


def get_act_dim(env):
    spec = env.action_spec()
    return int(np.prod(spec.shape))


def get_act_limit(env):
    spec = env.action_spec()
    return float(np.max(np.abs(spec.maximum)))


def make_env(domain, task):
    """Load a dm_control environment."""
    if domain in ('cartpole', 'cheetah', 'hopper', 'walker', 'pendulum', 'fish',
                  'humanoid', 'point_mass', 'reacher', 'finger', 'manipulator',
                  'acrobot', 'ball_in_cup', 'dog', 'humanoid_CMU', 'lqr'):
        from dm_control import suite
        return suite.load(domain, task)
    else:
        import environments.suite as suite
        return suite.load(domain, task)


def evaluate(env, agent, num_episodes=5, deterministic=True):
    """Run evaluation episodes without storing transitions."""
    rewards = []
    for _ in range(num_episodes):
        time_step = env.reset()
        total_reward = 0.0
        while not time_step.last():
            action = agent.get_action(time_step.observation, deterministic=deterministic)
            action = np.clip(action, env.action_spec().minimum, env.action_spec().maximum)
            time_step = env.step(action)
            total_reward += float(time_step.reward) if time_step.reward is not None else 0.0
        rewards.append(total_reward)
    return rewards


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='cartpole')
    parser.add_argument('--task', type=str, default='balance')
    parser.add_argument('--steps', type=int, default=100000,
                        help='Total number of environment steps')
    parser.add_argument('--start_steps', type=int, default=1000,
                        help='Random exploration steps before training')
    parser.add_argument('--update_every', type=int, default=4,
                        help='Do gradient updates every N env steps')
    parser.add_argument('--updates_per_call', type=int, default=1,
                        help='Gradient steps per update call')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--eval_every', type=int, default=5000,
                        help='Evaluate every N env steps')
    parser.add_argument('--eval_episodes', type=int, default=5)
    parser.add_argument('--print_every', type=int, default=1000,
                        help='Print progress every N env steps')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--timing', action='store_true',
                        help='Print per-phase timing breakdown (env, critic, actor, target, eval)')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    if device == 'auto':
        device = 'cpu'
    print(f"Device: {device}")

    env = make_env(args.domain, args.task)
    eval_env = make_env(args.domain, args.task)

    obs_dim = get_obs_dim(env)
    act_dim = get_act_dim(env)
    act_limit = get_act_limit(env)
    print(f"Env: {args.domain}/{args.task} | obs_dim={obs_dim}, act_dim={act_dim}, act_limit={act_limit}")

    agent = MPO(obs_dim, act_dim, act_limit=act_limit, device=device)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f'mpo_{args.domain}_{args.task}.pt')

    # --- Random exploration ---
    print(f"Collecting {args.start_steps} random steps...", flush=True)
    time_step = env.reset()
    for step in range(args.start_steps):
        action = np.random.uniform(env.action_spec().minimum, env.action_spec().maximum, act_dim)
        next_step = env.step(action)
        reward = float(next_step.reward) if next_step.reward is not None else 0.0
        agent.store(time_step.observation, action, reward, next_step.observation, next_step.last())
        time_step = next_step
        if time_step.last():
            time_step = env.reset()
        if (step + 1) % 500 == 0:
            print(f"  random: {step+1}/{args.start_steps}", flush=True)
    print(f"Random collection done. Buffer size: {agent.buffer.size}", flush=True)

    # --- Training loop (step-based) ---
    total_steps = 0
    episode = 0
    best_eval = -1e9
    t0 = time.time()
    ep_reward = 0.0
    ep_steps = 0
    last_ep_reward = 0.0

    # Timing accumulators (seconds)
    t_env = 0.0
    t_critic = 0.0
    t_actor = 0.0
    t_target = 0.0
    t_eval = 0.0

    time_step = env.reset()

    while total_steps < args.steps:
        # --- Collect one step ---
        _t = time.time()
        obs = time_step.observation
        action = agent.get_action(obs, deterministic=False)
        action = np.clip(action, env.action_spec().minimum, env.action_spec().maximum)
        next_step = env.step(action)
        reward = float(next_step.reward) if next_step.reward is not None else 0.0
        done = next_step.last()
        agent.store(obs, action, reward, next_step.observation, done)
        t_env += time.time() - _t

        ep_reward += reward
        ep_steps += 1
        total_steps += 1
        time_step = next_step

        if done:
            episode += 1
            last_ep_reward = ep_reward
            ep_reward = 0.0
            ep_steps = 0
            time_step = env.reset()

        # --- Gradient updates ---
        if total_steps % args.update_every == 0:
            for _ in range(args.updates_per_call):
                _t = time.time()
                agent.update_critic(agent.buffer.sample_batch(args.batch_size))
                t_critic += time.time() - _t

                _t = time.time()
                agent.update_actor(agent.buffer.sample_batch(args.batch_size))
                t_actor += time.time() - _t

            _t = time.time()
            agent.update_targets()
            t_target += time.time() - _t

        # --- Progress ---
        if total_steps % args.print_every == 0:
            elapsed = time.time() - t0
            print(f"[{total_steps:>7d}/{args.steps}] ep={episode} | "
                  f"last_ep_reward={last_ep_reward:.3f} | "
                  f"cur_ep_reward={ep_reward:.3f} | "
                  f"eta={agent.log_eta.exp().item():.3f} | "
                  f"lam={agent.log_lambda.exp().item():.3f} | "
                  f"fps={total_steps/elapsed:.1f}", flush=True)
            if args.timing:
                total_timed = t_env + t_critic + t_actor + t_target + t_eval
                print(f"    TIMING (s): env={t_env:.2f} ({t_env/total_timed*100:.0f}%) | "
                      f"critic={t_critic:.2f} ({t_critic/total_timed*100:.0f}%) | "
                      f"actor={t_actor:.2f} ({t_actor/total_timed*100:.0f}%) | "
                      f"target={t_target:.2f} ({t_target/total_timed*100:.0f}%) | "
                      f"eval={t_eval:.2f} ({t_eval/total_timed*100:.0f}%)", flush=True)

        # --- Evaluation ---
        if total_steps % args.eval_every == 0:
            _t = time.time()
            eval_rewards = evaluate(eval_env, agent, num_episodes=args.eval_episodes, deterministic=True)
            t_eval += time.time() - _t
            mean_eval = np.mean(eval_rewards)
            print(f"  >>> EVAL @ {total_steps} steps: mean={mean_eval:.3f} "
                  f"(episodes: {[f'{r:.2f}' for r in eval_rewards]})", flush=True)
            if mean_eval > best_eval:
                best_eval = mean_eval
                agent.save(save_path)
                print(f"      Saved best model -> {save_path}", flush=True)

    # Final save
    agent.save(save_path)
    print(f"Training done. Final model saved -> {save_path}", flush=True)


if __name__ == '__main__':
    main()
