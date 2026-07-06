"""Train PPO on dm_control environments with MLflow logging.

Usage:
    python train_ppo.py --domain cartpole --task balance --steps 100000
    python train_ppo.py --steps 0                                # endless until Ctrl+C

MLflow metrics logged per evaluation step:
    ep_reward, eval_reward, policy_loss, value_loss, entropy, clip_frac, approx_kl, fps

Hyperparameters can be overridden via CLI args or environment variables
(used by Optuna/MLflow HPO in hpo_ppo.py).
"""
import argparse
import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.ppo import PPO

# SQLite busy timeout (ms) — allows multiple parallel workers to wait
# instead of immediately failing with "database is locked".
SQLITE_BUSY_TIMEOUT_MS = 30000


def _add_sqlite_busy_timeout(uri: str) -> str:
    """Append busy_timeout pragma to a sqlite:/// URI (no-op for other URIs)."""
    if uri.startswith('sqlite:///'):
        sep = '&' if '?' in uri else '?'
        if 'busy_timeout' not in uri:
            uri = f'{uri}{sep}busy_timeout={SQLITE_BUSY_TIMEOUT_MS}'
    return uri


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
            action, _, _ = agent.get_action(time_step.observation, deterministic=deterministic)
            action = np.clip(action, env.action_spec().minimum, env.action_spec().maximum)
            time_step = env.step(action)
            total_reward += float(time_step.reward) if time_step.reward is not None else 0.0
        rewards.append(total_reward)
    return rewards


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--domain', type=str, default='cartpole')
    p.add_argument('--task', type=str, default='balance')
    p.add_argument('--steps', type=int, default=100000,
                   help='Total environment steps (0 = endless, run until Ctrl+C)')
    p.add_argument('--eval_every', type=int, default=5000)
    p.add_argument('--eval_episodes', type=int, default=5)
    p.add_argument('--final_eval_episodes', type=int, default=10,
                   help='Number of episodes for final evaluation after training')
    p.add_argument('--print_every', type=int, default=1000)
    p.add_argument('--save_dir', type=str, default='checkpoints')
    p.add_argument('--checkpoint_tag', type=str, default='',
                   help='Append tag to checkpoint filename (e.g. for parallel HPO workers)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--timing', action='store_true')
    p.add_argument('--no-resume', dest='resume', action='store_false', default=True,
                   help='Start from scratch even if a checkpoint exists')

    # --- PPO hyperparameters ---
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--lam', type=float, default=0.95, help='GAE lambda')
    p.add_argument('--clip_eps', type=float, default=0.2)
    p.add_argument('--actor_lr', type=float, default=3e-4)
    p.add_argument('--critic_lr', type=float, default=1e-3)
    p.add_argument('--entropy_coef', type=float, default=0.0)
    p.add_argument('--value_coef', type=float, default=0.5)
    p.add_argument('--max_grad_norm', type=float, default=0.5)
    p.add_argument('--rollout_size', type=int, default=2048,
                   help='Steps collected per rollout before each update')
    p.add_argument('--update_epochs', type=int, default=10,
                   help='Epochs over the rollout data per update')
    p.add_argument('--num_minibatches', type=int, default=32)
    p.add_argument('--lr_schedule', type=str, default='linear',
                   choices=['linear', 'constant'],
                   help='Learning rate schedule: linear decay to 0 or constant')

    # --- MLflow ---
    p.add_argument('--mlflow', action='store_true', default=True,
                   help='Log metrics to MLflow (default: on)')
    p.add_argument('--no-mlflow', dest='mlflow', action='store_false')
    p.add_argument('--mlflow_tracking_uri', type=str, default='sqlite:///mlflow.db')
    p.add_argument('--mlflow_experiment', type=str, default=None,
                   help='MLflow experiment name (default: <domain>_<task>)')
    p.add_argument('--mlflow_run_name', type=str, default=None)
    return p


def main():
    args = build_parser().parse_args()

    # Default experiment name: domain_task (e.g. cartpole_balance)
    if args.mlflow_experiment is None:
        args.mlflow_experiment = f'{args.domain}_{args.task}'

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else args.device
    if device == 'auto':
        device = 'cpu'
    print(f"Device: {device}")

    # --- MLflow setup ---
    use_mlflow = args.mlflow
    if use_mlflow:
        import mlflow
        mlflow_uri = _add_sqlite_busy_timeout(args.mlflow_tracking_uri)
        for attempt in range(5):
            try:
                mlflow.set_tracking_uri(mlflow_uri)
                mlflow.set_experiment(args.mlflow_experiment)
                mlflow.start_run(run_name=args.mlflow_run_name)
                break
            except Exception as e:
                if attempt < 4:
                    print(f"MLflow init attempt {attempt+1} failed ({e}), retrying...")
                    time.sleep(2 ** attempt)
                else:
                    raise
        hp_keys = [
            'domain', 'task', 'steps', 'eval_every', 'seed',
            'gamma', 'lam', 'clip_eps', 'actor_lr', 'critic_lr',
            'entropy_coef', 'value_coef', 'max_grad_norm',
            'rollout_size', 'update_epochs', 'num_minibatches', 'lr_schedule',
        ]
        mlflow.log_params({k: getattr(args, k) for k in hp_keys})

    env = make_env(args.domain, args.task)
    eval_env = make_env(args.domain, args.task)

    obs_dim = get_obs_dim(env)
    act_dim = get_act_dim(env)
    act_limit = get_act_limit(env)
    print(f"Env: {args.domain}/{args.task} | obs_dim={obs_dim}, act_dim={act_dim}, act_limit={act_limit}")

    agent = PPO(
        obs_dim, act_dim, act_limit=act_limit, device=device,
        gamma=args.gamma, lam=args.lam, clip_eps=args.clip_eps,
        actor_lr=args.actor_lr, critic_lr=args.critic_lr,
        entropy_coef=args.entropy_coef, value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        rollout_size=args.rollout_size,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_name = f'ppo_{args.domain}_{args.task}'
    if args.checkpoint_tag:
        ckpt_name += f'_{args.checkpoint_tag}'
    save_path = os.path.join(args.save_dir, f'{ckpt_name}.pt')

    # --- Resume from checkpoint if it exists ---
    total_steps = 0
    best_eval = -1e9
    final_eval = -1e9
    if args.resume and os.path.exists(save_path):
        print(f"Loading checkpoint: {save_path}", flush=True)
        total_steps, best_eval, final_eval = agent.load(save_path)
        print(f"  Resumed from {total_steps} steps | best_eval={best_eval:.3f} | "
              f"final_eval={final_eval:.3f}", flush=True)

    # --- Training loop ---
    episode = 0
    t0 = time.time()
    ep_reward = 0.0
    ep_steps = 0
    last_ep_reward = 0.0

    t_env = 0.0
    t_train = 0.0
    t_eval = 0.0

    time_step = env.reset()
    results = {}

    max_steps = args.steps if args.steps > 0 else float('inf')
    if max_steps == float('inf'):
        print("Endless training. Press Ctrl+C to stop.\n", flush=True)

    while total_steps < max_steps:
        # --- Collect rollout ---
        _t = time.time()
        rollout_steps = min(args.rollout_size, int(max_steps - total_steps)) if max_steps != float('inf') else args.rollout_size
        for _ in range(rollout_steps):
            obs = time_step.observation
            action, val, logp = agent.get_action(obs, deterministic=False)
            action = np.clip(action, env.action_spec().minimum, env.action_spec().maximum)
            next_step = env.step(action)
            reward = float(next_step.reward) if next_step.reward is not None else 0.0
            episode_done = next_step.last()
            agent.store(obs, action, reward, val, logp, done=episode_done)
            ep_reward += reward
            ep_steps += 1
            total_steps += 1
            time_step = next_step

            if episode_done:
                episode += 1
                last_ep_reward = ep_reward
                if use_mlflow:
                    import mlflow
                    mlflow.log_metric('ep_reward', ep_reward, step=total_steps)
                ep_reward = 0.0
                ep_steps = 0
                time_step = env.reset()
        t_env += time.time() - _t

        # --- LR annealing ---
        if args.lr_schedule == 'linear' and max_steps != float('inf'):
            frac = max(0.0, 1.0 - total_steps / max_steps)
            for pg in agent.policy_optim.param_groups:
                pg['lr'] = args.actor_lr * frac
            for pg in agent.value_optim.param_groups:
                pg['lr'] = args.critic_lr * frac

        # --- Compute GAE and update ---
        _t = time.time()
        with torch.no_grad():
            last_obs_t = agent.obs_to_tensor(time_step.observation)
            last_val = agent.value(last_obs_t).item()
        agent.rollout.compute_gae(last_val, gamma=args.gamma, lam=args.lam)
        results = agent.update()
        t_train += time.time() - _t

        # --- Progress ---
        if total_steps % args.print_every < args.rollout_size:
            elapsed = time.time() - t0
            steps_display = f'{args.steps}' if args.steps > 0 else 'inf'
            print(f"[{total_steps:>7d}/{steps_display}] ep={episode} | "
                  f"ep_reward={last_ep_reward:.1f} | "
                  f"pi_loss={results.get('policy_loss', 0.0):.4f} | "
                  f"v_loss={results.get('value_loss', 0.0):.4f} | "
                  f"entropy={results.get('entropy', 0.0):.3f} | "
                  f"clip={results.get('clip_frac', 0.0):.3f} | "
                  f"kl={results.get('approx_kl', 0.0):.5f} | "
                  f"fps={total_steps/elapsed:.1f}", flush=True)
            if args.timing:
                total_timed = t_env + t_train + t_eval
                print(f"    TIMING (s): env={t_env:.2f} ({t_env/total_timed*100:.0f}%) | "
                      f"train={t_train:.2f} ({t_train/total_timed*100:.0f}%) | "
                      f"eval={t_eval:.2f} ({t_eval/total_timed*100:.0f}%)", flush=True)

        # --- Evaluation ---
        if total_steps % args.eval_every < args.rollout_size:
            _t = time.time()
            eval_rewards = evaluate(eval_env, agent, num_episodes=args.eval_episodes, deterministic=True)
            t_eval += time.time() - _t
            mean_eval = float(np.mean(eval_rewards))
            print(f"  >>> EVAL @ {total_steps} steps: mean={mean_eval:.3f} "
                  f"(episodes: {[f'{r:.2f}' for r in eval_rewards]})", flush=True)

            if use_mlflow:
                import mlflow
                mlflow.log_metric('eval_reward', mean_eval, step=total_steps)
                if results:
                    mlflow.log_metric('policy_loss', results.get('policy_loss', 0.0), step=total_steps)
                    mlflow.log_metric('value_loss', results.get('value_loss', 0.0), step=total_steps)
                    mlflow.log_metric('entropy', results.get('entropy', 0.0), step=total_steps)
                    mlflow.log_metric('clip_frac', results.get('clip_frac', 0.0), step=total_steps)
                    mlflow.log_metric('approx_kl', results.get('approx_kl', 0.0), step=total_steps)
                mlflow.log_metric('fps', total_steps / (time.time() - t0), step=total_steps)

            if mean_eval > best_eval:
                best_eval = mean_eval
                agent.save(save_path, total_steps=total_steps, best_eval=best_eval)
                print(f"      Saved best model -> {save_path}", flush=True)

    # --- Final evaluation ---
    final_rewards = evaluate(eval_env, agent, num_episodes=args.final_eval_episodes, deterministic=True)
    final_eval = float(np.mean(final_rewards))
    print(f"  >>> FINAL_EVAL @ {total_steps} steps: mean={final_eval:.3f} "
          f"({args.final_eval_episodes} episodes: {[f'{r:.2f}' for r in final_rewards]})", flush=True)

    agent.save(save_path, total_steps=total_steps, best_eval=best_eval, final_eval=final_eval)
    print(f"Training done. Final model saved -> {save_path}", flush=True)

    if use_mlflow:
        import mlflow
        mlflow.log_metric('best_eval_reward', best_eval)
        mlflow.log_metric('final_eval_reward', final_eval)
        mlflow.end_run()

    return final_eval


if __name__ == '__main__':
    main()
