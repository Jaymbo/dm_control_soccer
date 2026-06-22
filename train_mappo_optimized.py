"""
Optimiertes MAPPO-Training für DM Control Soccer.

Ziel: maximales Lernspeed durch
- vektorisierte Batch-Updates (keine Python-Loops pro Timestep)
- Mini-Batch PPO
- besseres Reward Shaping
- adaptive Hyperparameter (LR/Entropy-Annealing)
- längere Rollouts & mehr PPO-Epochs

Altes `train.py` und `train_mappo.py` bleiben unverändert.
"""
import os
import argparse
import subprocess
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from dm_control.locomotion import soccer as dm_soccer

try:
    from torch.utils.tensorboard import SummaryWriter
    _TENSORBOARD_AVAILABLE = True
except Exception:
    _TENSORBOARD_AVAILABLE = False
    class _DummySummaryWriter:
        def __init__(self, *args, **kwargs):
            pass
        def add_scalar(self, *args, **kwargs):
            pass
        def close(self):
            pass
    SummaryWriter = _DummySummaryWriter

from agent_mappo_optimized import (
    MAPPOAgent, MAPPOReplayBuffer, compute_gae, split_obs_by_agent
)
from env_wrapper_optimized import make_env_with_rewards_optimized


def flatten_obs(observation_list):
    """Flattene Observations aller Spieler."""
    flat = []
    for player_obs in observation_list:
        for key, val in player_obs.items():
            flat.append(val.flatten())
    return np.concatenate(flat).astype(np.float32)


def make_env(seed=None, use_reward_shaping=True, reward_scale=1.0, wrapper_kwargs=None):
    """Erstelle Environment."""
    if use_reward_shaping:
        wrapper_kwargs = wrapper_kwargs or {}
        env = make_env_with_rewards_optimized(seed=seed, reward_scale=reward_scale, **wrapper_kwargs)
    else:
        env = dm_soccer.load(
            team_size=2,
            time_limit=10.0,
            disable_walker_contacts=False,
            enable_field_box=True,
            terminate_on_goal=False,
            walker_type=dm_soccer.WalkerType.BOXHEAD
        )
        if seed is not None:
            env.task._random_state = np.random.RandomState(seed)
    return env


def show_viewer(checkpoint_path, device):
    """Starte dm_control viewer mit dem aktuellen Checkpoint."""
    print(f"\n{'='*60}")
    print(f"VISUALIZATION - Checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")

    viewer_script = f'''
import torch
import numpy as np
from dm_control.locomotion import soccer as dm_soccer
from dm_control import viewer
from agent_mappo_optimized import MAPPOAgent, split_obs_by_agent

def flatten_obs(obs):
    flat = []
    for player_obs in obs:
        for key, val in player_obs.items():
            flat.append(val.flatten())
    return np.concatenate(flat).astype(np.float32)

checkpoint = torch.load("{checkpoint_path}", map_location="{device}", weights_only=False)
agent = MAPPOAgent(obs_dim_per_agent=119, action_dim_per_agent=3, num_agents=4)
agent.load_state_dict(checkpoint['agent_state_dict'])
agent.eval()

env = dm_soccer.load(
    team_size=2, time_limit=10.0, disable_walker_contacts=False,
    enable_field_box=True, terminate_on_goal=False,
    walker_type=dm_soccer.WalkerType.BOXHEAD
)

def policy(timestep):
    obs_flat = flatten_obs(timestep.observation)
    obs_per_agent = split_obs_by_agent(obs_flat, num_agents=4, obs_dim_per_agent=119)
    with torch.no_grad():
        actions, _, _ = agent.get_actions(obs_per_agent, deterministic=True)
    return np.concatenate([a.cpu().numpy() for a in actions])

viewer.launch(env, policy=policy, title="MAPPO Optimized Soccer Agent")
'''
    path = '/tmp/soccer_mappo_optimized_viewer.py'
    with open(path, 'w') as f:
        f.write(viewer_script)
    try:
        subprocess.run(['python', path], timeout=60, cwd=os.getcwd())
    except subprocess.TimeoutExpired:
        print("[Viewer] Timeout - closing...")
    except Exception as e:
        print(f"[Viewer] Error: {e}")
    finally:
        if os.path.exists(path):
            os.remove(path)
    print("[Viewer] Closed\n")


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        if torch.version.hip is not None:
            print("Using AMD ROCm GPU")
        else:
            print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("Using Apple MPS")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")


def collect_episode(env, agent, device, num_agents, obs_dim_per_agent, buffer):
    obs = env.reset()
    obs_flat = flatten_obs(obs.observation)
    episode_reward = 0.0
    steps = 0

    while not obs.last():
        obs_per_agent = split_obs_by_agent(obs_flat, num_agents, obs_dim_per_agent)
        obs_tensor = torch.FloatTensor(np.stack(obs_per_agent, axis=0)).unsqueeze(0).to(device)

        with torch.no_grad():
            actions, log_probs, value = agent.get_actions(obs_tensor, deterministic=False)

        actions_np = actions.squeeze(0).cpu().numpy()
        log_probs_np = log_probs.squeeze(0).cpu().numpy()
        value_np = value.squeeze(0).cpu().numpy()

        obs = env.step(actions_np.flatten())
        rewards = obs.reward
        done = obs.last()

        episode_reward += float(np.sum(rewards))
        steps += 1

        buffer.add(
            observations=obs_per_agent,
            actions=actions_np,
            rewards=rewards,
            dones=[done] * num_agents,
            log_probs=log_probs_np,
            value=value_np,
        )
        obs_flat = flatten_obs(obs.observation)

    return episode_reward, steps


def update_policy(agent, optimizer, buffer, device, args, episode, writer, scaler=None):
    obs_batch, actions_batch, rewards_batch, dones_batch, old_log_probs_batch, values_batch = buffer.get_batch()

    # rewards_batch: (T, num_agents) -> summiere über Agenten für globale Value
    rewards_summed = rewards_batch.sum(axis=1)  # (T,)
    dones_any = dones_batch.any(axis=1)         # (T,)

    advantages, returns = compute_gae(
        rewards_summed,
        values_batch.flatten(),
        dones_any,
        gamma=args.gamma,
        lambda_=args.gae_lambda,
    )

    # Advantage Normalisierung
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Value Normalisierung (laufender Mean/Std für stabileres Training)
    if not hasattr(agent, '_value_mean'):
        agent._value_mean = returns.mean()
        agent._value_std = returns.std() + 1e-8
    else:
        # Exponential Moving Average
        alpha = 0.1
        agent._value_mean = (1 - alpha) * agent._value_mean + alpha * returns.mean()
        agent._value_std = (1 - alpha) * agent._value_std + alpha * (returns.std() + 1e-8)

    returns_norm = (returns - agent._value_mean) / agent._value_std

    # Tensors
    obs_t = torch.FloatTensor(obs_batch).to(device)
    actions_t = torch.FloatTensor(actions_batch).to(device)
    old_log_probs_t = torch.FloatTensor(old_log_probs_batch).to(device)
    advantages_t = advantages.to(device)
    returns_norm_t = returns_norm.to(device)

    T, num_agents, obs_dim = obs_t.shape
    total_timesteps = T

    # Annealing
    progress = min(1.0, episode / args.num_episodes)
    current_lr = args.lr * (1.0 - progress * args.lr_decay)
    current_entropy = args.entropy_coef * (1.0 - progress * args.entropy_decay)
    for pg in optimizer.param_groups:
        pg['lr'] = current_lr

    # PPO Mini-Batch Updates
    dataset_size = total_timesteps
    indices = np.arange(dataset_size)

    policy_losses = []
    value_losses = []
    entropy_losses = []
    approx_kls = []
    use_amp = scaler is not None and device.type == 'cuda'

    for epoch in range(args.ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, dataset_size, args.mini_batch_size):
            end = min(start + args.mini_batch_size, dataset_size)
            mb_idx = indices[start:end]

            mb_obs = obs_t[mb_idx]
            mb_actions = actions_t[mb_idx]
            mb_old_log_probs = old_log_probs_t[mb_idx]
            mb_advantages = advantages_t[mb_idx]
            mb_returns = returns_norm_t[mb_idx]

            if use_amp:
                with autocast():
                    log_probs, values, entropy = agent.evaluate_actions(mb_obs, mb_actions)
                    log_probs_sum = log_probs.sum(dim=1)
                    old_log_probs_sum = mb_old_log_probs.sum(dim=1)
                    entropy_mean = entropy.mean(dim=1)

                    ratio = torch.exp(log_probs_sum - old_log_probs_sum)
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon) * mb_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = F.mse_loss(values.squeeze(-1), mb_returns)
                    entropy_loss = -current_entropy * entropy_mean.mean()
                    loss = policy_loss + args.value_coef * value_loss + entropy_loss

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                # Vektorisierte Evaluation
                log_probs, values, entropy = agent.evaluate_actions(mb_obs, mb_actions)
                log_probs_sum = log_probs.sum(dim=1)
                old_log_probs_sum = mb_old_log_probs.sum(dim=1)
                entropy_mean = entropy.mean(dim=1)

                ratio = torch.exp(log_probs_sum - old_log_probs_sum)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values.squeeze(-1), mb_returns)
                entropy_loss = -current_entropy * entropy_mean.mean()
                loss = policy_loss + args.value_coef * value_loss + entropy_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy_loss.item())
            approx_kls.append(approx_kl)

    # Logging
    writer.add_scalar("Loss/policy", np.mean(policy_losses), episode)
    writer.add_scalar("Loss/value", np.mean(value_losses), episode)
    writer.add_scalar("Loss/entropy", np.mean(entropy_losses), episode)
    writer.add_scalar("Loss/approx_kl", np.mean(approx_kls), episode)
    writer.add_scalar("Train/lr", current_lr, episode)
    writer.add_scalar("Train/entropy_coef", current_entropy, episode)


def train(args):
    set_seed(args.seed)
    device = get_device()

    wrapper_kwargs = {}
    if args.reward_config == "aggressive":
        wrapper_kwargs = dict(
            ball_proximity_weight=0.1,
            ball_to_goal_weight=2.0,
            moving_to_ball_weight=0.8,
            possession_bonus=0.5,
            shot_to_goal_weight=1.5,
            movement_bonus=0.3,
            fall_penalty=0.5,
            idle_penalty=0.2,
        )
    elif args.reward_config == "balanced":
        wrapper_kwargs = dict(
            ball_proximity_weight=0.1,
            ball_to_goal_weight=1.5,
            moving_to_ball_weight=0.6,
            possession_bonus=0.3,
            shot_to_goal_weight=1.0,
            movement_bonus=0.2,
            fall_penalty=0.5,
            idle_penalty=0.15,
        )

    env = make_env(
        seed=args.seed,
        use_reward_shaping=not args.no_reward_shaping,
        reward_scale=args.reward_scale,
        wrapper_kwargs=wrapper_kwargs,
    )
    print(f"Reward Shaping: {not args.no_reward_shaping} (scale={args.reward_scale}, config={args.reward_config})")
    print(f"Centralized Critic: {args.centralized_critic}")

    num_agents = 4
    obs_dim_per_agent = 119
    action_dim_per_agent = 3

    agent = MAPPOAgent(
        obs_dim_per_agent=obs_dim_per_agent,
        action_dim_per_agent=action_dim_per_agent,
        num_agents=num_agents,
        hidden_dim=args.hidden_dim,
        centralized_critic=args.centralized_critic,
        actor_layers=args.actor_layers,
        critic_layers=args.critic_layers,
        use_layer_norm=args.use_layer_norm,
    ).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=args.adam_eps)

    # Mixed Precision Scaler für GPU
    scaler = GradScaler() if device.type == 'cuda' else None
    if scaler is not None:
        print("Using Mixed Precision (AMP) for faster GPU training")

    buffer = MAPPOReplayBuffer(max_size=args.episodes_per_batch * 2000, num_agents=num_agents)

    writer = SummaryWriter(log_dir=args.log_dir)
    if not _TENSORBOARD_AVAILABLE:
        print("[WARN] tensorboard nicht verfügbar - Logging deaktiviert.")

    episode_rewards = []
    total_steps = 0
    start_time = time.time()
    episode = 0
    checkpoint_path = None
    
    # Best Model Tracking
    best_avg_reward = float('-inf')
    best_avg_window = 100  # Fenster für "best" Berechnung

    print(f"\nStarting optimized MAPPO training for {args.num_episodes} episodes...")
    print(f"Episodes per batch: {args.episodes_per_batch}")
    print(f"PPO epochs: {args.ppo_epochs}, Mini-batch size: {args.mini_batch_size}")
    print("-" * 60)

    while episode < args.num_episodes:
        buffer.reset()
        batch_rewards = []
        batch_steps = 0

        for _ in range(args.episodes_per_batch):
            ep_reward, ep_steps = collect_episode(
                env, agent, device, num_agents, obs_dim_per_agent, buffer
            )
            episode_rewards.append(ep_reward)
            batch_rewards.append(ep_reward)
            batch_steps += ep_steps
            total_steps += ep_steps
            episode += 1

            if episode % args.log_interval == 0:
                avg_reward = np.mean(episode_rewards[-args.log_interval:])
                avg100 = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
                elapsed = time.time() - start_time
                
                # Best Model Check
                if len(episode_rewards) >= best_avg_window:
                    current_avg = np.mean(episode_rewards[-best_avg_window:])
                    if current_avg > best_avg_reward:
                        best_avg_reward = current_avg
                        best_path = os.path.join(args.log_dir, "best_agent.pt")
                        torch.save({
                            'episode': episode,
                            'agent_state_dict': agent.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'episode_rewards': episode_rewards,
                            'best_avg_reward': best_avg_reward,
                            'best_avg_window': best_avg_window,
                        }, best_path)
                        print(f"💾 NEW BEST MODEL! Avg{best_avg_window}: {best_avg_reward:.2f} -> Saved: {best_path}")
                
                print(f"Episode {episode}/{args.num_episodes} | "
                      f"Avg Reward: {avg_reward:8.2f} | "
                      f"Avg100: {avg100:8.2f} | "
                      f"Steps: {total_steps} | "
                      f"Time: {elapsed/60:.1f}m")
                writer.add_scalar("Reward/episode", ep_reward, episode)
                writer.add_scalar("Reward/avg_interval", avg_reward, episode)
                writer.add_scalar("Reward/avg_100", avg100, episode)
                writer.add_scalar("Reward/batch_steps", batch_steps, episode)

        # Update
        update_policy(agent, optimizer, buffer, device, args, episode, writer, scaler)

        # Periodischer Viewer
        if args.viewer and episode > 0 and episode % args.viewer_interval == 0:
            checkpoint_path = os.path.join(args.log_dir, "checkpoint_current.pt")
            torch.save({
                'episode': episode,
                'agent_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'episode_rewards': episode_rewards,
            }, checkpoint_path)
            show_viewer(checkpoint_path, str(device))

        # Checkpoint
        if episode % args.save_interval == 0 and episode > 0:
            checkpoint_path = os.path.join(args.log_dir, f"checkpoint_ep{episode}.pt")
            torch.save({
                'episode': episode,
                'agent_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'episode_rewards': episode_rewards,
            }, checkpoint_path)
            print(f"Saved: {checkpoint_path}")

    final_path = os.path.join(args.log_dir, "final_agent.pt")
    torch.save({
        'agent_state_dict': agent.state_dict(),
        'episode_rewards': episode_rewards,
    }, final_path)

    avg100 = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
    print(f"\n{'='*60}")
    print(f"Optimized MAPPO Training finished!")
    print(f"Avg reward (last 100): {avg100:.2f}")
    print(f"Total steps: {total_steps}")
    print(f"Total time: {(time.time()-start_time)/60:.1f}m")
    print(f"Saved to: {final_path}")
    print(f"{'='*60}")

    if args.eval_at_end and checkpoint_path is not None:
        show_viewer(checkpoint_path, str(device))

    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimized MAPPO Training for Soccer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Training
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--episodes-per-batch", type=int, default=20,
                        help="Episodes pro PPO-Update (höher = stabiler)")
    parser.add_argument("--ppo-epochs", type=int, default=10,
                        help="Anzahl PPO-Epochs pro Update")
    parser.add_argument("--mini-batch-size", type=int, default=256,
                        help="Mini-Batch Größe für PPO-Updates")

    # Modell
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--actor-layers", type=int, default=2)
    parser.add_argument("--critic-layers", type=int, default=2)
    parser.add_argument("--use-layer-norm", action="store_true", default=False,
                        help="LayerNorm in Actor/Critic")

    # Optimizer
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-decay", type=float, default=0.9,
                        help="Linearer LR-Decay bis 0 über Training")
    parser.add_argument("--adam-eps", type=float, default=1e-5)

    # PPO
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.7,
                        help="Entropy Bonus am Anfang (höher = mehr Exploration)")
    parser.add_argument("--entropy-decay", type=float, default=0.95,
                        help="Entropy-Decay: 0.95 = sinkt auf ~0.01 am Ende")
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    # Critic
    parser.add_argument("--centralized-critic", action="store_true", default=True)
    parser.add_argument("--decentralized-critic", action="store_false", dest="centralized_critic")

    # Reward
    parser.add_argument("--no-reward-shaping", action="store_true")
    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--reward-config", type=str, default="balanced",
                        choices=["balanced", "aggressive"],
                        help="Vordefinierte Reward-Shaping-Konfiguration")

    # Viewer
    parser.add_argument("--viewer", action="store_true", default=False)
    parser.add_argument("--viewer-interval", type=int, default=100)
    parser.add_argument("--eval-at-end", action="store_true")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str, default="logs/soccer_mappo_optimized")
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=10)

    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    print("\n" + "="*60)
    print("OPTIMIZED MAPPO CONFIGURATION")
    print("="*60)
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")

    train(args)
