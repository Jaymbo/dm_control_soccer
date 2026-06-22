"""
MAPPO-Training mit Dynamic Scoring für DM Control Soccer.

Ersetzt das Phasen-Curriculum durch einen stets aktiven, zustandsbasierten
Reward-Switcher. Nach jedem Batch werden die Rewards nach Branches aufgeteilt
(Recovery, Pursuit, Possession, Defense), um den Lernfortschritt sichtbar zu machen.
"""
import os
import argparse
import subprocess
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import optuna
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

# Optional: MLflow logging
try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except Exception:
    _MLFLOW_AVAILABLE = False

from agent_mappo_optimized import (
    MAPPOAgent, MAPPOReplayBuffer, compute_gae, split_obs_by_agent
)
from env_wrapper_dynamic import make_env_with_dynamic_rewards


def flatten_obs(observation_list):
    """Flattene Observations aller Spieler."""
    flat = []
    for player_obs in observation_list:
        for key, val in player_obs.items():
            flat.append(val.flatten())
    return np.concatenate(flat).astype(np.float32)


def make_env(seed=None, reward_scale=1.0, dynamic_kwargs=None):
    """Erstelle Environment mit Dynamic Scoring Wrapper."""
    dynamic_kwargs = dynamic_kwargs or {}
    env = make_env_with_dynamic_rewards(seed=seed, reward_scale=reward_scale, **dynamic_kwargs)
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

    # Branch-Rewards dieser Episode (vom Wrapper abrufen)
    env.get_branch_rewards(reset=True)

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

    # Branch-Stats am Ende der Episode sammeln
    branch_rewards = env.get_branch_rewards(reset=True)

    return episode_reward, steps, branch_rewards


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
    
    # MLflow: Log loss metrics
    if _MLFLOW_AVAILABLE and mlflow.active_run():
        mlflow.log_metrics({
            "Loss/policy": np.mean(policy_losses),
            "Loss/value": np.mean(value_losses),
            "Loss/entropy": np.mean(entropy_losses),
            "Loss/approx_kl": np.mean(approx_kls),
            "Train/lr": current_lr,
            "Train/entropy_coef": current_entropy,
        }, step=episode)


def train(args, trial=None, mlflow_run_id=None):
    """
    Train MAPPO agent with Dynamic Scoring.
    
    Args:
        args: Training arguments
        trial: Optuna trial (optional, for pruning)
        mlflow_run_id: MLflow run ID (optional, for logging to existing run)
    """
    set_seed(args.seed)
    device = get_device()

    # MLflow: Log to existing run if provided
    if mlflow_run_id and _MLFLOW_AVAILABLE:
        mlflow.set_tags({"mlflow.runId": mlflow_run_id})
        print(f"MLflow: Logging to existing run {mlflow_run_id}")
    elif _MLFLOW_AVAILABLE and mlflow.active_run() is None:
        # No active run - start a new one (shouldn't happen when called from worker_entrypoint)
        print("[WARN] No active MLflow run - starting new one")
        mlflow.start_run(run_name=f"dynamic_training_{int(time.time())}")

    dynamic_kwargs = dict(
        gamma=args.gamma,
        possession_radius=args.possession_radius,
        goal_threshold=args.goal_threshold,
        lambda_recover=args.lambda_recover,
        lambda_pursuit=args.lambda_pursuit,
        lambda_possession=args.lambda_possession,
        lambda_defense=args.lambda_defense,
    )

    env = make_env(
        seed=args.seed,
        reward_scale=args.reward_scale,
        dynamic_kwargs=dynamic_kwargs,
    )
    print(f"Dynamic Scoring: active (scale={args.reward_scale})")
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

    print(f"\nStarting DYNAMIC SCORING MAPPO training for {args.num_episodes} episodes...")
    print(f"Episodes per batch: {args.episodes_per_batch}")
    print(f"PPO epochs: {args.ppo_epochs}, Mini-batch size: {args.mini_batch_size}")
    print("-" * 60)

    while episode < args.num_episodes:
        buffer.reset()
        batch_rewards = []
        batch_steps = 0
        batch_branch_sums = {"recovery": 0.0, "pursuit": 0.0, "possession": 0.0, "defense": 0.0}

        for _ in range(args.episodes_per_batch):
            ep_reward, ep_steps, ep_branch_rewards = collect_episode(
                env, agent, device, num_agents, obs_dim_per_agent, buffer
            )
            episode_rewards.append(ep_reward)
            batch_rewards.append(ep_reward)
            batch_steps += ep_steps
            total_steps += ep_steps
            episode += 1

            for branch, value in ep_branch_rewards.items():
                batch_branch_sums[branch] += value

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

                # MLflow: Log metrics periodically
                if _MLFLOW_AVAILABLE and mlflow.active_run() and episode % args.log_interval == 0:
                    mlflow.log_metrics({
                        "Reward/avg_interval": avg_reward,
                        "Reward/avg_100": avg100,
                        "Reward/episode": ep_reward,
                        "Steps/total": total_steps,
                    }, step=episode)

                # Optuna Pruning Report
                if trial is not None:
                    trial.report(avg100, episode)
                    if trial.should_prune():
                        print(f"  Trial {trial.number} pruned at episode {episode}")
                        raise optuna.TrialPruned()

        # Branch-Reward-Verteilung nach jedem Batch loggen
        for branch, value in batch_branch_sums.items():
            writer.add_scalar(f"Reward/branch_{branch}", value, episode)
            writer.add_scalar(f"Reward/branch_{branch}_per_episode",
                              value / max(args.episodes_per_batch, 1), episode)
        
        # MLflow: Log branch rewards
        if _MLFLOW_AVAILABLE and mlflow.active_run():
            branch_metrics = {
                f"Reward/branch_{branch}": value / max(args.episodes_per_batch, 1)
                for branch, value in batch_branch_sums.items()
            }
            mlflow.log_metrics(branch_metrics, step=episode)

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
    print(f"Dynamic Scoring MAPPO Training finished!")
    print(f"Avg reward (last 100): {avg100:.2f}")
    print(f"Total steps: {total_steps}")
    print(f"Total time: {(time.time()-start_time)/60:.1f}m")
    print(f"Saved to: {final_path}")
    print(f"{'='*60}")

    if args.eval_at_end and checkpoint_path is not None:
        show_viewer(checkpoint_path, str(device))

    # MLflow: Log final model as artifact
    if _MLFLOW_AVAILABLE and mlflow.active_run():
        try:
            mlflow.log_artifact(final_path, "models")
            mlflow.log_artifact(best_path if os.path.exists(best_path) else final_path, "models")
        except Exception as e:
            print(f"[WARN] Could not log MLflow artifacts: {e}")

    writer.close()
    return avg100


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

    # Dynamic Scoring
    parser.add_argument("--reward-scale", type=float, default=1.0,
                        help="Globaler Skalierungsfaktor für alle Branches")
    parser.add_argument("--possession-radius", type=float, default=0.6,
                        help="Maximaler Ballabstand für Ballbesitz (m)")
    parser.add_argument("--goal-threshold", type=float, default=6.0,
                        help="Torlinien-Distanz für Schuss-Detection (m)")
    parser.add_argument("--lambda-recover", type=float, default=1.0,
                        help="Gewicht RECOVERY-Branch")
    parser.add_argument("--lambda-pursuit", type=float, default=1.0,
                        help="Gewicht PURSUIT-Branch")
    parser.add_argument("--lambda-possession", type=float, default=1.0,
                        help="Gewicht POSSESSION-Branch")
    parser.add_argument("--lambda-defense", type=float, default=1.0,
                        help="Gewicht DEFENSE-Branch")

    # Viewer
    parser.add_argument("--viewer", action="store_true", default=False)
    parser.add_argument("--viewer-interval", type=int, default=100)
    parser.add_argument("--eval-at-end", action="store_true")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str, default="logs/soccer_mappo_dynamic")
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=10)

    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    print("\n" + "="*60)
    print("DYNAMIC SCORING MAPPO CONFIGURATION")
    print("="*60)
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")

    train(args)
