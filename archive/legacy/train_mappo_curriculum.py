"""
MAPPO Training mit Curriculum Learning für DM Control Soccer.

Verwendet den schnellen, optimierten MAPPO-Agenten aus agent_mappo_optimized.py,
aber trainiert schrittweise über verschiedene Schwierigkeitsstufen:

  Phase 0 (MOVE):     Agenten lernen, sich aufrecht zu bewegen
  Phase 1 (APPROACH): Agenten lernen, zum Ball zu laufen
  Phase 2 (DRIBBLE):  Agenten lernen, den Ball Richtung Tor zu bewegen
  Phase 3 (SHOOT):    Agenten lernen, Tore zu schießen

Das Curriculum steigt automatisch auf, wenn KPIs erreicht werden.
"""
import os
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

# Optional: Optuna für Hyperparameter-Optimierung
try:
    import optuna
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False

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
from env_wrapper_curriculum import make_env_with_curriculum


def flatten_obs(observation_list):
    """Flattene Observations aller Spieler."""
    flat = []
    for player_obs in observation_list:
        for key, val in player_obs.items():
            flat.append(val.flatten())
    return np.concatenate(flat).astype(np.float32)


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
    """Sammelt eine Episode im Buffer."""
    obs = env.reset()
    obs_flat = flatten_obs(obs.observation)
    episode_reward = 0.0
    steps = 0
    debug_rewards = []

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
        if steps <= 5:  # Erste 5 Steps debuggen
            debug_rewards.append(float(np.sum(rewards)))

        buffer.add(
            observations=obs_per_agent,
            actions=actions_np,
            rewards=rewards,
            dones=[done] * num_agents,
            log_probs=log_probs_np,
            value=value_np,
        )
        obs_flat = flatten_obs(obs.observation)

    # Debug-Stats vom Wrapper holen
    debug_stats = {}
    if hasattr(env, 'get_debug_stats'):
        debug_stats = env.get_debug_stats()

    return episode_reward, steps, debug_stats, debug_rewards


def update_policy(agent, optimizer, buffer, device, args, episode, writer, scaler=None):
    """Führt PPO-Update auf gesammelten Daten durch."""
    obs_batch, actions_batch, rewards_batch, dones_batch, old_log_probs_batch, values_batch = buffer.get_batch()

    rewards_summed = rewards_batch.sum(axis=1)
    dones_any = dones_batch.any(axis=1)

    advantages, returns = compute_gae(
        rewards_summed, values_batch.flatten(), dones_any,
        gamma=args.gamma, lambda_=args.gae_lambda,
    )

    # Advantage Normalisierung
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Value Normalisierung (EMA)
    if not hasattr(agent, '_value_mean'):
        agent._value_mean = returns.mean()
        agent._value_std = returns.std() + 1e-8
    else:
        alpha = 0.1
        agent._value_mean = (1 - alpha) * agent._value_mean + alpha * returns.mean()
        agent._value_std = (1 - alpha) * agent._value_std + alpha * (returns.std() + 1e-8)

    returns_norm = (returns - agent._value_mean) / agent._value_std

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


def train(args, trial=None):
    """
    Training mit optionalem Optuna Trial für Hyperparameter-Optimierung.
    
    Args:
        args: Trainingsparameter
        trial: Optionales Optuna Trial für Pruning
    """
    set_seed(args.seed)
    device = get_device()
    
    # Für Optuna Pruning
    episode_rewards_global = []

    # Callback für Curriculum-Progress
    def on_phase_change(new_phase):
        print(f"\n{'='*60}")
        print(f"🎓 CURRICULUM: Phase {new_phase} erreicht!")
        print(f"{'='*60}\n")
        if args.save_on_phase_change:
            phase_path = os.path.join(args.log_dir, f"checkpoint_phase{new_phase}.pt")
            torch.save({
                'phase': new_phase,
                'agent_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'episode_rewards': episode_rewards,
                'curriculum_phase': new_phase,
            }, phase_path)
            print(f"💾 Phase Checkpoint saved: {phase_path}")

    env = make_env_with_curriculum(
        seed=args.seed,
        curriculum_phase=args.start_phase,
        auto_advance=args.auto_advance,
        phase_episodes=args.phase_episodes,
        phase_success_rate=args.phase_success_rate,
        progress_callback=on_phase_change if args.save_on_phase_change else None,
        reward_scale=args.reward_scale,
    )

    print(f"Curriculum Learning: {args.auto_advance}")
    print(f"Start Phase: {args.start_phase} ({env.PHASE_NAMES[args.start_phase]})")
    print(f"Phase Episodes: {args.phase_episodes}")
    print(f"Success Rate Target: {args.phase_success_rate}")
    print(f"Reward Scale: {args.reward_scale}")
    print("-" * 60)

    num_agents = 4
    obs_dim_per_agent = 119
    action_dim_per_agent = 3

    agent = MAPPOAgent(
        obs_dim_per_agent=obs_dim_per_agent,
        action_dim_per_agent=action_dim_per_agent,
        num_agents=num_agents,
        hidden_dim=args.hidden_dim,
        centralized_critic=True,
        actor_layers=args.actor_layers,
        critic_layers=args.critic_layers,
        use_layer_norm=args.use_layer_norm,
    ).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=args.adam_eps)
    scaler = GradScaler() if device.type == 'cuda' else None
    if scaler is not None:
        print("Using Mixed Precision (AMP)")

    buffer = MAPPOReplayBuffer(max_size=args.episodes_per_batch * 2000, num_agents=num_agents)

    writer = SummaryWriter(log_dir=args.log_dir)
    if not _TENSORBOARD_AVAILABLE:
        print("[WARN] tensorboard not available")

    episode_rewards = []
    total_steps = 0
    start_time = time.time()
    episode = 0
    best_avg_reward = float('-inf')
    best_avg_window = 100

    print(f"\nStarting CURRICULUM MAPPO training for {args.num_episodes} episodes...")
    print(f"Current Phase: {env.curriculum_phase} ({env.phase_name})")
    print(f"Episodes per batch: {args.episodes_per_batch}")
    print(f"PPO epochs: {args.ppo_epochs}, Mini-batch size: {args.mini_batch_size}")
    print("-" * 60)

    while episode < args.num_episodes:
        buffer.reset()
        batch_rewards = []
        batch_steps = 0

        for _ in range(args.episodes_per_batch):
            ep_reward, ep_steps, ep_debug_stats, ep_debug_rewards = collect_episode(
                env, agent, device, num_agents, obs_dim_per_agent, buffer
            )
            episode_rewards.append(ep_reward)
            batch_rewards.append(ep_reward)
            batch_steps += ep_steps
            total_steps += ep_steps
            episode += 1

            # Curriculum-Check nach jeder Episode

            # Curriculum-Check nach jeder Episode
            if args.auto_advance and episode % args.phase_episodes == 0:
                if env.evaluate_phase_progress():
                    env.advance_phase()

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
                            'phase': env.curriculum_phase,
                            'agent_state_dict': agent.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'episode_rewards': episode_rewards,
                            'best_avg_reward': best_avg_reward,
                            'curriculum_phase': env.curriculum_phase,
                        }, best_path)
                        print(f"💾 NEW BEST! Phase {env.curriculum_phase}, Avg{best_avg_window}: {best_avg_reward:.2f}")

                print(f"Ep {episode}/{args.num_episodes} | "
                      f"Phase: {env.curriculum_phase} ({env.phase_name}) | "
                      f"Avg: {avg_reward:8.2f} | "
                      f"Avg100: {avg100:8.2f} | "
                      f"Steps: {total_steps} | "
                      f"Time: {elapsed/60:.1f}m")

                writer.add_scalar("Reward/episode", ep_reward, episode)
                writer.add_scalar("Reward/avg100", avg100, episode)
                writer.add_scalar("Curriculum/phase", env.curriculum_phase, episode)
                
                # Optuna Pruning Report
                if trial is not None and episode % args.log_interval == 0:
                    trial.report(avg100, episode)
                    if trial.should_prune():
                        print(f"  Trial {trial.number} pruned at episode {episode}")
                        raise optuna.TrialPruned()

        # PPO Update
        update_policy(agent, optimizer, buffer, device, args, episode, writer, scaler)

        # Checkpoint
        if episode % args.save_interval == 0 and episode > 0:
            checkpoint_path = os.path.join(args.log_dir, f"checkpoint_ep{episode}.pt")
            torch.save({
                'episode': episode,
                'phase': env.curriculum_phase,
                'agent_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'episode_rewards': episode_rewards,
                'curriculum_phase': env.curriculum_phase,
            }, checkpoint_path)

    # Final Save
    final_path = os.path.join(args.log_dir, "final_agent.pt")
    torch.save({
        'episode': episode,
        'phase': env.curriculum_phase,
        'agent_state_dict': agent.state_dict(),
        'episode_rewards': episode_rewards,
        'curriculum_phase': env.curriculum_phase,
    }, final_path)

    avg100 = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
    print(f"\n{'='*60}")
    print(f"Curriculum MAPPO Training finished!")
    print(f"Final Phase: {env.curriculum_phase} ({env.phase_name})")
    print(f"Avg reward (last 100): {avg100:.2f}")
    print(f"Total steps: {total_steps}")
    print(f"Total time: {(time.time()-start_time)/60:.1f}m")
    print(f"Saved to: {final_path}")
    print(f"{'='*60}")

    writer.close()
    
    # Return für Optuna
    return avg100


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MAPPO Curriculum Training for Soccer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Training
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--episodes-per-batch", type=int, default=20)
    parser.add_argument("--ppo-epochs", type=int, default=10)
    parser.add_argument("--mini-batch-size", type=int, default=256)

    # Modell
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--actor-layers", type=int, default=2)
    parser.add_argument("--critic-layers", type=int, default=2)
    parser.add_argument("--use-layer-norm", action="store_true", default=False)

    # Optimizer
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-decay", type=float, default=0.9)
    parser.add_argument("--adam-eps", type=float, default=1e-5)

    # PPO
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.7)
    parser.add_argument("--entropy-decay", type=float, default=0.95)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)

    # Curriculum
    parser.add_argument("--start-phase", type=int, default=0, choices=[0, 1, 2, 3],
                        help="Startphase: 0=MOVE, 1=APPROACH, 2=DRIBBLE, 3=SHOOT")
    parser.add_argument("--auto-advance", action="store_true", default=True,
                        help="Automatisch zur nächsten Phase bei Erfolg")
    parser.add_argument("--no-auto-advance", action="store_false", dest="auto_advance")
    parser.add_argument("--phase-episodes", type=int, default=40,
                        help="Minimale Episoden pro Phase vor Evaluation")
    parser.add_argument("--phase-success-rate", type=float, default=0.6,
                        help="Erfolgsrate für Phasen-Upgrade (0.0-1.0)")
    parser.add_argument("--save-on-phase-change", action="store_true", default=True,
                        help="Checkpoint bei Phasenwechsel speichern")

    # Reward
    parser.add_argument("--reward-scale", type=float, default=1.0)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str, default="logs/soccer_mappo_curriculum")
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=10)

    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    print("\n" + "="*60)
    print("CURRICULUM MAPPO CONFIGURATION")
    print("="*60)
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")

    train(args)
