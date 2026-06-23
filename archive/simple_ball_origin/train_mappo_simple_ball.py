"""
Simple Ball-Chase MAPPO Training.

NUR ein Ziel: So schnell wie möglich zum Ball laufen!
- Kein Marking, kein Shooting, kein Goalkeeping
- Nur: Ball-Chaser bekommt Reward für Annäherung + Speed

Perfekt für den Trainingsstart!
"""
import os
import argparse
import subprocess
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
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

try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except Exception:
    _MLFLOW_AVAILABLE = False

from agent_mappo_optimized import MAPPOAgent, split_obs_by_agent
from env_wrapper_simple_ball import make_env_with_simple_ball_chase


# ------------------------------------------------------------------
# Parallele Environments
# ------------------------------------------------------------------

def make_env(seed=None, reward_scale=1.0, team_size=2, simple_kwargs=None, time_limit=10.0):
    """Erstelle ein einzelnes Environment mit Simple Ball-Chase Wrapper."""
    simple_kwargs = simple_kwargs or {}
    return make_env_with_simple_ball_chase(
        seed=seed,
        reward_scale=reward_scale,
        team_size=team_size,
        time_limit=time_limit,
        **simple_kwargs,
    )


class ParallelEnvWorker:
    """
    Worker für ein einzelnes Environment, der Schritt-für-Schritt abläuft
    und kurze Rollouts (N Steps) liefert.
    """

    def __init__(self, worker_id, seed, reward_scale, team_size, simple_kwargs, time_limit):
        self.worker_id = worker_id
        self.env = make_env(
            seed=seed,
            reward_scale=reward_scale,
            team_size=team_size,
            simple_kwargs=simple_kwargs,
            time_limit=time_limit,
        )
        self.team_size = team_size
        self.num_agents = team_size * 2
        self.obs_dim_per_agent = 119
        self.obs = self.env.reset()
        self.obs_flat = self._flatten_obs(self.obs.observation)
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.last_log_step = 0

    def _flatten_obs(self, observation_list):
        flat = []
        for player_obs in observation_list:
            for key in sorted(player_obs.keys()):
                flat.append(np.asarray(player_obs[key]).flatten())
        return np.concatenate(flat).astype(np.float32)

    def step(self, actions):
        """
        Führt einen einzelnen Schritt aus.
        actions: (num_agents, action_dim)
        Rückgabe: (obs_flat_next, reward_sum, done, info)
        """
        self.obs = self.env.step(actions.flatten())
        rewards = np.asarray(self.obs.reward, dtype=np.float32)
        done = self.obs.last()
        self.episode_reward += float(np.sum(rewards))
        self.episode_steps += 1

        # Track rewards auch für laufende Episodes (für detailliertes Logging)
        current_step_reward = float(np.sum(rewards))

        info = {
            "done": done,
            "episode_reward": self.episode_reward if done else None,
            "episode_steps": self.episode_steps if done else None,
            "branch_rewards": self.env.get_branch_rewards(reset=done),
            "current_step_reward": current_step_reward,
        }

        self.obs_flat = self._flatten_obs(self.obs.observation)

        if done:
            self.obs = self.env.reset()
            self.obs_flat = self._flatten_obs(self.obs.observation)
            self.episode_reward = 0.0
            self.episode_steps = 0
            self.last_log_step = 0

        return self.obs_flat.copy(), np.sum(rewards), done, info


# ------------------------------------------------------------------
# Tensorboard / Logging
# ------------------------------------------------------------------

def show_viewer(checkpoint_path, device, team_size=2, time_limit=10.0):
    """Starte dm_control viewer mit dem aktuellen Checkpoint."""
    print(f"\n{'='*60}")
    print(f"VISUALIZATION - Checkpoint: {checkpoint_path}")
    print(f"{'='*60}\n")

    num_agents = team_size * 2
    obs_dim_per_agent = 119

    viewer_script = f'''
import torch
import numpy as np
from dm_control.locomotion import soccer as dm_soccer
from dm_control import viewer
from agent_mappo_optimized import MAPPOAgent, split_obs_by_agent

def flatten_obs(obs):
    flat = []
    for player_obs in obs:
        for key in sorted(player_obs.keys()):
            flat.append(np.asarray(player_obs[key]).flatten())
    return np.concatenate(flat).astype(np.float32)

checkpoint = torch.load("{checkpoint_path}", map_location="{device}", weights_only=False)
agent = MAPPOAgent(obs_dim_per_agent={obs_dim_per_agent}, action_dim_per_agent=3, num_agents={num_agents})
agent.load_state_dict(checkpoint['agent_state_dict'])
agent.eval()

env = dm_soccer.load(
    team_size={team_size}, time_limit={time_limit}, disable_walker_contacts=False,
    enable_field_box=True, terminate_on_goal=False,
    walker_type=dm_soccer.WalkerType.BOXHEAD
)

def policy(timestep):
    obs_flat = flatten_obs(timestep.observation)
    obs_per_agent = split_obs_by_agent(obs_flat, num_agents={num_agents}, obs_dim_per_agent={obs_dim_per_agent})
    with torch.no_grad():
        actions, _, _ = agent.get_actions(obs_per_agent, deterministic=True)
    return np.concatenate([a.cpu().numpy() for a in actions])

viewer.launch(env, policy=policy, title="MAPPO Dynamic V2 Online Soccer Agent")
'''
    path = '/tmp/soccer_mappo_dynamic_v2_online_viewer.py'
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


# ------------------------------------------------------------------
# Online MAPPO Update
# ------------------------------------------------------------------

def compute_nstep_returns(rewards, values, dones, gamma=0.99):
    """
    Berechnet N-Step Returns für einen Rollout.
    rewards/values/dones: (T, num_envs) oder (T,)
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)

    # values enthält T+1 Werte: [V(s_0), V(s_1), ..., V(s_T)]
    # rewards/dones haben Länge T
    T = len(rewards)
    returns = np.zeros_like(rewards)
    running = values[T] * (1.0 - dones[-1])  # Bootstrap mit letztem Value
    for t in reversed(range(T)):
        running = rewards[t] + gamma * running * (1.0 - dones[t])
        returns[t] = running

    # Advantages nur für die T tatsächlichen Steps
    advantages = returns - values[:T]
    return torch.FloatTensor(returns), torch.FloatTensor(advantages)


def collect_rollout(workers, agent, device, rollout_steps, obs_dim_per_agent, deterministic=False):
    """
    Sammelt rollout_steps von allen parallelen Environments.
    Rückgabe: (rollout_tensors, episode_stats)
      rollout_tensors: (obs, actions, old_log_probs, rewards, dones, values)
      episode_stats: dict mit 'episode_rewards', 'episode_steps', 'branch_sums', 'env_rewards'
    """
    num_envs = len(workers)
    num_agents = workers[0].num_agents

    obs_buffer = []
    actions_buffer = []
    log_probs_buffer = []
    rewards_buffer = []
    dones_buffer = []
    values_buffer = []

    episode_rewards = []
    episode_steps = []
    branch_sums = {b: 0.0 for b in workers[0].env.BRANCH_NAMES}
    
    # Per-Environment Reward-Tracking (für detailliertes Logging)
    env_rewards = {i: [] for i in range(num_envs)}

    for step in range(rollout_steps):
        obs_stack = np.stack([w.obs_flat for w in workers], axis=0)
        obs_per_agent = np.stack(split_obs_by_agent(obs_stack, num_agents, obs_dim_per_agent), axis=1)
        obs_t = torch.FloatTensor(obs_per_agent).to(device)

        with torch.no_grad():
            actions, log_probs, values = agent.get_actions(obs_t, deterministic=deterministic)

        actions_np = actions.cpu().numpy()
        log_probs_np = log_probs.cpu().numpy()
        values_np = values.cpu().numpy()

        results = []
        for env_id in range(num_envs):
            results.append(workers[env_id].step(actions_np[env_id]))

        rewards = np.array([r[1] for r in results], dtype=np.float32)
        dones = np.array([r[2] for r in results], dtype=np.float32)
        infos = [r[3] for r in results]

        # Per-Environment Rewards tracken
        for env_id, info in enumerate(infos):
            env_rewards[env_id].append(info["current_step_reward"])
            if info["done"]:
                episode_rewards.append(info["episode_reward"])
                episode_steps.append(info["episode_steps"])
                for b, v in info["branch_rewards"].items():
                    branch_sums[b] += v

        obs_buffer.append(obs_per_agent)
        actions_buffer.append(actions_np)
        log_probs_buffer.append(log_probs_np)
        rewards_buffer.append(rewards)
        dones_buffer.append(dones)
        values_buffer.append(values_np.squeeze(-1))

    # Bootstrap-Wert
    obs_stack_last = np.stack([w.obs_flat for w in workers], axis=0)
    obs_per_agent_last = np.stack(split_obs_by_agent(obs_stack_last, num_agents, obs_dim_per_agent), axis=1)
    with torch.no_grad():
        _, _, last_values = agent.get_actions(torch.FloatTensor(obs_per_agent_last).to(device))
        last_values_np = last_values.cpu().numpy().squeeze(-1)

    values_buffer.append(last_values_np)

    rollout_tensors = (
        np.array(obs_buffer, dtype=np.float32),
        np.array(actions_buffer, dtype=np.float32),
        np.array(log_probs_buffer, dtype=np.float32),
        np.array(rewards_buffer, dtype=np.float32),
        np.array(dones_buffer, dtype=np.float32),
        np.array(values_buffer, dtype=np.float32),
    )

    episode_stats = {
        "episode_rewards": episode_rewards,
        "episode_steps": episode_steps,
        "branch_sums": branch_sums,
        "num_episodes": len(episode_rewards),
        "env_rewards": env_rewards,  # Per-Environment step rewards
    }

    return rollout_tensors, episode_stats


def update_policy(agent, optimizer, rollout_data, args, update_step, writer, scaler=None):
    """
    Führt ein PPO-Update auf dem gesammelten Rollout durch.
    rollout_data: (obs, actions, old_log_probs, rewards, dones, values)
    """
    obs, actions, old_log_probs, rewards, dones, values = rollout_data
    T, num_envs, num_agents, obs_dim = obs.shape

    # Pro Environment N-Step Returns
    returns_envs = []
    advantages_envs = []
    for env_id in range(num_envs):
        ret, adv = compute_nstep_returns(
            rewards[:, env_id],
            values[:, env_id],
            dones[:, env_id],
            gamma=args.gamma,
        )
        returns_envs.append(ret)
        advantages_envs.append(adv)

    returns = torch.stack(returns_envs, dim=1)      # (T, num_envs)
    advantages = torch.stack(advantages_envs, dim=1) # (T, num_envs)

    # Auf (T*num_envs, num_agents, ...) umformen
    obs_t = torch.FloatTensor(obs).to(device).view(T * num_envs, num_agents, obs_dim)
    actions_t = torch.FloatTensor(actions).to(device).view(T * num_envs, num_agents, -1)
    old_log_probs_t = torch.FloatTensor(old_log_probs).to(device).view(T * num_envs, num_agents, -1)
    advantages_t = advantages.view(T * num_envs, 1).to(device)
    returns_t = returns.view(T * num_envs, 1).to(device)

    # Advantage Normalisierung - ROBUST mit Minimum-Std
    adv_mean = advantages_t.mean()
    adv_std = advantages_t.std()
    if adv_std < 1e-3:
        advantages_t = advantages_t - adv_mean
    else:
        advantages_t = (advantages_t - adv_mean) / (adv_std + 1e-8)
    advantages_t = torch.clamp(advantages_t, -3.0, 3.0)

    # Returns: pro Batch zentrieren (keine laufende Norm, die instabil ist)
    returns_mean = returns_t.mean()
    returns_std = returns_t.std()
    if returns_std < 1e-3:
        returns_norm_t = returns_t - returns_mean
    else:
        returns_norm_t = (returns_t - returns_mean) / (returns_std + 1e-8)
    returns_norm_t = torch.clamp(returns_norm_t, -3.0, 3.0)

    # Lernraten-Decay
    progress = min(1.0, update_step / args.num_updates)
    current_lr = args.lr * (1.0 - progress * args.lr_decay)
    current_entropy = args.entropy_coef * (1.0 - progress * args.entropy_decay)
    for pg in optimizer.param_groups:
        pg['lr'] = current_lr

    dataset_size = T * num_envs
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
                    
                    # NaN-Check nach Forward-Pass
                    if torch.isnan(log_probs).any() or torch.isnan(values).any():
                        print(f"[WARN] NaN detected in forward pass (AMP)! Skipping batch...")
                        optimizer.zero_grad()
                        continue
                    
                    log_probs_sum = log_probs.sum(dim=1)
                    old_log_probs_sum = mb_old_log_probs.sum(dim=1)
                    entropy_mean = entropy.mean(dim=1)

                    ratio = torch.exp(log_probs_sum - old_log_probs_sum)
                    surr1 = ratio * mb_advantages
                    surr2 = torch.clamp(ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon) * mb_advantages
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_loss = F.mse_loss(values.squeeze(-1), mb_returns.squeeze(-1))
                    entropy_loss = -current_entropy * entropy_mean.mean()
                    loss = policy_loss + args.value_coef * value_loss + entropy_loss
                    
                    # NaN-Check vor Backward
                    if torch.isnan(loss):
                        print(f"[WARN] NaN in loss (AMP)! Skipping batch...")
                        optimizer.zero_grad()
                        continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                
                # Gradient NaN-Check (nach unscale)
                has_nan_grad = False
                for param in agent.parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    print(f"[WARN] NaN in gradients (AMP)! Skipping update...")
                    optimizer.zero_grad()
                    continue
                
                torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                log_probs, values, entropy = agent.evaluate_actions(mb_obs, mb_actions)
                
                # NaN-Check nach Forward-Pass
                if torch.isnan(log_probs).any() or torch.isnan(values).any():
                    print(f"[WARN] NaN detected in forward pass! Skipping batch...")
                    optimizer.zero_grad()
                    continue
                
                log_probs_sum = log_probs.sum(dim=1)
                old_log_probs_sum = mb_old_log_probs.sum(dim=1)
                entropy_mean = entropy.mean(dim=1)

                ratio = torch.exp(log_probs_sum - old_log_probs_sum)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(values.squeeze(-1), mb_returns.squeeze(-1))
                entropy_loss = -current_entropy * entropy_mean.mean()
                loss = policy_loss + args.value_coef * value_loss + entropy_loss

                # NaN-Check vor Backward
                if torch.isnan(loss):
                    print(f"[WARN] NaN in loss! Skipping batch...")
                    optimizer.zero_grad()
                    continue

                optimizer.zero_grad()
                loss.backward()
                
                # Gradient NaN-Check
                has_nan_grad = False
                for param in agent.parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    print(f"[WARN] NaN in gradients! Skipping update...")
                    optimizer.zero_grad()
                    continue
                
                torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy_loss.item())
            approx_kls.append(approx_kl)

    writer.add_scalar("Loss/policy", np.mean(policy_losses), update_step)
    writer.add_scalar("Loss/value", np.mean(value_losses), update_step)
    writer.add_scalar("Loss/entropy", np.mean(entropy_losses), update_step)
    writer.add_scalar("Loss/approx_kl", np.mean(approx_kls), update_step)
    writer.add_scalar("Train/lr", current_lr, update_step)
    writer.add_scalar("Train/entropy_coef", current_entropy, update_step)

    # Reward/Return Statistik für Diagnose
    writer.add_scalar("Stats/returns_mean", returns_t.mean().item(), update_step)
    writer.add_scalar("Stats/returns_std", returns_t.std().item(), update_step)
    writer.add_scalar("Stats/returns_min", returns_t.min().item(), update_step)
    writer.add_scalar("Stats/returns_max", returns_t.max().item(), update_step)
    writer.add_scalar("Stats/adv_mean", adv_mean.item(), update_step)
    writer.add_scalar("Stats/adv_std", adv_std.item(), update_step)
    writer.add_scalar("Stats/rewards_mean", rewards.mean().item(), update_step)
    writer.add_scalar("Stats/rewards_max", rewards.max().item(), update_step)

    if _MLFLOW_AVAILABLE and mlflow.active_run():
        mlflow.log_metrics({
            "Loss/policy": np.mean(policy_losses),
            "Loss/value": np.mean(value_losses),
            "Loss/entropy": np.mean(entropy_losses),
            "Loss/approx_kl": np.mean(approx_kls),
            "Train/lr": current_lr,
            "Train/entropy_coef": current_entropy,
        }, step=update_step)


def train(args):
    set_seed(args.seed)
    global device
    device = get_device()

    if _MLFLOW_AVAILABLE and mlflow.active_run() is None:
        mlflow.start_run(run_name=f"dynamic_v2_online_{int(time.time())}")

    num_agents = args.team_size * 2
    obs_dim_per_agent = 119

    simple_kwargs = dict(
        possession_weight=args.possession_weight,
        proximity_weight=args.proximity_weight,
        time_penalty=args.time_penalty,
    )

    # Parallele Environments erstellen
    workers = []
    for i in range(args.num_envs):
        seed = args.seed + i * 1000
        worker = ParallelEnvWorker(
            worker_id=i,
            seed=seed,
            reward_scale=args.reward_scale,
            team_size=args.team_size,
            simple_kwargs=simple_kwargs,
            time_limit=args.time_limit,
        )
        workers.append(worker)

    print(f"Created {args.num_envs} parallel environments")

    agent = MAPPOAgent(
        obs_dim_per_agent=obs_dim_per_agent,
        action_dim_per_agent=3,
        num_agents=num_agents,
        hidden_dim=args.hidden_dim,
        centralized_critic=args.centralized_critic,
        actor_layers=args.actor_layers,
        critic_layers=args.critic_layers,
        use_layer_norm=args.use_layer_norm,
    ).to(device)

    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=args.adam_eps)
    scaler = GradScaler() if device.type == 'cuda' else None
    if scaler is not None:
        print("Using Mixed Precision (AMP)")

    writer = SummaryWriter(log_dir=args.log_dir)
    if not _TENSORBOARD_AVAILABLE:
        print("[WARN] TensorBoard nicht verfügbar")

    episode_rewards = []
    total_steps = 0
    total_episodes = 0
    start_time = time.time()
    best_avg_reward = float('-inf')
    checkpoint_path = None

    print(f"\nStarting SIMPLE BALL-CHASE V10 MAPPO training (SYMMETRISCH + KONTROLLE)")
    print(f"Team size: {args.team_size}v{args.team_size} ({num_agents} agents)")
    print(f"Parallel envs: {args.num_envs}, Rollout steps: {args.rollout_steps}")
    print(f"Updates: {args.num_updates}, PPO epochs: {args.ppo_epochs}")
    print(f"LR: {args.lr}, Entropy: {args.entropy_coef}, Max Grad Norm: {args.max_grad_norm}")
    print(f"Possession Bonus: SYMMETRISCH (max +{args.possession_weight * 2:.3f}, nur < 4m)")
    print(f"Delta Reward: +2.0 pro Meter Annäherung")
    print(f"Chaser Bonus: +0.15 pro Step (nur eigener Chaser, < 4m)")
    print(f"Ball Control Bonus: +0.25 pro Step (wenn < 1m)")
    print(f"Proximity Weight: {args.proximity_weight} (sehr klein)")
    print(f"Time Penalty: -{args.time_penalty} pro Step")
    print("-" * 60)

    for update_step in range(1, args.num_updates + 1):
        rollout_data, ep_stats = collect_rollout(
            workers, agent, device,
            args.rollout_steps, obs_dim_per_agent,
            deterministic=False,
        )

        update_policy(agent, optimizer, rollout_data, args, update_step, writer, scaler)

        total_steps += args.num_envs * args.rollout_steps
        total_episodes += ep_stats["num_episodes"]
        episode_rewards.extend(ep_stats["episode_rewards"])

        if update_step % args.log_interval == 0:
            avg100 = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards) if episode_rewards else 0.0
            avg_steps = np.mean(ep_stats["episode_steps"]) if ep_stats["episode_steps"] else 0.0
            elapsed = time.time() - start_time
            
            # Per-Environment Durchschnitts-Rewards berechnen
            env_avg_rewards = {}
            for env_id, rewards in ep_stats.get("env_rewards", {}).items():
                if rewards:
                    env_avg_rewards[env_id] = np.mean(rewards)
            
            print(f"Update {update_step}/{args.num_updates} | "
                  f"Episodes: {total_episodes} | "
                  f"Steps: {total_steps} | "
                  f"Avg100 Reward: {avg100:8.2f} | "
                  f"Avg Ep Steps: {avg_steps:6.1f} | "
                  f"Time: {elapsed/60:.1f}m")
            
            # Per-Environment Rewards anzeigen
            if env_avg_rewards:
                env_rewards_str = " | ".join([f"Env{i}: {r:6.4f}" for i, r in sorted(env_avg_rewards.items())])
                print(f"  Per-Env Avg: {env_rewards_str}")
            
            writer.add_scalar("Reward/avg_100", avg100, update_step)
            writer.add_scalar("Steps/total", total_steps, update_step)

            # Branch Reward Breakdown
            if ep_stats["num_episodes"] > 0:
                batch_total = sum(ep_stats["branch_sums"].values())
                print(f"\n--- Branch Breakdown (since last log, {ep_stats['num_episodes']} episodes) ---")
                sorted_branches = sorted(ep_stats["branch_sums"].items(), key=lambda x: abs(x[1]), reverse=True)
                for branch, total_value in sorted_branches:
                    avg_per_ep = total_value / ep_stats["num_episodes"]
                    pct = (total_value / abs(batch_total) * 100) if abs(batch_total) > 0.01 else 0.0
                    sign = "+" if total_value > 0 else "-" if total_value < 0 else " "
                    print(f"  {branch:<20s}: {total_value:>8.2f} total | {avg_per_ep:>7.3f}/ep | {pct:>5.1f}% | {sign}")
                    writer.add_scalar(f"Reward/branch_{branch}", avg_per_ep, update_step)
                print(f"  {'TOTAL':<20s}: {batch_total:>8.2f} total")
                print("-" * 60)

            if len(episode_rewards) >= 100:
                current_avg = np.mean(episode_rewards[-100:])
                if current_avg > best_avg_reward:
                    best_avg_reward = current_avg
                    best_path = os.path.join(args.log_dir, "best_agent.pt")
                    torch.save({
                        'update': update_step,
                        'agent_state_dict': agent.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'episode_rewards': episode_rewards,
                        'best_avg_reward': best_avg_reward,
                    }, best_path)
                    print(f"NEW BEST MODEL! Avg100: {best_avg_reward:.2f} -> Saved: {best_path}")

        if args.viewer and update_step > 0 and update_step % args.viewer_interval == 0:
            checkpoint_path = os.path.join(args.log_dir, "checkpoint_current.pt")
            torch.save({
                'update': update_step,
                'agent_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'episode_rewards': episode_rewards,
            }, checkpoint_path)
            show_viewer(checkpoint_path, str(device), team_size=args.team_size, time_limit=args.time_limit)

        if update_step % args.save_interval == 0 and update_step > 0:
            checkpoint_path = os.path.join(args.log_dir, f"checkpoint_up{update_step}.pt")
            torch.save({
                'update': update_step,
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

    avg100 = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards) if episode_rewards else 0.0
    print(f"\n{'='*60}")
    print(f"Simple Ball-Chase MAPPO Training finished!")
    print(f"Updates: {args.num_updates}, Total steps: {total_steps}, Episodes: {total_episodes}")
    print(f"Avg reward (last 100): {avg100:.2f}")
    print(f"Total time: {(time.time()-start_time)/60:.1f}m")
    print(f"Saved to: {final_path}")
    print(f"{'='*60}")

    if args.eval_at_end and checkpoint_path is not None:
        show_viewer(checkpoint_path, str(device), team_size=args.team_size, time_limit=args.time_limit)

    writer.close()

    if _MLFLOW_AVAILABLE and mlflow.active_run():
        try:
            best_path = os.path.join(args.log_dir, "best_agent.pt")
            mlflow.log_artifact(final_path, "models")
            mlflow.log_artifact(best_path if os.path.exists(best_path) else final_path, "models")
        except Exception as e:
            print(f"[WARN] Could not log MLflow artifacts: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Online Dynamic Scoring V2 MAPPO Training for Soccer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--num-updates", type=int, default=1000,
                        help="Anzahl der Policy-Updates (jeder = ein Rollout + Update)")
    parser.add_argument("--num-envs", type=int, default=10,
                        help="Anzahl paralleler Environments")
    parser.add_argument("--rollout-steps", type=int, default=32,
                        help="Schritte pro Environment pro Rollout")
    parser.add_argument("--ppo-epochs", type=int, default=4,
                        help="PPO Epochs pro Update (niedriger, da häufiger updates)")
    parser.add_argument("--mini-batch-size", type=int, default=256)

    parser.add_argument("--team-size", type=int, default=2,
                        help="Spieler pro Team (2=2v2)")

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--actor-layers", type=int, default=3)
    parser.add_argument("--critic-layers", type=int, default=3)
    parser.add_argument("--use-layer-norm", action="store_true", default=False)

    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate - 1e-4 ist Standard für Stabilität")
    parser.add_argument("--lr-decay", type=float, default=0.99)
    parser.add_argument("--adam-eps", type=float, default=1e-5,
                        help="Adam epsilon - Standard ist 1e-5")

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="Entropy coefficient - niedriger (0.01) für stabiles Training")
    parser.add_argument("--entropy-decay", type=float, default=0.999)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="Gradient clipping norm - 0.5 ist Standard")

    parser.add_argument("--centralized-critic", action="store_true", default=True)
    parser.add_argument("--decentralized-critic", action="store_false", dest="centralized_critic")

    parser.add_argument("--reward-scale", type=float, default=1.0)
    parser.add_argument("--possession-weight", type=float, default=0.02,
                        help="Bonus wenn Team näher ist (wird mit 5 skaliert → 0.1), default: 0.02")
    parser.add_argument("--proximity-weight", type=float, default=0.01,
                        help="Sehr kleiner Bonus für Ball-Nähe, default: 0.01")
    parser.add_argument("--time-penalty", type=float, default=0.0001,
                        help="Kleine Strafe pro Step (verhindert Stillstand, default: 0.0001)")

    parser.add_argument("--viewer", action="store_true", default=False)
    parser.add_argument("--viewer-interval", type=int, default=100)
    parser.add_argument("--eval-at-end", action="store_true")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str, default="logs/soccer_mappo_simple_ball")
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=10)

    parser.add_argument("--time-limit", type=float, default=20.0,
                        help="Episode-Dauer in Sekunden (20s für echtes Soccer-Verhalten)")

    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)

    print("\n" + "="*60)
    print("ONLINE DYNAMIC SCORING V2 MAPPO CONFIGURATION")
    print("="*60)
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")

    train(args)
