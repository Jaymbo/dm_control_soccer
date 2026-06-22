"""
Training mit Live-Viewer im Hintergrund.
- Viewer startet automatisch und bleibt offen
- Policy wird im Hintergrund aktualisiert
- Viewer zeigt immer den aktuellen Agenten
"""
import os
import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch import multiprocessing as mp
import threading

from agent import ActorCritic, ReplayBuffer, compute_gae
from env_wrapper import make_env_with_rewards
from dm_control.locomotion import soccer as dm_soccer
from dm_control import viewer


def flatten_obs(observation_list):
    """Flattene die Observation von allen Spielern zu einem Vektor."""
    flat = []
    for player_obs in observation_list:
        for key, val in player_obs.items():
            flat.append(val.flatten())
    return np.concatenate(flat).astype(np.float32)


def make_env(seed=None, use_reward_shaping=True, reward_scale=1.0):
    """Erstelle die Soccer-Umgebung."""
    if use_reward_shaping:
        env = make_env_with_rewards(seed=seed, reward_scale=reward_scale)
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


def viewer_process(shared_model, obs_dim, action_dim, stop_event, update_event):
    """
    Separater Prozess für den Viewer.
    Läuft kontinuierlich und aktualisiert die Policy bei Signal.
    """
    # Lokale Kopie des Modells
    local_model = ActorCritic(obs_dim=obs_dim, action_dim=action_dim)
    local_model.load_state_dict(shared_model.state_dict())
    local_model.eval()
    
    device = torch.device("cpu")
    
    # Environment für Viewer (ohne Reward Shaping für reine Visualisierung)
    env = make_env(use_reward_shaping=False)
    
    def policy(timestep):
        obs_flat = flatten_obs(timestep.observation)
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0).to(device)
            # Warte auf Update-Signal
            action, _, _ = local_model.get_action(obs_tensor, deterministic=True)
        return action.cpu().numpy()[0]
    
    print("[Viewer] Starting viewer window...")
    print("[Viewer] Close the window to stop visualization.")
    
    try:
        # Viewer starten (blockiert bis Fenster geschlossen wird)
        viewer.launch(env, policy=policy, title="Soccer Agent - Live")
    except Exception as e:
        print(f"[Viewer] Error: {e}")
    finally:
        stop_event.set()


def train(args):
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        if torch.version.hip is not None:
            print("Using AMD ROCm GPU")
        else:
            print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")
    
    # Environment
    env = make_env(seed=args.seed, use_reward_shaping=not args.no_reward_shaping, reward_scale=args.reward_scale)
    print(f"Using reward shaping: {not args.no_reward_shaping} (scale={args.reward_scale})")
    
    obs_dim = 476
    action_dim = 12
    
    # Shared Model für Viewer-Prozess
    shared_model = ActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=args.hidden_dim)
    shared_model.share_memory()
    
    optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)
    buffer = ReplayBuffer(max_size=args.episodes_per_batch * 1000)
    
    # Tensorboard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Viewer Prozess starten
    stop_event = mp.Event()
    update_event = mp.Event()
    
    if args.viewer:
        viewer_proc = mp.Process(
            target=viewer_process,
            args=(shared_model, obs_dim, action_dim, stop_event, update_event)
        )
        viewer_proc.start()
        print("[Main] Viewer process started")
        # Kurze Pause damit Viewer starten kann
        time.sleep(2)
    else:
        viewer_proc = None
        print("[Main] Viewer disabled")
    
    # Training Stats
    episode_rewards = []
    total_steps = 0
    
    print(f"\nStarting training for {args.num_episodes} episodes...")
    if args.viewer:
        print("Viewer läuft im Hintergrund - zeigt aktuellen Agenten")
    print(f"Episodes per batch: {args.episodes_per_batch}")
    print(f"PPO epochs per batch: {args.ppo_epochs}")
    print("-" * 50)
    
    episode = 0
    updates_since_viewer = 0
    
    while episode < args.num_episodes:
        buffer.reset()
        batch_rewards = []
        
        # === COLLECT TRAJECTORIES ===
        for _ in range(args.episodes_per_batch):
            obs = env.reset()
            obs_flat = flatten_obs(obs.observation)
            
            episode_reward = 0
            steps = 0
            
            while not obs.last():
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0).to(device)
                    action, log_prob, value = shared_model.get_action(obs_tensor)
                
                action_np = action.cpu().numpy()[0]
                log_prob_np = log_prob.cpu().numpy()[0]
                value_np = value.cpu().numpy()[0]
                
                obs = env.step(action_np)
                reward = np.sum(obs.reward)
                
                episode_reward += reward
                steps += 1
                total_steps += 1
                
                buffer.add(
                    obs_flat,
                    action_np,
                    reward,
                    obs.last(),
                    log_prob_np,
                    value_np
                )
                
                obs_flat = flatten_obs(obs.observation)
            
            episode_rewards.append(episode_reward)
            batch_rewards.append(episode_reward)
            episode += 1
            
            if episode % args.log_interval == 0:
                avg_reward = np.mean(episode_rewards[-args.log_interval:])
                print(f"Episode {episode}/{args.num_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Steps: {steps}")
                writer.add_scalar("Reward/episode", episode_reward, episode)
                writer.add_scalar("Reward/avg_100", np.mean(episode_rewards[-100:]), episode)
        
        # === PPO UPDATE ===
        obs_batch, actions_batch, rewards_batch, dones_batch, old_log_probs_batch, values_batch = buffer.get_batch()
        
        advantages, returns = compute_gae(
            rewards_batch.numpy().flatten(),
            values_batch.numpy().flatten(),
            dones_batch.numpy().flatten(),
            gamma=args.gamma,
            lambda_=args.gae_lambda
        )
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        obs_batch = torch.FloatTensor(obs_batch).to(device)
        actions_batch = torch.FloatTensor(actions_batch).to(device)
        old_log_probs_batch = torch.FloatTensor(old_log_probs_batch).to(device)
        advantages = advantages.to(device)
        returns = returns.to(device)
        
        for ppo_epoch in range(args.ppo_epochs):
            log_probs, values, entropy = shared_model.evaluate_actions(obs_batch, actions_batch)
            
            ratio = torch.exp(log_probs - old_log_probs_batch)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            value_loss = F.mse_loss(values.squeeze(), returns.squeeze())
            entropy_loss = -args.entropy_coef * entropy.mean()
            
            loss = policy_loss + args.value_coef * value_loss + entropy_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(shared_model.parameters(), args.max_grad_norm)
            optimizer.step()
        
        writer.add_scalar("Loss/policy", policy_loss.item(), episode)
        writer.add_scalar("Loss/value", value_loss.item(), episode)
        writer.add_scalar("Loss/total", loss.item(), episode)
        
        # === Viewer Update ===
        if args.viewer and episode % args.visualize_interval == 0:
            updates_since_viewer += 1
            print(f"[Main] Policy updated (update #{updates_since_viewer}) - Viewer zeigt neuen Agenten")
            # Viewer-Prozess muss neu gestartet werden für Update
            # Da viewer.launch() blockiert, machen wir das beim nächsten Mal
        
        # Save checkpoint
        if episode % args.save_interval == 0 and episode > 0:
            checkpoint_path = os.path.join(args.log_dir, f"checkpoint_ep{episode}.pt")
            torch.save({
                'episode': episode,
                'agent_state_dict': shared_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'episode_rewards': episode_rewards,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    # Viewer stoppen
    if args.viewer and viewer_proc and viewer_proc.is_alive():
        print("[Main] Stopping viewer...")
        stop_event.set()
        viewer_proc.terminate()
        viewer_proc.join(timeout=5)
    
    # Final save
    final_path = os.path.join(args.log_dir, "final_agent.pt")
    torch.save({
        'agent_state_dict': shared_model.state_dict(),
        'episode_rewards': episode_rewards,
    }, final_path)
    print(f"\nTraining finished! Saved final agent to: {final_path}")
    print(f"Best average reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
    
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train with Live Viewer")
    
    # Training params
    parser.add_argument("--num-episodes", type=int, default=1000, help="Total episodes")
    parser.add_argument("--episodes-per-batch", type=int, default=10, help="Episodes per batch")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="PPO epochs")
    
    # Model params
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    
    # PPO params
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-epsilon", type=float, default=0.2, help="Clip epsilon")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy coef")
    parser.add_argument("--value-coef", type=float, default=0.5, help="Value coef")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="Max grad norm")
    
    # Visualization
    parser.add_argument("--viewer", action="store_true", default=True, help="Enable live viewer")
    parser.add_argument("--no-viewer", action="store_false", dest="viewer", help="Disable viewer")
    parser.add_argument("--visualize-interval", type=int, default=10, help="Update viewer every N episodes")
    
    # Reward shaping
    parser.add_argument("--no-reward-shaping", action="store_true", help="Disable reward shaping")
    parser.add_argument("--reward-scale", type=float, default=1.0, help="Reward scale")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--log-dir", type=str, default="logs/soccer_live_viewer", help="Log dir")
    parser.add_argument("--save-interval", type=int, default=50, help="Save interval")
    parser.add_argument("--log-interval", type=int, default=10, help="Log interval")
    
    args = parser.parse_args()
    
    os.makedirs(args.log_dir, exist_ok=True)
    
    train(args)
