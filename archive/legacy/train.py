"""
Unified Training Script für DM Control Soccer mit PPO.

Alle Features in einem Script:
- Training mit/ohne Visualisierung
- Reward Shaping einstellbar
- Periodischer Viewer alle N Episoden
- Checkpoint Saving
- Tensorboard Logging
- AMD GPU (ROCm) Support

Beispiele:
  python train.py --num-episodes 1000                    # Standard Training
  python train.py --viewer --viewer-interval 50          # Mit Viewer alle 50 Episoden
  python train.py --no-reward-shaping                    # Ohne Reward Shaping
  python train.py --viewer --eval-at-end                 # Viewer am Ende
"""
import os
import argparse
import subprocess
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dm_control.locomotion import soccer as dm_soccer

from agent import ActorCritic, ReplayBuffer, compute_gae
from env_wrapper import make_env_with_rewards


def flatten_obs(observation_list):
    """Flattene die Observation von allen Spielern zu einem Vektor."""
    flat = []
    for player_obs in observation_list:
        for key, val in player_obs.items():
            flat.append(val.flatten())
    return np.concatenate(flat).astype(np.float32)


def make_env(seed=None, use_reward_shaping=True, reward_scale=1.0):
    """
    Erstelle die Soccer-Umgebung.
    
    Args:
        seed: Random seed
        use_reward_shaping: Wenn True, verwende shaped rewards (dichtere Signale)
        reward_scale: Skalierungsfaktor für shaped rewards
    """
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


def show_viewer(checkpoint_path, device, num_episodes=1):
    """
    Öffnet Viewer für Visualisierung mit dem gegebenen Checkpoint.
    Wird als subprocess ausgeführt und nach der Episode automatisch geschlossen.
    
    Args:
        checkpoint_path: Pfad zum Checkpoint
        device: Device (cuda/cpu)
        num_episodes: Anzahl der Episoden zur Visualisierung
    """
    print(f"\n{'='*60}")
    print(f"VISUALIZATION - Checkpoint: {checkpoint_path}")
    print(f"Viewer öffnet sich für {num_episodes} Episode(n)...")
    print(f"{'='*60}\n")
    
    # Script für Visualisierung
    viewer_script = f'''
import torch
import numpy as np
from dm_control.locomotion import soccer as dm_soccer
from dm_control import viewer
from agent import ActorCritic
from env_wrapper import make_env_with_rewards

def flatten_obs(obs):
    flat = []
    for player_obs in obs:
        for key, val in player_obs.items():
            flat.append(val.flatten())
    return np.concatenate(flat).astype(np.float32)

# Load agent
checkpoint = torch.load("{checkpoint_path}", map_location="{device}", weights_only=False)
agent = ActorCritic(obs_dim=476, action_dim=12)
agent.load_state_dict(checkpoint['agent_state_dict'])
agent.eval()

# Environment
env = dm_soccer.load(
    team_size=2, time_limit=10.0, disable_walker_contacts=False,
    enable_field_box=True, terminate_on_goal=False,
    walker_type=dm_soccer.WalkerType.BOXHEAD
)

episode_count = 0

def policy(timestep):
    global episode_count
    obs_flat = flatten_obs(timestep.observation)
    with torch.no_grad():
        obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0)
        action, _, _ = agent.get_action(obs_tensor, deterministic=True)
    return action.cpu().numpy()[0]

# Run viewer
viewer.launch(env, policy=policy, title="Soccer Agent")
'''
    
    # Temporäres Script schreiben
    with open('/tmp/soccer_viewer_temp.py', 'w') as f:
        f.write(viewer_script)
    
    # Als subprocess ausführen (blockiert bis Viewer geschlossen)
    try:
        subprocess.run(
            ['python', '/tmp/soccer_viewer_temp.py'], 
            timeout=60,
            cwd=os.getcwd()  # Wichtig: Im Projektverzeichnis ausführen
        )
    except subprocess.TimeoutExpired:
        print("[Viewer] Timeout - closing...")
    except Exception as e:
        print(f"[Viewer] Error: {e}")
    finally:
        # Cleanup
        if os.path.exists('/tmp/soccer_viewer_temp.py'):
            os.remove('/tmp/soccer_viewer_temp.py')
    
    print(f"[Viewer] Closed - continuing...\n")


def train(args):
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device selection - supports CUDA, ROCm (AMD), and CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if torch.version.hip is not None:
            print(f"Using AMD ROCm GPU")
        else:
            print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU (no GPU available)")
    
    # Environment erstellen (mit Reward Shaping für bessere Lernsignale)
    use_shaping = not args.no_reward_shaping
    env = make_env(seed=args.seed, use_reward_shaping=use_shaping, reward_scale=args.reward_scale)
    print(f"Using reward shaping: {use_shaping} (scale={args.reward_scale})")
    
    # Hyperparameter aus Environment ableiten
    obs_dim = 476  # Aus der Analyse
    action_dim = 12  # 4 Spieler × 3 Actions
    
    # Agent, Buffer, Optimizer
    agent = ActorCritic(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.lr)
    buffer = ReplayBuffer(max_size=args.episodes_per_batch * 1000)
    
    # Tensorboard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Training Stats
    episode_rewards = []
    total_steps = 0
    
    print(f"\nStarting training for {args.num_episodes} episodes...")
    print(f"Episodes per batch: {args.episodes_per_batch}")
    print(f"PPO epochs per batch: {args.ppo_epochs}")
    if args.viewer:
        print(f"Viewer: Enabled (every {args.viewer_interval} episodes)")
    else:
        print(f"Viewer: Disabled")
    print("-" * 50)
    
    episode = 0
    checkpoint_path = None
    
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
                # Action von Policy holen
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0).to(device)
                    action, log_prob, value = agent.get_action(obs_tensor)
                
                action_np = action.cpu().numpy()[0]
                log_prob_np = log_prob.cpu().numpy()[0]
                value_np = value.cpu().numpy()[0]
                
                # Environment step
                obs = env.step(action_np)
                reward = np.sum(obs.reward)  # Team reward
                
                episode_reward += reward
                steps += 1
                total_steps += 1
                
                # Speichern im Buffer
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
        
        # Compute GAE
        advantages, returns = compute_gae(
            rewards_batch.numpy().flatten(),
            values_batch.numpy().flatten(),
            dones_batch.numpy().flatten(),
            gamma=args.gamma,
            lambda_=args.gae_lambda
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        obs_batch = torch.FloatTensor(obs_batch).to(device)
        actions_batch = torch.FloatTensor(actions_batch).to(device)
        old_log_probs_batch = torch.FloatTensor(old_log_probs_batch).to(device)
        advantages = advantages.to(device)
        returns = returns.to(device)
        
        # PPO Epochs
        for ppo_epoch in range(args.ppo_epochs):
            # Evaluate current policy
            log_probs, values, entropy = agent.evaluate_actions(obs_batch, actions_batch)
            
            # Ratio
            ratio = torch.exp(log_probs - old_log_probs_batch)
            
            # Clipped surrogate loss
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values.squeeze(), returns.squeeze())
            
            # Entropy bonus
            entropy_loss = -args.entropy_coef * entropy.mean()
            
            # Total loss
            loss = policy_loss + args.value_coef * value_loss + entropy_loss
            
            # Update
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()
        
        # Log PPO stats
        writer.add_scalar("Loss/policy", policy_loss.item(), episode)
        writer.add_scalar("Loss/value", value_loss.item(), episode)
        writer.add_scalar("Loss/entropy", entropy_loss.item(), episode)
        writer.add_scalar("Loss/total", loss.item(), episode)
        
        # === VIEWER (periodic) ===
        if args.viewer and episode > 0 and episode % args.viewer_interval == 0:
            # Save current checkpoint
            checkpoint_path = os.path.join(args.log_dir, f"checkpoint_current.pt")
            torch.save({
                'episode': episode,
                'agent_state_dict': agent.state_dict(),
                'episode_rewards': episode_rewards,
            }, checkpoint_path)
            
            # Show viewer
            show_viewer(checkpoint_path, str(device))
        
        # === REGULAR CHECKPOINT ===
        if episode % args.save_interval == 0 and episode > 0:
            checkpoint_path = os.path.join(args.log_dir, f"checkpoint_ep{episode}.pt")
            torch.save({
                'episode': episode,
                'agent_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'episode_rewards': episode_rewards,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    # === FINAL SAVE ===
    final_path = os.path.join(args.log_dir, "final_agent.pt")
    torch.save({
        'agent_state_dict': agent.state_dict(),
        'episode_rewards': episode_rewards,
    }, final_path)
    
    print(f"\n{'='*60}")
    print(f"Training finished!")
    print(f"Best average reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Saved final agent to: {final_path}")
    print(f"{'='*60}")
    
    # === VIEWER AT END (optional) ===
    if args.eval_at_end and checkpoint_path:
        show_viewer(checkpoint_path, str(device), num_episodes=3)
    
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Soccer Agent with PPO - Unified Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # === TRAINING PARAMS ===
    train_group = parser.add_argument_group("Training Parameters")
    train_group.add_argument("--num-episodes", type=int, default=1000, 
                             help="Total episodes to train")
    train_group.add_argument("--episodes-per-batch", type=int, default=10, 
                             help="Episodes per PPO update batch")
    train_group.add_argument("--ppo-epochs", type=int, default=4, 
                             help="PPO epochs per batch")
    
    # === MODEL PARAMS ===
    model_group = parser.add_argument_group("Model Parameters")
    model_group.add_argument("--hidden-dim", type=int, default=256, 
                             help="Hidden layer dimension")
    model_group.add_argument("--lr", type=float, default=3e-4, 
                             help="Learning rate")
    
    # === PPO HYPERPARAMS ===
    ppo_group = parser.add_argument_group("PPO Hyperparameters")
    ppo_group.add_argument("--gamma", type=float, default=0.99, 
                           help="Discount factor")
    ppo_group.add_argument("--gae-lambda", type=float, default=0.95, 
                           help="GAE lambda")
    ppo_group.add_argument("--clip-epsilon", type=float, default=0.2, 
                           help="PPO clip epsilon")
    ppo_group.add_argument("--entropy-coef", type=float, default=0.01, 
                           help="Entropy coefficient")
    ppo_group.add_argument("--value-coef", type=float, default=0.5, 
                           help="Value loss coefficient")
    ppo_group.add_argument("--max-grad-norm", type=float, default=0.5, 
                           help="Max gradient norm")
    
    # === REWARD SHAPING ===
    reward_group = parser.add_argument_group("Reward Shaping")
    reward_group.add_argument("--no-reward-shaping", action="store_true", 
                              help="Disable reward shaping (use sparse rewards only)")
    reward_group.add_argument("--reward-scale", type=float, default=1.0, 
                              help="Scaling factor for shaped rewards")
    
    # === VISUALIZATION ===
    viewer_group = parser.add_argument_group("Visualization")
    viewer_group.add_argument("--viewer", action="store_true", default=False,
                              help="Enable periodic viewer during training")
    viewer_group.add_argument("--viewer-interval", type=int, default=50, 
                              help="Show viewer every N episodes")
    viewer_group.add_argument("--eval-at-end", action="store_true",
                              help="Show viewer at end of training (3 episodes)")
    
    # === MISC ===
    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument("--seed", type=int, default=42, 
                            help="Random seed")
    misc_group.add_argument("--log-dir", type=str, default="logs/soccer_ppo", 
                            help="Tensorboard log directory")
    misc_group.add_argument("--save-interval", type=int, default=100, 
                            help="Save checkpoint every N episodes")
    misc_group.add_argument("--log-interval", type=int, default=10, 
                            help="Log every N episodes")
    
    args = parser.parse_args()
    
    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Print configuration
    print("\n" + "="*60)
    print("CONFIGURATION")
    print("="*60)
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    train(args)
