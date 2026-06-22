"""
MAPPO (Multi-Agent PPO) Training für DM Control Soccer.

Centralized Training with Decentralized Execution (CTDE):
- Jeder Spieler hat eigene Policy (Actor)
- Critic nutzt globale Informationen (alle Observations)
- Shared Weights für alle Spieler

Vorteile:
- Bessere Skalierbarkeit
- Individuelles Lernen pro Spieler
- Robuster bei partiellen Observations
"""
import os
import argparse
import subprocess
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dm_control.locomotion import soccer as dm_soccer

from agent_mappo import (
    MAPPOAgent, MAPPOReplayBuffer, compute_gae, 
    split_obs_by_agent
)
from env_wrapper import make_env_with_rewards


def flatten_obs(observation_list):
    """Flattene Observations von allen Spielern."""
    flat = []
    for player_obs in observation_list:
        for key, val in player_obs.items():
            flat.append(val.flatten())
    return np.concatenate(flat).astype(np.float32)


def make_env(seed=None, use_reward_shaping=True, reward_scale=1.0):
    """Environment erstellen."""
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


def show_viewer(checkpoint_path, device):
    """Viewer für Visualisierung öffnen."""
    print(f"\n{'='*60}")
    print(f"VISUALIZATION - Checkpoint: {checkpoint_path}")
    print(f"Viewer öffnet sich...")
    print(f"{'='*60}\n")
    
    viewer_script = f'''
import torch
import numpy as np
from dm_control.locomotion import soccer as dm_soccer
from dm_control import viewer
from agent_mappo import MAPPOAgent, split_obs_by_agent
from env_wrapper import make_env_with_rewards

def flatten_obs(obs):
    flat = []
    for player_obs in obs:
        for key, val in player_obs.items():
            flat.append(val.flatten())
    return np.concatenate(flat).astype(np.float32)

# Load agent
checkpoint = torch.load("{checkpoint_path}", map_location="{device}", weights_only=False)
agent = MAPPOAgent(obs_dim_per_agent=119, action_dim_per_agent=3, num_agents=4)
agent.load_state_dict(checkpoint['agent_state_dict'])
agent.eval()

# Environment
env = dm_soccer.load(
    team_size=2, time_limit=10.0, disable_walker_contacts=False,
    enable_field_box=True, terminate_on_goal=False,
    walker_type=dm_soccer.WalkerType.BOXHEAD
)

def policy(timestep):
    obs_flat = flatten_obs(timestep.observation)
    obs_per_agent = split_obs_by_agent(obs_flat, num_agents=4, obs_dim_per_agent=119)
    
    with torch.no_grad():
        actions, log_probs, value = agent.get_actions(obs_per_agent, deterministic=True)
    
    # Concatenate actions for all agents
    action_list = [a.cpu().numpy() for a in actions]
    return np.concatenate(action_list)

viewer.launch(env, policy=policy, title="MAPPO Soccer Agent")
'''
    
    with open('/tmp/soccer_mappo_viewer.py', 'w') as f:
        f.write(viewer_script)
    
    try:
        subprocess.run(
            ['python', '/tmp/soccer_mappo_viewer.py'],
            timeout=60,
            cwd=os.getcwd()
        )
    except subprocess.TimeoutExpired:
        print("[Viewer] Timeout - closing...")
    except Exception as e:
        print(f"[Viewer] Error: {e}")
    finally:
        if os.path.exists('/tmp/soccer_mappo_viewer.py'):
            os.remove('/tmp/soccer_mappo_viewer.py')
    
    print(f"[Viewer] Closed\n")


def train(args):
    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if torch.version.hip is not None:
            print("Using AMD ROCm GPU")
        else:
            print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Environment
    use_shaping = not args.no_reward_shaping
    env = make_env(seed=args.seed, use_reward_shaping=use_shaping, reward_scale=args.reward_scale)
    print(f"Reward Shaping: {use_shaping} (scale={args.reward_scale})")
    print(f"Critic Type: {'Centralized' if args.centralized_critic else 'Decentralized'}")
    
    # Hyperparameter
    num_agents = 4
    obs_dim_per_agent = 119  # DM Control Soccer BoxHead
    action_dim_per_agent = 3
    
    # Agent, Buffer, Optimizer
    agent = MAPPOAgent(
        obs_dim_per_agent=obs_dim_per_agent,
        action_dim_per_agent=action_dim_per_agent,
        num_agents=num_agents,
        hidden_dim=args.hidden_dim,
        centralized_critic=args.centralized_critic
    ).to(device)
    
    optimizer = optim.Adam(agent.parameters(), lr=args.lr)
    buffer = MAPPOReplayBuffer(max_size=args.episodes_per_batch * 1000, num_agents=num_agents)
    
    # Tensorboard
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Stats
    episode_rewards = []
    
    print(f"\nStarting MAPPO training for {args.num_episodes} episodes...")
    print(f"Episodes per batch: {args.episodes_per_batch}")
    print(f"PPO epochs: {args.ppo_epochs}")
    if args.viewer:
        print(f"Viewer: Every {args.viewer_interval} episodes")
    print("-" * 60)
    
    episode = 0
    
    while episode < args.num_episodes:
        buffer.reset()
        
        # === COLLECT TRAJECTORIES ===
        for _ in range(args.episodes_per_batch):
            obs = env.reset()
            obs_flat = flatten_obs(obs.observation)
            
            episode_reward = 0
            steps = 0
            
            while not obs.last():
                # Split observation per agent
                obs_per_agent = split_obs_by_agent(obs_flat, num_agents, obs_dim_per_agent)
                
                # Get actions from MAPPO agent
                with torch.no_grad():
                    actions_list, log_probs_list, value = agent.get_actions(
                        obs_per_agent, deterministic=False
                    )
                
                # Convert to numpy
                actions_np = np.concatenate([a.cpu().numpy() for a in actions_list])
                log_probs_np = np.array([lp.cpu().numpy()[0] for lp in log_probs_list])
                value_np = value.cpu().numpy()[0]
                
                # Environment step
                obs = env.step(actions_np)
                rewards = obs.reward  # List of rewards per agent
                done = obs.last()
                
                episode_reward += np.sum(rewards)
                steps += 1
                
                # Store in buffer
                buffer.add(
                    observations=obs_per_agent,
                    actions=actions_list,
                    rewards=rewards,
                    dones=[done] * num_agents,
                    log_probs=log_probs_list,
                    value=value_np
                )
                
                obs_flat = flatten_obs(obs.observation)
            
            episode_rewards.append(episode_reward)
            episode += 1
            
            if episode % args.log_interval == 0:
                avg_reward = np.mean(episode_rewards[-args.log_interval:])
                print(f"Episode {episode}/{args.num_episodes} | "
                      f"Avg Reward: {avg_reward:.2f} | "
                      f"Steps: {steps}")
                writer.add_scalar("Reward/episode", episode_reward, episode)
                writer.add_scalar("Reward/avg_100", np.mean(episode_rewards[-100:]), episode)
        
        # === MAPPO UPDATE ===
        obs_batch, actions_batch, rewards_batch, dones_batch, old_log_probs_batch, values_batch = buffer.get_batch()
        
        # Compute GAE (per timestep, summed across agents)
        T = len(rewards_batch)  # Timesteps
        
        # Sum rewards across agents per timestep
        rewards_summed = rewards_batch.sum(axis=1)  # (T,)
        dones_any = dones_batch.any(axis=1)  # (T,)
        values_flat = values_batch  # (T,)
        
        advantages, returns = compute_gae(
            rewards_summed,
            values_flat,
            dones_any,
            gamma=args.gamma,
            lambda_=args.gae_lambda
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        # obs_batch: (T, num_agents, obs_dim)
        obs_batch = torch.FloatTensor(obs_batch).to(device)
        actions_batch = torch.FloatTensor(actions_batch).to(device)
        old_log_probs_batch = torch.FloatTensor(old_log_probs_batch).to(device)
        advantages = advantages.to(device)
        returns = returns.to(device)
        
        # PPO Epochs
        for ppo_epoch in range(args.ppo_epochs):
            # Evaluate current policy
            # obs_batch: (T, num_agents, obs_dim)
            # Need to evaluate for each timestep
            
            total_policy_loss = 0
            total_value_loss = 0
            total_entropy = 0
            num_updates = 0
            
            for t in range(T):
                # Get observations and actions at timestep t
                obs_t = obs_batch[t]  # (num_agents, obs_dim)
                actions_t = actions_batch[t]  # (num_agents, action_dim)
                old_log_probs_t = old_log_probs_batch[t]  # (num_agents, 1)
                
                # Evaluate
                log_probs_t, value_t, entropy_t = agent.evaluate_actions(obs_t, actions_t)
                
                # Compute ratio
                ratio = torch.exp(torch.cat(log_probs_t) - old_log_probs_t)
                
                # PPO loss
                adv_t = advantages[t].unsqueeze(0).expand(num_agents, -1)
                surr1 = ratio * adv_t
                surr2 = torch.clamp(ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon) * adv_t
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss - ensure matching dimensions
                # value_t: (num_agents, 1), returns[t]: scalar
                ret_t = returns[t].expand(num_agents)  # (num_agents,)
                value_loss = F.mse_loss(value_t.squeeze(), ret_t)
                
                # Entropy
                entropy_loss = -args.entropy_coef * torch.cat(entropy_t).mean()
                
                total_policy_loss += policy_loss
                total_value_loss += args.value_coef * value_loss
                total_entropy += entropy_loss
                num_updates += 1
            
            # Average losses
            policy_loss = total_policy_loss / num_updates
            value_loss = total_value_loss / num_updates
            entropy_loss = total_entropy / num_updates
            loss = policy_loss + value_loss + entropy_loss
            
            # Update
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()
        
        # Log
        writer.add_scalar("Loss/policy", policy_loss.item(), episode)
        writer.add_scalar("Loss/value", value_loss.item(), episode)
        writer.add_scalar("Loss/entropy", entropy_loss.item(), episode)
        
        # === VIEWER ===
        if args.viewer and episode > 0 and episode % args.viewer_interval == 0:
            checkpoint_path = os.path.join(args.log_dir, "checkpoint_current.pt")
            torch.save({
                'episode': episode,
                'agent_state_dict': agent.state_dict(),
                'episode_rewards': episode_rewards,
            }, checkpoint_path)
            show_viewer(checkpoint_path, str(device))
        
        # === CHECKPOINT ===
        if episode % args.save_interval == 0 and episode > 0:
            checkpoint_path = os.path.join(args.log_dir, f"checkpoint_ep{episode}.pt")
            torch.save({
                'episode': episode,
                'agent_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'episode_rewards': episode_rewards,
            }, checkpoint_path)
            print(f"Saved: {checkpoint_path}")
    
    # Final save
    final_path = os.path.join(args.log_dir, "final_agent.pt")
    torch.save({
        'agent_state_dict': agent.state_dict(),
        'episode_rewards': episode_rewards,
    }, final_path)
    
    print(f"\n{'='*60}")
    print(f"MAPPO Training finished!")
    print(f"Best avg reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Saved to: {final_path}")
    print(f"{'='*60}")
    
    if args.eval_at_end and 'checkpoint_path' in locals():
        show_viewer(checkpoint_path, str(device))
    
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MAPPO Training for Soccer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Training
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--episodes-per-batch", type=int, default=10)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    
    # Model
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    
    # PPO
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    
    # Critic
    parser.add_argument("--centralized-critic", action="store_true", default=True,
                        help="Use centralized critic (CTDE)")
    parser.add_argument("--decentralized-critic", action="store_false", dest="centralized_critic",
                        help="Use decentralized critic")
    
    # Reward
    parser.add_argument("--no-reward-shaping", action="store_true")
    parser.add_argument("--reward-scale", type=float, default=1.0)
    
    # Viewer
    parser.add_argument("--viewer", action="store_true", default=False)
    parser.add_argument("--viewer-interval", type=int, default=50)
    parser.add_argument("--eval-at-end", action="store_true")
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str, default="logs/soccer_mappo")
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=10)
    
    args = parser.parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Print config
    print("\n" + "="*60)
    print("MAPPO CONFIGURATION")
    print("="*60)
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")
    
    train(args)
