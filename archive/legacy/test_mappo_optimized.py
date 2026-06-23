"""
MAPPO Optimized Agent testen und visualisieren.

Usage:
    python test_mappo_optimized.py                          # Neuesten Checkpoint laden
    python test_mappo_optimized.py --checkpoint path.pt     # Spezifischen Checkpoint laden
    python test_mappo_optimized.py --num-episodes 5         # Mehrere Episoden
"""
import argparse
import os
import glob
import numpy as np
import torch
from dm_control.locomotion import soccer as dm_soccer
from dm_control import viewer

from agent_mappo_optimized import MAPPOAgent, split_obs_by_agent


def flatten_obs(observation_list):
    """Flattene Observations aller Spieler."""
    flat = []
    for player_obs in observation_list:
        for key, val in player_obs.items():
            flat.append(val.flatten())
    return np.concatenate(flat).astype(np.float32)


def find_latest_checkpoint(log_dir="logs/soccer_mappo_optimized"):
    """Finde den besten Checkpoint im Log-Verzeichnis.
    
    Priorität:
    1. best_agent.pt (wenn vorhanden)
    2. Neuester checkpoint_ep*.pt
    3. checkpoint_current.pt
    4. final_agent.pt
    """
    if not os.path.exists(log_dir):
        return None
    
    # 1. Priorität: best_agent.pt
    best_path = os.path.join(log_dir, "best_agent.pt")
    if os.path.exists(best_path):
        return best_path
    
    # 2. Priorität: checkpoint_ep*.pt (neuester)
    checkpoints = glob.glob(os.path.join(log_dir, "checkpoint_ep*.pt"))
    if checkpoints:
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        return checkpoints[0]
    
    # 3. Priorität: checkpoint_current.pt
    current_path = os.path.join(log_dir, "checkpoint_current.pt")
    if os.path.exists(current_path):
        return current_path
    
    # 4. Priorität: final_agent.pt
    final_path = os.path.join(log_dir, "final_agent.pt")
    if os.path.exists(final_path):
        return final_path
    
    return None


def test(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Checkpoint finden oder laden
    if args.checkpoint is None:
        args.checkpoint = find_latest_checkpoint(args.log_dir)
        if args.checkpoint is None:
            print(f"Error: No checkpoint found in {args.log_dir}")
            print("Please train a model first or specify --checkpoint")
            return
        print(f"Found latest checkpoint: {args.checkpoint}")
    
    # Agent laden
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Agent-Parameter aus Checkpoint oder Defaults
    obs_dim_per_agent = 119
    action_dim_per_agent = 3
    num_agents = 4
    hidden_dim = 512  # Default aus Training
    
    agent = MAPPOAgent(
        obs_dim_per_agent=obs_dim_per_agent,
        action_dim_per_agent=action_dim_per_agent,
        num_agents=num_agents,
        hidden_dim=hidden_dim,
        centralized_critic=True,
        actor_layers=2,
        critic_layers=2,
        use_layer_norm=False,
    )
    
    # Versuche State Dict zu laden (manche Checkpoints haben 'agent_state_dict')
    if 'agent_state_dict' in checkpoint:
        agent.load_state_dict(checkpoint['agent_state_dict'])
        print(f"Loaded agent from episode {checkpoint.get('episode', 'unknown')}")
    else:
        agent.load_state_dict(checkpoint)
        print("Loaded agent (raw state dict)")
    
    agent.eval()
    agent.to(device)
    
    # Stats anzeigen
    if 'episode_rewards' in checkpoint:
        rewards = checkpoint['episode_rewards']
        print(f"Training rewards: min={min(rewards):.2f}, max={max(rewards):.2f}")
        if len(rewards) >= 100:
            print(f"                   avg(last 100)={np.mean(rewards[-100:]):.2f}")
        else:
            print(f"                   avg={np.mean(rewards):.2f}")
    
    # Environment
    env = dm_soccer.load(
        team_size=2,
        time_limit=10.0,
        disable_walker_contacts=False,
        enable_field_box=True,
        terminate_on_goal=False,
        walker_type=dm_soccer.WalkerType.BOXHEAD
    )
    
    episode_count = 0
    episode_rewards = []
    
    def policy(timestep):
        nonlocal episode_count, episode_rewards
        
        # Episode zählen (bei first timestep)
        if timestep.first() or (hasattr(timestep, 'step_type') and timestep.step_type == 0):
            if episode_count > 0:
                # Vorherige Episode fertig
                pass
            episode_count += 1
            print(f"\n[Episode {episode_count}/{args.num_episodes}]")
        
        obs_flat = flatten_obs(timestep.observation)
        obs_per_agent = split_obs_by_agent(obs_flat, num_agents=4, obs_dim_per_agent=119)
        
        with torch.no_grad():
            actions, log_probs, value = agent.get_actions(
                obs_per_agent, deterministic=args.deterministic
            )
        
        action_list = [a.cpu().numpy() for a in actions]
        return np.concatenate(action_list)
    
    # Run viewer
    print(f"\nStarting viewer for {args.num_episodes} episode(s)...")
    print("Close the viewer window to exit or press 'q' to skip to next episode.")
    
    try:
        viewer.launch(env, policy=policy, title="MAPPO Optimized Soccer Agent")
    except Exception as e:
        print(f"Viewer error: {e}")
        print("Make sure you have a display available.")
    
    print(f"\nCompleted {episode_count} episode(s)")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test MAPPO Optimized Agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file (default: latest in log_dir)")
    parser.add_argument("--log-dir", type=str, default="logs/soccer_mappo_optimized",
                        help="Directory to search for checkpoints")
    parser.add_argument("--num-episodes", type=int, default=1,
                        help="Number of episodes to visualize")
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Use deterministic actions (mean policy)")
    parser.add_argument("--stochastic", action="store_false", dest="deterministic",
                        help="Use stochastic actions (sample from policy)")
    
    args = parser.parse_args()
    test(args)
