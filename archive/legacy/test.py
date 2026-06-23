"""
Zentralisierten PPO Agent testen und visualisieren.
"""
import argparse
import numpy as np
import torch
from dm_control.locomotion import soccer as dm_soccer
from dm_control import viewer

from agent import ActorCritic


def flatten_obs(observation_list):
    flat = []
    for player_obs in observation_list:
        for key, val in player_obs.items():
            flat.append(val.flatten())
    return np.concatenate(flat).astype(np.float32)


def test(args):
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load agent
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    agent = ActorCritic(
        obs_dim=476,
        action_dim=12,
        hidden_dim=256
    )
    agent.load_state_dict(checkpoint['agent_state_dict'])
    agent.eval()
    agent.to(device)
    
    print(f"Loaded episode {checkpoint.get('episode', 'unknown')}")
    if 'episode_rewards' in checkpoint:
        rewards = checkpoint['episode_rewards']
        print(f"Training rewards: min={min(rewards):.2f}, max={max(rewards):.2f}, "
              f"avg(last 100)={np.mean(rewards[-100:]):.2f}")
    
    # Environment
    env = dm_soccer.load(
        team_size=2,
        time_limit=10.0,
        disable_walker_contacts=False,
        enable_field_box=True,
        terminate_on_goal=False,
        walker_type=dm_soccer.WalkerType.BOXHEAD
    )
    
    def policy(timestep):
        obs_flat = flatten_obs(timestep.observation)
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0).to(device)
            action, log_prob, value = agent.get_action(
                obs_tensor, deterministic=args.deterministic
            )
        
        return action.cpu().numpy()[0]
    
    # Run viewer
    print("\nStarting viewer...")
    print("Close the viewer window to exit.")
    viewer.launch(env, policy=policy, title="PPO Soccer Agent")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test PPO Agent")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file")
    parser.add_argument("--deterministic", action="store_true", default=True,
                        help="Use deterministic actions")
    parser.add_argument("--stochastic", action="store_false", dest="deterministic",
                        help="Use stochastic actions")
    
    args = parser.parse_args()
    test(args)
