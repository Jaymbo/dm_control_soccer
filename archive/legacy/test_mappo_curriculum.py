"""
Test-Skript für MAPPO Curriculum-Modelle.

Lädt ein trainiertes Modell und zeigt Performance-Statistiken oder startet den Viewer.
"""
import os
import argparse
import numpy as np
import torch
from dm_control.locomotion import soccer as dm_soccer
from dm_control import viewer

from agent_mappo_optimized import MAPPOAgent, split_obs_by_agent
from env_wrapper_curriculum import SoccerCurriculumWrapper


def flatten_obs(observation_list):
    flat = []
    for player_obs in observation_list:
        for key, val in player_obs.items():
            flat.append(val.flatten())
    return np.concatenate(flat).astype(np.float32)


def load_agent(checkpoint_path, device):
    """Lädt Agent aus Checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    agent = MAPPOAgent(
        obs_dim_per_agent=119,
        action_dim_per_agent=3,
        num_agents=4,
        hidden_dim=512,
        centralized_critic=True,
        actor_layers=2,
        critic_layers=2,
    ).to(device)
    
    agent.load_state_dict(checkpoint['agent_state_dict'])
    agent.eval()
    
    phase = checkpoint.get('curriculum_phase', checkpoint.get('phase', 3))
    episode = checkpoint.get('episode', 0)
    
    print(f"Loaded: Episode {episode}, Curriculum Phase {phase}")
    return agent, phase


def evaluate_agent(agent, device, num_episodes=20, seed=None):
    """Evaluiert Agent über mehrere Episoden."""
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
    
    episode_rewards = []
    total_steps = 0
    goals_scored = 0
    
    for ep in range(num_episodes):
        obs = env.reset()
        obs_flat = flatten_obs(obs.observation)
        ep_reward = 0.0
        steps = 0
        prev_ball_to_goal = None
        
        while not obs.last():
            obs_per_agent = split_obs_by_agent(obs_flat, num_agents=4, obs_dim_per_agent=119)
            obs_tensor = torch.FloatTensor(np.stack(obs_per_agent, axis=0)).unsqueeze(0).to(device)
            
            with torch.no_grad():
                actions, _, _ = agent.get_actions(obs_tensor, deterministic=True)
            
            actions_np = actions.squeeze(0).cpu().numpy()
            obs = env.step(actions_np.flatten())
            
            rewards = obs.reward
            ep_reward += float(np.sum(rewards))
            steps += 1
            
            # Torerkennung (großer Reward-Spike)
            if prev_ball_to_goal is not None:
                ball_pos = obs.observation[0].get('ball_ego_position')
                goal_pos = obs.observation[0].get('opponent_goal_mid')
                if ball_pos is not None and goal_pos is not None:
                    ball_pos = np.asarray(ball_pos)[0]
                    goal_pos = np.asarray(goal_pos)[0]
                    curr_dist = np.linalg.norm(ball_pos - goal_pos)
                    if curr_dist < 0.5 and prev_ball_to_goal > 2.0:
                        goals_scored += 1
                    prev_ball_to_goal = curr_dist
            else:
                ball_pos = obs.observation[0].get('ball_ego_position')
                goal_pos = obs.observation[0].get('opponent_goal_mid')
                if ball_pos is not None and goal_pos is not None:
                    ball_pos = np.asarray(ball_pos)[0]
                    goal_pos = np.asarray(goal_pos)[0]
                    prev_ball_to_goal = np.linalg.norm(ball_pos - goal_pos)
            
            obs_flat = flatten_obs(obs.observation)
        
        episode_rewards.append(ep_reward)
        total_steps += steps
    
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    avg_steps = total_steps / num_episodes
    
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS ({num_episodes} episodes)")
    print(f"{'='*60}")
    print(f"Avg Reward:     {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"Avg Steps:      {avg_steps:.1f}")
    print(f"Goals Scored:   {goals_scored} ({goals_scored/num_episodes*100:.1f}% of episodes)")
    print(f"{'='*60}\n")
    
    return avg_reward, std_reward, goals_scored


def launch_viewer(agent, device, seed=None):
    """Startet dm_control Viewer mit Agent."""
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
    
    def policy(timestep):
        obs_flat = flatten_obs(timestep.observation)
        obs_per_agent = split_obs_by_agent(obs_flat, num_agents=4, obs_dim_per_agent=119)
        with torch.no_grad():
            actions, _, _ = agent.get_actions(
                torch.FloatTensor(np.stack(obs_per_agent, axis=0)).unsqueeze(0).to(device),
                deterministic=True
            )
        return np.concatenate([a.cpu().numpy() for a in actions])
    
    print("\nLaunching viewer... Close window to exit.\n")
    viewer.launch(env, policy=policy, title="MAPPO Curriculum Soccer Agent")


def main():
    parser = argparse.ArgumentParser(description="Test MAPPO Curriculum Agent")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Pfad zum Checkpoint (.pt Datei)")
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="Anzahl Episoden für Evaluation")
    parser.add_argument("--viewer", action="store_true",
                        help="Viewer starten nach Evaluation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    agent, phase = load_agent(args.checkpoint, device)
    
    evaluate_agent(agent, device, num_episodes=args.eval_episodes, seed=args.seed)
    
    if args.viewer:
        launch_viewer(agent, device, seed=args.seed)


if __name__ == "__main__":
    main()
