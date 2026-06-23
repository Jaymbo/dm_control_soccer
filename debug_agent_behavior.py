"""
Debug-Skript um Agenten-Verhalten zu analysieren.

Zeigt:
  - Wer ist Ball-Chaser?
  - Wie bewegen sich die Spieler?
  - Welche Rewards werden vergeben?
  - Ballpositionen und Distanzen

Usage:
  python debug_agent_behavior.py --checkpoint <path>
  python debug_agent_behavior.py --checkpoint <path> --viewer
"""
import argparse
import numpy as np
import torch
from dm_control.locomotion import soccer as dm_soccer
from dm_control import viewer

from agent_mappo_optimized import MAPPOAgent, split_obs_by_agent
from env_wrapper_dynamic_v2 import DynamicScoringWrapperV2, make_env_with_dynamic_rewards_v2


def flatten_obs(observation_list):
    flat = []
    for player_obs in observation_list:
        for key, val in player_obs.items():
            flat.append(val.flatten())
    return np.concatenate(flat).astype(np.float32)


def analyze_observation(obs, team_size=2):
    """Analysiere eine Observation und gib detaillierte Infos."""
    num_players = len(obs)
    
    print("\n" + "="*80)
    print("OBSERVATION ANALYSIS")
    print("="*80)
    
    for p in range(num_players):
        team = "Team A" if p < team_size else "Team B"
        print(f"\n--- Player {p} ({team}) ---")
        
        # Ball position (ego)
        ball = obs[p].get('ball_ego_position')
        if ball is not None:
            ball = np.asarray(ball).flatten()
            ball_dist = np.linalg.norm(ball)
            print(f"  Ball: {ball} (dist: {ball_dist:.2f})")
        
        # Own goal (defensive)
        own_goal = obs[p].get('own_goal_mid')
        if own_goal is not None:
            own_goal = np.asarray(own_goal).flatten()
            print(f"  Own Goal: {own_goal}")
        
        # Opponent goal (offensive)
        opp_goal = obs[p].get('opponent_goal_mid')
        if opp_goal is not None:
            opp_goal = np.asarray(opp_goal).flatten()
            print(f"  Opp Goal: {opp_goal}")
        
        # Teammates
        for i in range(team_size - 1):
            key = f'teammate_{i}_ego_position'
            tm = obs[p].get(key)
            if tm is not None:
                tm = np.asarray(tm).flatten()
                print(f"  Teammate {i}: {tm}")
        
        # Opponents
        for i in range(team_size):
            key = f'opponent_{i}_ego_position'
            opp = obs[p].get(key)
            if opp is not None:
                opp = np.asarray(opp).flatten()
                print(f"  Opponent {i}: {opp}")
        
        # Velocity
        vel = obs[p].get('sensors_velocimeter')
        if vel is not None:
            vel = np.asarray(vel).flatten()
            speed = np.linalg.norm(vel)
            print(f"  Velocity: {vel:.2f} (speed: {speed:.2f})")
        
        # Body height (fallen?)
        h = obs[p].get('body_height')
        if h is not None:
            h = np.asarray(h).flatten()[0]
            fallen = "FALLEN!" if h < 0.5 else "OK"
            print(f"  Body Height: {h:.2f} ({fallen})")


def analyze_rewards(env, obs, actions, rewards, team_size=2):
    """Analysiere die Rewards nach dem Step."""
    print("\n" + "="*80)
    print("REWARD ANALYSIS")
    print("="*80)
    
    # Branch Rewards
    branch_rewards = env.get_last_branch_rewards()
    print("\nBranch Rewards (current step):")
    for branch, value in branch_rewards.items():
        if abs(value) > 0.01:  # Nur signifikante Rewards
            print(f"  {branch:15s}: {value:+8.4f}")
    
    # Total rewards per player
    print("\nTotal Rewards per Player:")
    for p, r in enumerate(rewards):
        team = "A" if p < team_size else "B"
        print(f"  Player {p} (Team {team}): {r:+8.4f}")
    
    # Ball ownership
    ball_owner = -1
    best_dist = float('inf')
    for p in range(len(obs)):
        ball = obs[p].get('ball_ego_position')
        if ball is not None:
            dist = np.linalg.norm(np.asarray(ball).flatten())
            if dist < best_dist:
                best_dist = dist
                ball_owner = p
    
    if ball_owner >= 0:
        team = "A" if ball_owner < team_size else "B"
        print(f"\nBall Owner: Player {ball_owner} (Team {team}), dist: {best_dist:.2f}")


def run_analysis(args):
    """Führe Analyse durch."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    # Create agent
    num_agents = 4
    obs_dim_per_agent = 119
    hidden_dim = checkpoint.get('hidden_dim', 512)
    
    agent = MAPPOAgent(
        obs_dim_per_agent=obs_dim_per_agent,
        action_dim_per_agent=3,
        num_agents=num_agents,
        hidden_dim=hidden_dim,
        centralized_critic=True,
        actor_layers=2,
        critic_layers=2,
        use_layer_norm=False,
    )
    
    if 'agent_state_dict' in checkpoint:
        agent.load_state_dict(checkpoint['agent_state_dict'])
    else:
        agent.load_state_dict(checkpoint)
    
    agent.eval()
    agent.to(device)
    print(f"Agent loaded (hidden_dim={hidden_dim})")
    
    # Create environment
    env = make_env_with_dynamic_rewards_v2(
        team_size=args.team_size,
        time_limit=args.time_limit,
        reward_scale=0.1,
        lambda_recovery=args.lambda_recovery,
        lambda_marking=args.lambda_marking,
        lambda_possession=args.lambda_possession,
        lambda_shooting=args.lambda_shooting,
        lambda_blocking=args.lambda_blocking,
        lambda_goalkeeping=args.lambda_goalkeeping,
        lambda_attack_pos=args.lambda_attack_pos,
    )
    print(f"Environment created: {args.team_size}v{args.team_size}, {args.time_limit}s")
    
    # Run episodes
    for ep in range(args.num_episodes):
        print(f"\n{'='*80}")
        print(f"EPISODE {ep+1}/{args.num_episodes}")
        print('='*80)
        
        timestep = env.reset()
        obs = timestep.observation
        
        # Initial analysis
        if args.analyze_obs:
            analyze_observation(obs, args.team_size)
        
        step_count = 0
        ep_rewards = []
        
        while not timestep.last():
            step_count += 1
            
            # Get actions
            obs_flat = flatten_obs(obs)
            obs_per_agent = split_obs_by_agent(obs_flat, num_agents, obs_dim_per_agent)
            obs_tensor = torch.FloatTensor(np.stack(obs_per_agent)).unsqueeze(0).to(device)
            
            with torch.no_grad():
                actions, log_probs, value = agent.get_actions(obs_tensor, deterministic=True)
            
            actions_np = actions.squeeze(0).cpu().numpy()
            
            # Step
            timestep = env.step(actions_np.flatten())
            rewards = timestep.reward
            obs = timestep.observation
            
            ep_rewards.append(rewards)
            
            # Analyze first few steps or when goals scored
            if step_count <= args.analyze_steps or timestep.last():
                print(f"\n--- Step {step_count} ---")
                analyze_rewards(env, obs, actions_np, rewards, args.team_size)
                
                # Show ball position
                for p in range(num_agents):
                    ball = obs[p].get('ball_ego_position')
                    if ball is not None:
                        ball = np.asarray(ball).flatten()
                        dist = np.linalg.norm(ball)
                        if dist < 1.0:  # Very close to ball
                            print(f"  Player {p} is VERY close to ball (dist={dist:.2f})")
            
            # Early exit if needed
            if step_count >= args.max_steps:
                break
        
        # Episode summary
        total_reward = np.sum(ep_rewards)
        avg_reward = np.mean([np.sum(r) for r in ep_rewards])
        print(f"\nEpisode {ep+1} Summary:")
        print(f"  Steps: {step_count}")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Avg Reward/Step: {avg_reward:.4f}")
        
        # Branch summary
        branch_totals = env.get_branch_rewards(reset=True)
        print(f"  Branch Rewards:")
        for branch, value in branch_totals.items():
            print(f"    {branch:15s}: {value:+8.2f}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


def run_viewer(args):
    """Run with dm_control viewer for visual inspection."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    num_agents = 4
    obs_dim_per_agent = 119
    hidden_dim = checkpoint.get('hidden_dim', 512)
    
    agent = MAPPOAgent(
        obs_dim_per_agent=obs_dim_per_agent,
        action_dim_per_agent=3,
        num_agents=num_agents,
        hidden_dim=hidden_dim,
        centralized_critic=True,
    )
    
    if 'agent_state_dict' in checkpoint:
        agent.load_state_dict(checkpoint['agent_state_dict'])
    else:
        agent.load_state_dict(checkpoint)
    
    agent.eval()
    
    env = make_env_with_dynamic_rewards_v2(
        team_size=args.team_size,
        time_limit=args.time_limit,
    )
    
    def policy(timestep):
        obs_flat = flatten_obs(timestep.observation)
        obs_per_agent = split_obs_by_agent(obs_flat, num_agents, obs_dim_per_agent)
        obs_tensor = torch.FloatTensor(np.stack(obs_per_agent)).unsqueeze(0)
        
        with torch.no_grad():
            actions, _, _ = agent.get_actions(obs_tensor, deterministic=True)
        
        return actions.squeeze(0).cpu().numpy().flatten()
    
    print("\nStarting viewer...")
    print("Watch for:")
    print("  - Do players move towards the ball?")
    print("  - Does the ball-chaser chase the ball?")
    print("  - Do other players mark opponents or position for attack?")
    print("  - Does the ball owner try to score?")
    print("  - Do players fall down frequently?")
    print("\nPress 'q' to skip episode, close window to exit.")
    
    viewer.launch(env, policy=policy, title="Agent Behavior Analysis")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Debug Agent Behavior",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file")
    parser.add_argument("--team-size", type=int, default=2,
                        help="Team size (2 = 2v2)")
    parser.add_argument("--time-limit", type=float, default=30.0,
                        help="Time limit per episode")
    parser.add_argument("--num-episodes", type=int, default=3,
                        help="Number of episodes to analyze")
    parser.add_argument("--max-steps", type=int, default=600,
                        help="Maximum steps per episode")
    parser.add_argument("--analyze-steps", type=int, default=10,
                        help="Analyze first N steps in detail")
    parser.add_argument("--analyze-obs", action="store_true",
                        help="Analyze initial observation in detail")
    parser.add_argument("--viewer", action="store_true",
                        help="Run with dm_control viewer instead of text analysis")
    
    # Reward weights for analysis
    parser.add_argument("--lambda-recovery", type=float, default=1.0)
    parser.add_argument("--lambda-marking", type=float, default=1.0)
    parser.add_argument("--lambda-possession", type=float, default=1.0)
    parser.add_argument("--lambda-shooting", type=float, default=1.0)
    parser.add_argument("--lambda-blocking", type=float, default=1.0)
    parser.add_argument("--lambda-goalkeeping", type=float, default=1.0)
    parser.add_argument("--lambda-attack-pos", type=float, default=0.5)
    
    args = parser.parse_args()
    
    if args.viewer:
        run_viewer(args)
    else:
        run_analysis(args)
