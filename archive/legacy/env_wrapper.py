"""
Custom Environment Wrapper mit Reward Shaping für Soccer.
Fügt dichte Rewards hinzu für:
- Annäherung an den Ball
- Annäherung an das gegnerische Tor
- Ballbewegung in Richtung Tor
"""
import numpy as np
from dm_control.locomotion import soccer as dm_soccer


class SoccerRewardWrapper:
    """
    Wrapper um das Soccer-Environment mit verbessertem Reward Shaping.
    
    Reward-Komponenten:
    1. Tor-Reward: +10 für eigenes Tor, -10 für Gegentor
    2. Ball-Reward: +0.01 * (delta_distance_to_ball) - belohnt Annäherung
    3. Tor-Reward: +0.01 * (delta_distance_ball_to_goal) - belohnt Ball zum Tor
    4. Teamwork-Reward: +0.001 wenn Teammate näher am Ball ist (vermeidet Doppelungen)
    """
    
    def __init__(self, env, reward_scale=1.0):
        self.env = env
        self.reward_scale = reward_scale
        
        # Speichert vorherige Distanzen für Delta-Berechnung
        self._prev_ball_dist = None
        self._prev_ball_to_goal = None
        
        # Team-Zuordnung: Spieler 0,1 = Home Team, Spieler 2,3 = Away Team
        self._team_size = 2
    
    def _get_distance_to_ball(self, obs, player_idx):
        """Extrahiert Distanz zum Ball aus Observation."""
        ball_pos = obs[player_idx]['ball_ego_position'][0]  # (x, y, z)
        return np.sqrt(np.sum(ball_pos**2))
    
    def _get_ball_distance_to_goal(self, obs, player_idx):
        """
        Schätzt Distanz des Balls zum gegnerischen Tor.
        Verwendet Feld-Positionen aus Observation.
        """
        # Ball-Position in ego-Koordinaten
        ball_pos = obs[player_idx]['ball_ego_position'][0]
        
        # Gegner-Tor-Mitte (approximativ in ego-Koordinaten)
        opponent_goal_mid = obs[player_idx]['opponent_goal_mid'][0]
        
        # Distanz vom Ball zum gegnerischen Tor
        dx = ball_pos[0] - opponent_goal_mid[0]
        dy = ball_pos[1] - opponent_goal_mid[1]
        return np.sqrt(dx**2 + dy**2)
    
    def _compute_shaped_reward(self, obs, timestep):
        """Berechnet shaped reward basierend auf aktuellen Observations."""
        shaped_rewards = np.zeros(self._team_size * 2)
        
        # Home Team (Spieler 0, 1)
        home_ball_dist_old = self._prev_ball_dist[0] if self._prev_ball_dist is not None else None
        home_ball_to_goal_old = self._prev_ball_to_goal[0] if self._prev_ball_to_goal is not None else None
        
        for p_idx in range(self._team_size):
            # Aktuelle Distanzen
            ball_dist = self._get_distance_to_ball(obs, p_idx)
            ball_to_goal = self._get_ball_distance_to_goal(obs, p_idx)
            
            # Reward für Annäherung an Ball (stärker gewichtet)
            if home_ball_dist_old is not None:
                delta_ball = home_ball_dist_old - ball_dist
                shaped_rewards[p_idx] += 0.1 * np.clip(delta_ball, -1, 1)
            
            # Reward für Ballbewegung zum gegnerischen Tor (stärker gewichtet)
            if home_ball_to_goal_old is not None:
                delta_goal = home_ball_to_goal_old - ball_to_goal
                shaped_rewards[p_idx] += 0.2 * np.clip(delta_goal, -1, 1)
        
        # Away Team (Spieler 2, 3)
        away_ball_dist_old = self._prev_ball_dist[1] if self._prev_ball_dist is not None else None
        away_ball_to_goal_old = self._prev_ball_to_goal[1] if self._prev_ball_to_goal is not None else None
        
        for p_idx in range(self._team_size, self._team_size * 2):
            # Aktuelle Distanzen
            ball_dist = self._get_distance_to_ball(obs, p_idx)
            ball_to_goal = self._get_ball_distance_to_goal(obs, p_idx)
            
            # Reward für Annäherung an Ball
            if away_ball_dist_old is not None:
                delta_ball = away_ball_dist_old - ball_dist
                shaped_rewards[p_idx] += 0.1 * np.clip(delta_ball, -1, 1)
            
            # Reward für Ballbewegung zum gegnerischen Tor
            if away_ball_to_goal_old is not None:
                delta_goal = away_ball_to_goal_old - ball_to_goal
                shaped_rewards[p_idx] += 0.2 * np.clip(delta_goal, -1, 1)
        
        # Update previous values
        self._prev_ball_dist = [
            self._get_distance_to_ball(obs, 0),
            self._get_distance_to_ball(obs, self._team_size)
        ]
        self._prev_ball_to_goal = [
            self._get_ball_distance_to_goal(obs, 0),
            self._get_ball_distance_to_goal(obs, self._team_size)
        ]
        
        return shaped_rewards
    
    def reset(self):
        """Reset Environment und speichert initiale Distanzen."""
        timestep = self.env.reset()
        
        # Initiale Distanzen speichern
        self._prev_ball_dist = [
            self._get_distance_to_ball(timestep.observation, 0),
            self._get_distance_to_ball(timestep.observation, self._team_size)
        ]
        self._prev_ball_to_goal = [
            self._get_ball_distance_to_goal(timestep.observation, 0),
            self._get_ball_distance_to_goal(timestep.observation, self._team_size)
        ]
        
        return timestep
    
    def step(self, actions):
        """Environment step mit shaped reward."""
        timestep = self.env.step(actions)
        
        # Base Reward (Tore: +1/-1)
        base_rewards = timestep.reward
        
        # Shaped Reward berechnen
        shaped_rewards = self._compute_shaped_reward(timestep.observation, timestep)
        
        # Kombinieren: base + shaped
        combined_rewards = []
        for i in range(len(base_rewards)):
            combined = base_rewards[i] + shaped_rewards[i] * self.reward_scale
            combined_rewards.append(combined)
        
        # Neues Timestep mit modifiziertem Reward erstellen
        from dm_env import TimeStep
        new_timestep = TimeStep(
            step_type=timestep.step_type,
            reward=combined_rewards,
            discount=timestep.discount,
            observation=timestep.observation
        )
        
        return new_timestep
    
    def __getattr__(self, name):
        """Delegiere alle anderen Attribute an das Environment."""
        return getattr(self.env, name)


def make_env_with_rewards(seed=None, reward_scale=1.0):
    """
    Erstellt ein Soccer-Environment mit Reward Shaping.
    
    Args:
        seed: Random seed
        reward_scale: Skalierungsfaktor für shaped rewards (1.0 = default)
    
    Returns:
        Environment mit Reward Shaping
    """
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
    
    return SoccerRewardWrapper(env, reward_scale=reward_scale)
