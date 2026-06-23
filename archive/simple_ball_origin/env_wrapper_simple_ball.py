"""
Simple Ball-Chase Reward Wrapper V9.

Vollständig symmetrischer, positiver Reward für 2v2 Soccer.
Jedes Team wird aus EGO-PERSPEKTIVE belohnt – es gibt keinen
„Gewinner“-Bias im Shaped Reward. Ziel: beide Teams lernen
gleichermaßen Ball-Anlaufen, Annäherung und Besitz.
"""
import numpy as np
from dm_control.locomotion import soccer as dm_soccer


class SimpleBallChaseWrapper:
    """
    Ego-zentrischer, symmetrischer Reward für Soccer.
    Kein Team-bias: beide Teams erhalten identische Reward-Komponenten
    basierend auf ihrer eigenen Leistung.
    """

    BRANCH_NAMES = ["ball_chase"]

    def __init__(
        self,
        env,
        reward_scale=1.0,
        possession_weight=0.02,       # NICHT MEHR GENUTZT (nur für API-Kompatibilität)
        proximity_weight=0.1,         # Reward für Ball-Nähe (max 0.1 pro Step)
        time_penalty=0.0001,          # Sehr kleine Strafe pro Step
    ):
        self.env = env
        self.reward_scale = reward_scale
        self.possession_weight = possession_weight
        self.proximity_weight = proximity_weight
        self.time_penalty = time_penalty

        self._num_players = None
        self._team_size = None
        self._prev_dist_to_ball = None
        self._episode_reward = 0.0

    def _detect_team_size(self, obs):
        num_players = len(obs)
        if num_players % 2 != 0:
            raise ValueError(f"Ungerade Spieleranzahl: {num_players}")
        return num_players // 2, num_players

    def _get(self, obs, player_idx, key):
        if player_idx >= len(obs):
            return None
        val = obs[player_idx].get(key)
        if val is None:
            return None
        arr = np.asarray(val)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        return arr

    def _norm(self, vec):
        if vec is None:
            return 0.0
        return float(np.linalg.norm(vec))

    def _distance_to_ball(self, obs, player_idx):
        ball_pos = self._get(obs, player_idx, "ball_ego_position")
        return self._norm(ball_pos)

    def _velocity(self, obs, player_idx):
        v = self._get(obs, player_idx, "sensors_velocimeter")
        if v is None:
            return np.zeros(3, dtype=np.float32)
        return np.asarray(v, dtype=np.float32).flatten()[:3]

    def _find_ball_chaser(self, obs, team):
        """Index des eigenen Spielers, der dem Ball am nächsten ist."""
        players = list(range(team * self._team_size, (team + 1) * self._team_size))
        dists = {p: self._distance_to_ball(obs, p) for p in players}
        return min(dists, key=dists.get)

    def _find_closest_player_to_ball(self, obs):
        """Findet den absolut nächsten Spieler zum Ball (global)."""
        best_dist = float("inf")
        best_player = -1
        for p in range(self._num_players):
            dist = self._distance_to_ball(obs, p)
            if dist < best_dist:
                best_dist = dist
                best_player = p
        return best_player, best_dist

    def _get_opponent_chaser_dist(self, obs, team):
        """Distanz des nächsten Gegners zum Ball."""
        opp_start = self._team_size if team == 0 else 0
        opp_end = opp_start + self._team_size
        best_dist = float("inf")
        for p in range(opp_start, opp_end):
            dist = self._distance_to_ball(obs, p)
            if dist < best_dist:
                best_dist = dist
        return best_dist

    def _compute_reward(self, obs):
        """
        Ultra-simpler Reward: jeder Agent wird NUR nach eigener Ballnähe belohnt.

        reward_p = proximity_weight * max(0, 1 - dist_p / max_dist) - time_penalty

        - Kein Team-Bonus, kein Gegner, kein Chaser, kein Delta, kein Besitz.
        - Vollständig symmetrisch: alle 4 Agents werden identisch behandelt.
        - Sofort erkennbar, ob die Policy lernt, zum Ball zu laufen.
        """
        if self._num_players is None:
            self._team_size, self._num_players = self._detect_team_size(obs)

        rewards = np.zeros(self._num_players, dtype=np.float32)
        max_dist = 5.0  # ab 5m Distanz gibt es keinen Proximity-Bonus mehr

        for p in range(self._num_players):
            dist = self._distance_to_ball(obs, p)
            proximity = max(0.0, 1.0 - dist / max_dist)
            reward = self.proximity_weight * proximity - self.time_penalty
            rewards[p] = reward

        return rewards

    def reset(self):
        self._prev_dist_to_ball = None
        self._episode_reward = 0.0

        timestep = self.env.reset()
        self._team_size, self._num_players = self._detect_team_size(timestep.observation)
        _ = self._compute_reward(timestep.observation)  # Init prev_dist
        return timestep

    def step(self, actions):
        timestep = self.env.step(actions)
        shaped = self._compute_reward(timestep.observation)
        
        # Speichern für nächsten Step
        for p in range(self._num_players):
            if self._prev_dist_to_ball is None:
                self._prev_dist_to_ball = {}
            self._prev_dist_to_ball[p] = self._distance_to_ball(timestep.observation, p)
        
        # Base + Shaped
        base_rewards = np.asarray(timestep.reward, dtype=np.float32)
        combined = base_rewards + shaped * self.reward_scale
        combined = combined.astype(np.float32)
        
        # Episode-Tracking
        self._episode_reward += float(np.sum(combined))

        from dm_env import TimeStep
        return TimeStep(
            step_type=timestep.step_type,
            reward=tuple(combined.tolist()),
            discount=timestep.discount,
            observation=timestep.observation,
        )

    def __getattr__(self, name):
        return getattr(self.env, name)

    def get_branch_rewards(self, reset=True):
        stats = {"ball_chase": self._episode_reward}
        if reset:
            self._episode_reward = 0.0
        return stats

    def get_last_branch_rewards(self):
        return {"ball_chase": self._episode_reward}


def make_env_with_simple_ball_chase(
    team_size=2,
    seed=None,
    reward_scale=1.0,
    possession_weight=0.02,
    proximity_weight=0.01,
    time_penalty=0.0001,
    time_limit=10.0,
):
    """
    Erstellt Soccer-Environment mit Simple Ball-Chase Reward V9.

    Positive, symmetrische Rewards für Soccer-Verhalten:
    - Team-Nähe-Bonus: jeder bekommt Reward für eigenen Chaser-Abstand
    - Delta Reward: +1.0 pro Meter Annäherung
    - Chaser Bonus: team-interner Chaser bekommt +0.5 pro Step
    - Proximity Bonus: sehr klein

    Erwartetes Verhalten:
    - Beide Teams laufen gleichermaßen zum Ball
    - Kein Team-Bias durch globale Besitz-Definition
    - Kein negativer Reward (nur positive Anreize!)
    
    Parameter:
        team_size: Spieler pro Team (2 = 2v2)
        seed: Random Seed
        reward_scale: Skalierung des shaped Rewards
        possession_weight: Bonus wenn Team näher ist (default: 0.02 → 0.1 nach Skalierung)
        proximity_weight: Bonus für Ball-Nähe (default: 0.01)
        time_penalty: Kleine Strafe pro Step (default: 0.0001)
        time_limit: Episoden-Dauer in Sekunden
    """
    env = dm_soccer.load(
        team_size=team_size,
        time_limit=time_limit,
        disable_walker_contacts=False,
        enable_field_box=True,
        terminate_on_goal=False,
        walker_type=dm_soccer.WalkerType.BOXHEAD,
    )
    if seed is not None:
        env.task._random_state = np.random.RandomState(seed)
    
    return SimpleBallChaseWrapper(
        env,
        reward_scale=reward_scale,
        possession_weight=possession_weight,
        proximity_weight=proximity_weight,
        time_penalty=time_penalty,
    )
