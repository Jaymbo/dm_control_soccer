"""
Verbesserter Reward Shaping Wrapper für DM Control Soccer.

Ziele:
- Stärkere, gecappte Signale für schnelleres Lernen.
- Verwendung nur von ego-basierten Distanzen (keine Weltkoordinaten-Annäherung nötig).
- Belohnung für Ballkontrolle, Ball-Richtung gegnerisches Tor, Schüsse.
- Strafen für Sturz und Stillstand.
"""
import numpy as np
from dm_control.locomotion import soccer as dm_soccer


class SoccerRewardWrapperOptimized:
    """
    Wrapper mit starkem, aber begrenztem Reward Shaping.

    Team-Indizes:
        Home: 0, 1
        Away: 2, 3
    """

    def __init__(self, env, reward_scale=1.0,
                 ball_proximity_weight=0.1,
                 ball_to_goal_weight=2.0,
                 moving_to_ball_weight=0.8,
                 possession_bonus=0.5,
                 shot_to_goal_weight=1.5,
                 movement_bonus=0.3,
                 fall_penalty=0.5,
                 idle_penalty=0.2):
        self.env = env
        self.reward_scale = reward_scale

        self.ball_proximity_weight = ball_proximity_weight
        self.ball_to_goal_weight = ball_to_goal_weight
        self.moving_to_ball_weight = moving_to_ball_weight
        self.possession_bonus = possession_bonus
        self.shot_to_goal_weight = shot_to_goal_weight
        self.movement_bonus = movement_bonus
        self.fall_penalty = fall_penalty
        self.idle_penalty = idle_penalty

        self._team_size = 2
        self._num_players = 4

        self._prev_ball_dist = None
        self._prev_ball_to_goal = None
        self._prev_player_pos = None

    # --- Hilfsmethoden ---
    def _get(self, obs, player_idx, key):
        """Sicheres Auslesen eines Observations-Keys."""
        val = obs[player_idx].get(key)
        if val is None:
            return None
        # DM Control liefert oft (1, n) Arrays für Einzelwerte
        arr = np.asarray(val)
        if arr.ndim == 2 and arr.shape[0] == 1:
            arr = arr[0]
        return arr

    def _norm(self, vec):
        if vec is None:
            return 0.0
        return float(np.linalg.norm(vec))

    def _distance_to_ball(self, obs, player_idx):
        ball_pos = self._get(obs, player_idx, 'ball_ego_position')
        if ball_pos is None:
            return 0.0
        return self._norm(ball_pos)

    def _distance_ball_to_goal(self, obs, player_idx):
        ball_pos = self._get(obs, player_idx, 'ball_ego_position')
        goal_pos = self._get(obs, player_idx, 'opponent_goal_mid')
        if ball_pos is None or goal_pos is None:
            return 0.0
        return self._norm(ball_pos - goal_pos)

    def _upright(self, obs, player_idx):
        """
        Einfache Sturzerkennung über Z-Position des Walkers.
        Falls walker_ego_position nicht verfügbar, gehen wir von aufrecht aus.
        """
        walker_pos = self._get(obs, player_idx, 'walker_ego_position')
        if walker_pos is None or walker_pos.size < 3:
            return True
        return float(walker_pos[2]) > 0.25

    def _player_pos(self, obs, player_idx):
        return self._get(obs, player_idx, 'walker_ego_position')

    def _compute_shaped_reward(self, obs, base_rewards):
        shaped = np.zeros(self._num_players, dtype=np.float32)

        # Distanzen pro Team (wir nutzen den ersten Spieler pro Team als Referenz)
        ball_dist_team = [
            self._distance_to_ball(obs, 0),
            self._distance_to_ball(obs, self._team_size),
        ]
        ball_to_goal_team = [
            self._distance_ball_to_goal(obs, 0),
            self._distance_ball_to_goal(obs, self._team_size),
        ]

        for p in range(self._num_players):
            team = 0 if p < self._team_size else 1
            ball_dist = self._distance_to_ball(obs, p)
            ball_to_goal = self._distance_ball_to_goal(obs, p)

            # 1) Annäherung an Ball (Delta)
            if self._prev_ball_dist is not None:
                delta = self._prev_ball_dist[team] - ball_dist
                shaped[p] += self.moving_to_ball_weight * np.clip(delta, -1.0, 1.0)

            # 2) Ballnähe-Bonus (kleiner, um Passivität zu vermeiden)
            shaped[p] += self.ball_proximity_weight * np.clip(1.0 - ball_dist / 5.0, 0.0, 1.0)

            # 2b) Bewegungs-Bonus: belohne aktive Bewegung (vermeidet Stehenbleiben)
            if self._prev_player_pos is not None:
                curr_pos = self._player_pos(obs, p)
                if curr_pos is not None:
                    moved = self._norm(curr_pos - self._prev_player_pos[p])
                    shaped[p] += self.movement_bonus * np.clip(moved / 0.5, 0.0, 1.0)

            # 3) Ballbewegung Richtung gegnerisches Tor (Delta)
            if self._prev_ball_to_goal is not None:
                delta = self._prev_ball_to_goal[team] - ball_to_goal
                shaped[p] += self.ball_to_goal_weight * np.clip(delta, -1.0, 1.0)

            # 4) Ballbesitz-Bonus: sehr nah am Ball UND Ball näher am gegnerischen Tor
            if self._prev_ball_to_goal is not None and ball_dist < 0.5 and ball_to_goal < self._prev_ball_to_goal[team]:
                shaped[p] += self.possession_bonus

            # 5) Schuss/Pass Richtung Tor: schnelle Ballbewegung Richtung Tor.
            #    Da ego-Positionen sich mit dem Spieler bewegen, nutzen wir die Team-Referenz-Deltas
            #    als Proxy für Ballgeschwindigkeit Richtung Tor.
            if self._prev_ball_to_goal is not None:
                delta = self._prev_ball_to_goal[team] - ball_to_goal
                # Nur wenn der Ball sich schnell nähert
                shaped[p] += self.shot_to_goal_weight * np.clip(delta, -1.0, 2.0)

            # 6) Sturz-Strafe
            if not self._upright(obs, p):
                shaped[p] -= self.fall_penalty

            # 7) Idle-Strafe (wenig Bewegung) wenn nicht nah am Ball
            if self._prev_player_pos is not None and ball_dist > 1.0:
                curr_pos = self._player_pos(obs, p)
                if curr_pos is not None:
                    moved = self._norm(curr_pos - self._prev_player_pos[p])
                    if moved < 0.03:
                        shaped[p] -= self.idle_penalty

        self._prev_ball_dist = ball_dist_team
        self._prev_ball_to_goal = ball_to_goal_team
        self._prev_player_pos = np.stack([
            self._player_pos(obs, p) if self._player_pos(obs, p) is not None else np.zeros(3)
            for p in range(self._num_players)
        ])

        return shaped

    def reset(self):
        timestep = self.env.reset()
        obs = timestep.observation
        self._prev_ball_dist = [
            self._distance_to_ball(obs, 0),
            self._distance_to_ball(obs, self._team_size),
        ]
        self._prev_ball_to_goal = [
            self._distance_ball_to_goal(obs, 0),
            self._distance_ball_to_goal(obs, self._team_size),
        ]
        self._prev_player_pos = np.stack([
            self._player_pos(obs, p) if self._player_pos(obs, p) is not None else np.zeros(3)
            for p in range(self._num_players)
        ])
        return timestep

    def step(self, actions):
        timestep = self.env.step(actions)
        base_rewards = np.asarray(timestep.reward, dtype=np.float32)
        shaped = self._compute_shaped_reward(timestep.observation, base_rewards)
        combined = base_rewards + shaped * self.reward_scale
        combined = combined.astype(np.float32)

        from dm_env import TimeStep
        new_timestep = TimeStep(
            step_type=timestep.step_type,
            reward=tuple(combined.tolist()),
            discount=timestep.discount,
            observation=timestep.observation,
        )
        return new_timestep

    def __getattr__(self, name):
        return getattr(self.env, name)


def make_env_with_rewards_optimized(seed=None, reward_scale=1.0, **kwargs):
    """Erstellt Soccer-Environment mit optimiertem Reward Shaping."""
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
    return SoccerRewardWrapperOptimized(env, reward_scale=reward_scale, **kwargs)
