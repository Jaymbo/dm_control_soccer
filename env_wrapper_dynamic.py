"""
Dynamic Scoring Reward Wrapper für DM Control Soccer.

Ersetzt das Phasen-Curriculum durch einen stets aktiven, zustandsbasierten
Reward-Switcher. Jeder Spieler erhält zu jedem Zeitpunkt genau EIN dominantes
Behavioral-Objective (Branch):

  1. RECOVERY    – Agent liegt / ist gestürzt -> wieder aufstellen
  2. PURSUIT     – Ball ist frei -> Ball anlaufen
  3. POSSESSION  – Eigener Team hat Ballbesitz -> Ball Richtung gegnerisches Tor
  4. DEFENSE     – Gegner hat Ballbesitz -> verteidigen (Presser / Anchor)

Alle Branches nutzen Potential-Based Reward Shaping (PBRS), damit die
Policy-Optimum-Eigenschaft erhalten bleibt. Rewards werden pro Branch
akkumuliert, damit das Training die Verteilung loggen kann.

Team-Indizes:
    Home: 0, 1
    Away: 2, 3
"""
import numpy as np
from dm_control.locomotion import soccer as dm_soccer

from env_wrapper_optimized import SoccerRewardWrapperOptimized


class DynamicScoringWrapper(SoccerRewardWrapperOptimized):
    """
    Zustandsbasierter Reward-Wrapper mit Branch-Tracking.

    Parameter:
        env:                 DM Control Soccer Environment
        gamma:               Discount für PBRS (default 0.99)
        reward_scale:        Globaler Skalierungsfaktor für shaped rewards
        possession_radius:   Maximaler Ballabstand für Ballbesitz (m)
        goal_threshold:      Torlinien-Distanz für Schuss-Detection (m)
        upright_height:      Körperhöhe, unter der ein Agent als gestürzt gilt
        lambda_recover:      Gewicht RECOVERY-Branch
        lambda_pursuit:      Gewicht PURSUIT-Branch
        lambda_possession:   Gewicht POSSESSION-Branch
        lambda_defense:      Gewicht DEFENSE-Branch
        **base_kwargs:       Weitere Parameter für den Base-Wrapper (werden ignoriert)
    """

    BRANCH_NAMES = ["recovery", "pursuit", "possession", "defense"]

    def __init__(
        self,
        env,
        gamma=0.99,
        reward_scale=1.0,
        possession_radius=0.6,
        goal_threshold=6.0,
        upright_height=0.5,
        lambda_recover=1.0,
        lambda_pursuit=1.0,
        lambda_possession=1.0,
        lambda_defense=1.0,
        **base_kwargs,
    ):
        # Base-Wrapper mit Default-Werten initialisieren (wir überschreiben
        # _compute_shaped_reward komplett, daher sind dessen Gewichte egal).
        super().__init__(env, reward_scale=1.0, **base_kwargs)

        self.gamma = gamma
        self.reward_scale = reward_scale
        self.possession_radius = possession_radius
        self.goal_threshold = goal_threshold
        self.upright_height = upright_height

        # Branch-Gewichte (λ) – werden vor der Summierung angewendet.
        self.lambda_recover = lambda_recover
        self.lambda_pursuit = lambda_pursuit
        self.lambda_possession = lambda_possession
        self.lambda_defense = lambda_defense

        # Zustandsmerkmale für PBRS und Detection
        self._num_players = 4
        self._team_size = 2

        # Episode-Branch-Stats
        self._episode_branch_rewards = None
        self._last_branch_rewards = None
        self._reset_branch_stats()

        # Vorherige Werte für PBRS
        self._prev_phi = None

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------
    def _reset_branch_stats(self):
        self._episode_branch_rewards = {b: 0.0 for b in self.BRANCH_NAMES}
        self._last_branch_rewards = {b: 0.0 for b in self.BRANCH_NAMES}

    def _body_height(self, obs, player_idx):
        """Körperhöhe als Sturz-Proxy."""
        h = self._get(obs, player_idx, "body_height")
        if h is None:
            return 1.0  # Unbekannt -> aufrecht annehmen
        arr = np.asarray(h).flatten()
        return float(arr[0]) if arr.size > 0 else 1.0

    def _is_fallen(self, obs, player_idx):
        return self._body_height(obs, player_idx) < self.upright_height

    def _velocity(self, obs, player_idx):
        """Eigener Geschwindigkeitsvektor (m/s)."""
        v = self._get(obs, player_idx, "sensors_velocimeter")
        if v is None:
            return np.zeros(3, dtype=np.float32)
        return np.asarray(v, dtype=np.float32).flatten()[:3]

    def _ball_velocity_ego(self, obs, player_idx):
        """Ballgeschwindigkeit in ego-Koordinaten des Spielers."""
        bv = self._get(obs, player_idx, "ball_ego_velocity")
        if bv is None:
            return np.zeros(3, dtype=np.float32)
        return np.asarray(bv, dtype=np.float32).flatten()[:3]

    def _team_for_player(self, player_idx):
        return 0 if player_idx < self._team_size else 1

    def _opponent_goal_pos(self, obs, player_idx):
        """Ego-Position des gegnerischen Tormittelpunkts."""
        return self._get(obs, player_idx, "opponent_goal_mid")

    def _own_goal_pos(self, obs, player_idx):
        """Ego-Position des eigenen Tormittelpunkts."""
        return self._get(obs, player_idx, "own_goal_mid")

    def _predict_intercept(self, obs, player_idx, horizon=0.3):
        """
        Vorhergesagter Ball-ego-Positions-Schnittpunkt in 'horizon' Sekunden.
        Ego-Position des Spielers ist definitionsgemäß (0,0,0), daher ist die
        Distanz einfach ||predicted_ball_ego||.
        """
        ball_pos = self._get(obs, player_idx, "ball_ego_position")
        if ball_pos is None:
            return None
        ball_pos = np.asarray(ball_pos, dtype=np.float32).flatten()[:3]
        ball_vel = self._ball_velocity_ego(obs, player_idx)
        predicted = ball_pos + ball_vel * horizon
        # Nur horizontale Ebene relevant
        predicted[2] = 0.0
        return predicted

    # ------------------------------------------------------------------
    # Ballbesitz-Detection
    # ------------------------------------------------------------------
    def _detect_possession(self, obs):
        """
        Bestimmt für beide Teams, ob sie Ballbesitz haben.
        Rückgabe: (home_has_ball, away_has_ball, ball_owner_player_idx)
        """
        best_dist = float("inf")
        owner = -1
        for p in range(self._num_players):
            d = self._distance_to_ball(obs, p)
            if d < best_dist:
                best_dist = d
                owner = p

        if best_dist > self.possession_radius:
            return False, False, -1

        team = self._team_for_player(owner)
        home = team == 0
        away = not home
        return home, away, owner

    # ------------------------------------------------------------------
    # Defense-Rollen
    # ------------------------------------------------------------------
    def _assign_defense_roles(self, obs, defending_team):
        """
        Weist den Verteidigern dynamisch Rollen zu:
          - presser:  Spieler des verteidigenden Teams, der dem Ball am nächsten ist
          - anchor:   Spieler des verteidigenden Teams, der dem eigenen Tor am nächsten ist
        Rückgabe: dict player_idx -> 'presser' | 'anchor' | None
        """
        roles = {p: None for p in range(self._num_players)}
        defenders = (
            list(range(self._team_size))
            if defending_team == 0
            else list(range(self._team_size, self._num_players))
        )
        if not defenders:
            return roles

        dist_to_ball = {
            p: self._distance_to_ball(obs, p) for p in defenders
        }
        presser = min(defenders, key=lambda p: dist_to_ball[p])

        own_goal = self._own_goal_pos(obs, defenders[0])
        if own_goal is not None:
            dist_to_goal = {
                p: self._norm(self._get(obs, p, "walker_ego_position") - own_goal)
                if self._get(obs, p, "walker_ego_position") is not None
                else float("inf")
                for p in defenders
            }
            anchor = min(defenders, key=lambda p: dist_to_goal[p])
        else:
            anchor = presser

        # Verhindere gleiche Rolle, wenn es zwei Verteidiger gibt
        if len(defenders) > 1 and anchor == presser:
            anchor = max(defenders, key=lambda p: dist_to_ball[p])

        roles[presser] = "presser"
        roles[anchor] = "anchor"
        return roles

    # ------------------------------------------------------------------
    # Branch Reward Funktionen
    # ------------------------------------------------------------------
    def _branch_reward_and_phi(self, obs, player_idx, branch):
        """
        Berechnet ungefilterten Reward r(s) und Potential Φ(s) für einen Branch.
        Rückgabe: (reward, phi)
        """
        if branch == "recovery":
            height = self._body_height(obs, player_idx)
            # reward: hoch wenn aufrecht, niedrig wenn gestürzt
            reward = np.clip((height - self.upright_height) / self.upright_height, -1.0, 1.0)
            # potential: aufrecht sein ist gut
            phi = np.clip((height - 0.3) / 0.7, 0.0, 1.0)
            return reward, phi

        if branch == "pursuit":
            # Ziel: möglichst nah am Ball; zusätzlich Antizipation
            ball_dist = self._distance_to_ball(obs, player_idx)
            intercept = self._predict_intercept(obs, player_idx)
            intercept_dist = self._norm(intercept) if intercept is not None else ball_dist
            # Distanz-minimierung
            reward = -0.1 * np.clip(ball_dist / 10.0, 0.0, 1.0)
            # Antizipations-Bonus
            reward += -0.05 * np.clip(intercept_dist / 10.0, 0.0, 1.0)
            # Aktive Bewegung belohnen
            speed = self._norm(self._velocity(obs, player_idx))
            reward += 0.05 * np.clip(speed / 3.0, 0.0, 1.0)
            phi = -np.clip(ball_dist / 10.0, 0.0, 1.0)
            return reward, phi

        if branch == "possession":
            # Ziel: Ball näher ans gegnerische Tor bringen + Ballbesitz halten
            ball_to_goal = self._distance_ball_to_goal(obs, player_idx)
            ball_dist = self._distance_to_ball(obs, player_idx)
            speed = self._norm(self._velocity(obs, player_idx))
            reward = -0.2 * np.clip(ball_to_goal / 20.0, 0.0, 1.0)
            reward += 0.1 * np.clip(speed / 3.0, 0.0, 1.0)
            # Bonus für Ballnähe (Besitz halten)
            if ball_dist < self.possession_radius:
                reward += 0.2
            phi = -np.clip(ball_to_goal / 20.0, 0.0, 1.0)
            return reward, phi

        if branch == "defense":
            # Rollen-spezifische Potentiale
            # Wir wissen die Rolle nicht hier; daher berechnen wir beide
            # Varianten und die Hauptroutine wählt die passende aus.
            ball_to_goal = self._distance_ball_to_goal(obs, player_idx)
            ball_dist = self._distance_to_ball(obs, player_idx)
            speed = self._norm(self._velocity(obs, player_idx))
            reward = 0.05 * np.clip(speed / 3.0, 0.0, 1.0)
            # Generisches Verteidigungspotential: Ball sollte weg vom eigenen Tor sein
            phi = -np.clip(ball_to_goal / 20.0, 0.0, 1.0)
            return reward, phi

        return 0.0, 0.0

    def _defense_reward_and_phi(self, obs, player_idx, role):
        """
        Defense-Reward mit Rollenunterscheidung.
        """
        ball_to_goal = self._distance_ball_to_goal(obs, player_idx)
        ball_dist = self._distance_to_ball(obs, player_idx)
        speed = self._norm(self._velocity(obs, player_idx))

        if role == "presser":
            # Ball schnell erreichen / anlaufen
            reward = -0.2 * np.clip(ball_dist / 10.0, 0.0, 1.0)
            reward += 0.05 * np.clip(speed / 3.0, 0.0, 1.0)
            phi = -np.clip(ball_dist / 10.0, 0.0, 1.0)
        elif role == "anchor":
            # Zwischen Ball und eigenem Tor bleiben
            own_goal = self._own_goal_pos(obs, player_idx)
            player_pos = self._get(obs, player_idx, "walker_ego_position")
            ball_pos = self._get(obs, player_idx, "ball_ego_position")
            if own_goal is not None and player_pos is not None and ball_pos is not None:
                # Ideal: Spieler liegt auf der Verbindung Ball -> Tor
                vec_ball_goal = own_goal - ball_pos
                vec_ball_player = player_pos - ball_pos
                len_bg = np.linalg.norm(vec_ball_goal) + 1e-8
                projection = np.dot(vec_ball_player, vec_ball_goal) / len_bg
                alignment = projection / (len_bg + 1e-8)
                reward = 0.2 * np.clip(alignment, 0.0, 1.0)
                reward -= 0.1 * np.clip(ball_to_goal / 20.0, 0.0, 1.0)
                phi = 0.2 * np.clip(alignment, 0.0, 1.0)
            else:
                reward = -0.1 * np.clip(ball_to_goal / 20.0, 0.0, 1.0)
                phi = -np.clip(ball_to_goal / 20.0, 0.0, 1.0)
        else:
            reward = 0.0
            phi = -np.clip(ball_to_goal / 20.0, 0.0, 1.0)

        return reward, phi

    # ------------------------------------------------------------------
    # Haupt-Routine
    # ------------------------------------------------------------------
    def _branch_for_player(self, obs, player_idx, home_has_ball, away_has_ball):
        """Prioritätsbasierte Branch-Zuweisung pro Spieler."""
        team = self._team_for_player(player_idx)
        own_has = home_has_ball if team == 0 else away_has_ball
        opp_has = away_has_ball if team == 0 else home_has_ball

        # 1) Recovery höchste Priorität
        if self._is_fallen(obs, player_idx):
            return "recovery"

        # 2) Wenn gegnerischer Ballbesitz -> Defense
        if opp_has:
            return "defense"

        # 3) Wenn eigener Ballbesitz -> Possession
        if own_has:
            return "possession"

        # 4) Sonst: Ball verfolgen
        return "pursuit"

    def _compute_shaped_reward(self, obs, base_rewards):
        """
        Berechnet zustandsabhängige Branch-Rewards mit PBRS.
        base_rewards wird nicht verändert, sondern nur zum Timing genutzt.
        """
        shaped = np.zeros(self._num_players, dtype=np.float32)
        phi_current = np.zeros(self._num_players, dtype=np.float32)

        home_has_ball, away_has_ball, ball_owner = self._detect_possession(obs)

        # Defense-Rollen für beide Teams ermitteln (nur relevant, wenn Gegner Ball hat)
        defense_roles = [None] * self._num_players
        for team in (0, 1):
            opp_has = away_has_ball if team == 0 else home_has_ball
            if opp_has:
                roles = self._assign_defense_roles(obs, defending_team=team)
                for p, role in roles.items():
                    if role is not None:
                        defense_roles[p] = role

        for p in range(self._num_players):
            branch = self._branch_for_player(obs, p, home_has_ball, away_has_ball)

            if branch == "defense":
                role = defense_roles[p] or "presser"
                r, phi = self._defense_reward_and_phi(obs, p, role)
                lam = self.lambda_defense
            else:
                r, phi = self._branch_reward_and_phi(obs, p, branch)
                lam = getattr(self, f"lambda_{branch}")

            # PBRS: F = γΦ(s') - Φ(s). Hier wird pro Step angewendet:
            # shaped_reward = r(s) + γΦ(s) - Φ(s_prev)
            # Wir speichern phi_current für den nächsten Step.
            phi_current[p] = phi

            prev_phi = (
                self._prev_phi[p]
                if self._prev_phi is not None
                else phi
            )
            pbrs = self.gamma * phi - prev_phi

            shaped[p] = lam * (r + pbrs)

            # Anti-Hacking: Clipping pro Branch
            clip_val = 1.0
            if branch == "recovery":
                clip_val = 0.5
            elif branch == "defense":
                clip_val = 0.8
            shaped[p] = np.clip(shaped[p], -clip_val, clip_val)

            # Branch-Stats aktualisieren (für TensorBoard)
            self._episode_branch_rewards[branch] += float(shaped[p])

        self._prev_phi = phi_current
        self._last_branch_rewards = self._episode_branch_rewards.copy()
        return shaped

    # ------------------------------------------------------------------
    # dm_env API
    # ------------------------------------------------------------------
    def reset(self):
        self._reset_branch_stats()
        self._prev_phi = None
        timestep = super().reset()
        # Erstes Phi nach Reset berechnen (für PBRS-Konsistenz)
        _ = self._compute_shaped_reward(timestep.observation, np.zeros(self._num_players))
        return timestep

    def step(self, actions):
        timestep = self.env.step(actions)
        base_rewards = np.asarray(timestep.reward, dtype=np.float32)
        shaped = self._compute_shaped_reward(timestep.observation, base_rewards)
        combined = base_rewards + shaped * self.reward_scale
        combined = combined.astype(np.float32)

        from dm_env import TimeStep
        return TimeStep(
            step_type=timestep.step_type,
            reward=tuple(combined.tolist()),
            discount=timestep.discount,
            observation=timestep.observation,
        )

    # ------------------------------------------------------------------
    # Statistik-API für Training
    # ------------------------------------------------------------------
    def get_branch_rewards(self, reset=True):
        """
        Gibt die akkumulierten Branch-Rewards seit dem letzten Reset zurück.
        Wenn reset=True, werden die Stats zurückgesetzt.
        """
        stats = self._episode_branch_rewards.copy()
        if reset:
            self._reset_branch_stats()
        return stats

    def get_last_branch_rewards(self):
        """Letzte bekannte Branch-Rewards (für Logging zwischen Episoden)."""
        return self._last_branch_rewards.copy()


def make_env_with_dynamic_rewards(
    seed=None,
    reward_scale=1.0,
    possession_radius=0.6,
    goal_threshold=6.0,
    lambda_recover=1.0,
    lambda_pursuit=1.0,
    lambda_possession=1.0,
    lambda_defense=1.0,
    **kwargs,
):
    """Erstellt Soccer-Environment mit Dynamic Scoring Wrapper."""
    env = dm_soccer.load(
        team_size=2,
        time_limit=10.0,
        disable_walker_contacts=False,
        enable_field_box=True,
        terminate_on_goal=False,
        walker_type=dm_soccer.WalkerType.BOXHEAD,
    )
    if seed is not None:
        env.task._random_state = np.random.RandomState(seed)
    return DynamicScoringWrapper(
        env,
        reward_scale=reward_scale,
        possession_radius=possession_radius,
        goal_threshold=goal_threshold,
        lambda_recover=lambda_recover,
        lambda_pursuit=lambda_pursuit,
        lambda_possession=lambda_possession,
        lambda_defense=lambda_defense,
        **kwargs,
    )
