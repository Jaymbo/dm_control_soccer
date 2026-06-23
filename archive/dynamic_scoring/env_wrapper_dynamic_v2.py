"""
Dynamic Scoring Reward Wrapper V2 für DM Control Soccer.

Rollen-basiertes Reward-Shaping für beliebige Teamgrößen N vs N.
Alle Berechnungen erfolgen ausschließlich in ego-zentrierten Koordinaten,
da DM Control Soccer keine globale Walker-Position in den Observations
liefert. Jeder Spieler hat seine eigene Sicht auf:
  - ball_ego_position
  - teammate_<i>_ego_position
  - opponent_<i>_ego_position
  - team_goal_mid / opponent_goal_mid

Branches:
  RECOVERY    – Eigenster Spieler zum Ball läuft (nur er bekommt Reward).
  MARKING     – Andere eigene Spieler decken Gegner auf Linie Ball->Gegner
                oder nehmen Angriffsposition auf Linie Ball->Tor ein.
                Zuweisung per Hungarian Algorithmus mit quadratischen Kosten.
  POSSESSION  – Eigener Spieler hat Ball -> Ball Richtung gegnerisches Tor.
  ATTACK_POS  – Mitspieler mit freier Torsicht und offenem Ballzugang.
  SHOOTING    – Ball fliegt Richtung gegnerisches Tor.
  BLOCKING    – Ball fliegt Richtung eigenes Tor -> Abwehr.
  GOALKEEPING – Position zwischen Ball und eigenem Tor.
"""
import numpy as np
from dm_control.locomotion import soccer as dm_soccer

try:
    from scipy.optimize import linear_sum_assignment
    _SCIPY_AVAILABLE = True
except Exception:
    _SCIPY_AVAILABLE = False


class DynamicScoringWrapperV2:
    """
    Zustandsbasierter Reward-Wrapper V2 mit rollenbasierter Zuweisung.
    """

    BRANCH_NAMES = [
        "recovery",
        "marking",
        "possession",
        "attack_pos",
        "shooting",
        "blocking",
        "goalkeeping",
    ]

    def __init__(
        self,
        env,
        gamma=0.99,
        reward_scale=1.0,
        possession_radius=0.6,
        shot_speed_threshold=2.0,
        goal_width=2.0,
        upright_height=0.5,
        lambda_recovery=1.0,
        lambda_marking=1.0,
        lambda_possession=1.0,
        lambda_shooting=1.0,
        lambda_blocking=1.0,
        lambda_goalkeeping=1.0,
        lambda_attack_pos=0.5,
    ):
        self.env = env
        self.gamma = gamma
        self.reward_scale = reward_scale
        self.possession_radius = possession_radius
        self.shot_speed_threshold = shot_speed_threshold
        self.goal_width = goal_width
        self.upright_height = upright_height

        self.lambda_recovery = lambda_recovery
        self.lambda_marking = lambda_marking
        self.lambda_possession = lambda_possession
        self.lambda_shooting = lambda_shooting
        self.lambda_blocking = lambda_blocking
        self.lambda_goalkeeping = lambda_goalkeeping
        self.lambda_attack_pos = lambda_attack_pos

        # Wird dynamisch aus Observations gesetzt
        self._num_players = None
        self._team_size = None

        # Branch-Stats
        self._episode_branch_rewards = None
        self._last_branch_rewards = None
        self._reset_branch_stats()

        # Vorherige Werte für Deltas / PBRS
        self._prev_phi = None
        self._prev_marking_target_dist = {}
        self._prev_ball_to_goal = None
        self._prev_player_dist_to_ball = None

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------
    def _reset_branch_stats(self):
        self._episode_branch_rewards = {b: 0.0 for b in self.BRANCH_NAMES}
        self._last_branch_rewards = {b: 0.0 for b in self.BRANCH_NAMES}

    def _detect_team_size(self, obs):
        """Leitet team_size aus der Anzahl der Spieler ab."""
        num_players = len(obs)
        if num_players % 2 != 0:
            raise ValueError(f"Ungerade Spieleranzahl: {num_players}")
        return num_players // 2, num_players

    def _get(self, obs, player_idx, key):
        """Sicheres Auslesen eines Observations-Keys."""
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

    def _team_for_player(self, player_idx):
        return 0 if player_idx < self._team_size else 1

    def _teammates(self, player_idx):
        """Indizes der Mitspieler (ohne sich selbst)."""
        team = self._team_for_player(player_idx)
        start = 0 if team == 0 else self._team_size
        end = start + self._team_size
        return [i for i in range(start, end) if i != player_idx]

    def _opponents(self, player_idx):
        """Indizes der gegnerischen Spieler."""
        team = self._team_for_player(player_idx)
        start = self._team_size if team == 0 else 0
        end = start + self._team_size
        return list(range(start, end))

    def _local_teammate_index(self, player_idx, teammate_idx):
        """Lokaler Index für teammate_<i>_ego_position Keys."""
        teammates = self._teammates(player_idx)
        if teammate_idx not in teammates:
            return -1
        return teammates.index(teammate_idx)

    def _local_opponent_index(self, player_idx, opponent_idx):
        """Lokaler Index für opponent_<i>_ego_position Keys."""
        opps = self._opponents(player_idx)
        if opponent_idx not in opps:
            return -1
        return opps.index(opponent_idx)

    def _teammate_ego_position(self, obs, player_idx, teammate_idx):
        """Position eines Teamkollegen aus Sicht von player_idx."""
        local_idx = self._local_teammate_index(player_idx, teammate_idx)
        if local_idx < 0:
            return None
        key = f"teammate_{local_idx}_ego_position"
        return self._get(obs, player_idx, key)

    def _opponent_ego_position(self, obs, player_idx, opponent_idx):
        """Position eines Gegners aus Sicht von player_idx."""
        local_idx = self._local_opponent_index(player_idx, opponent_idx)
        if local_idx < 0:
            return None
        key = f"opponent_{local_idx}_ego_position"
        return self._get(obs, player_idx, key)

    def _distance_to_ball(self, obs, player_idx):
        ball_pos = self._get(obs, player_idx, "ball_ego_position")
        return self._norm(ball_pos)

    def _distance_ball_to_opponent_goal(self, obs, player_idx):
        ball_pos = self._get(obs, player_idx, "ball_ego_position")
        goal_pos = self._get(obs, player_idx, "opponent_goal_mid")
        if ball_pos is None or goal_pos is None:
            return 0.0
        return self._norm(ball_pos - goal_pos)

    def _body_height(self, obs, player_idx):
        h = self._get(obs, player_idx, "body_height")
        if h is None:
            return 1.0
        arr = np.asarray(h).flatten()
        return float(arr[0]) if arr.size > 0 else 1.0

    def _is_fallen(self, obs, player_idx):
        return self._body_height(obs, player_idx) < self.upright_height

    def _velocity(self, obs, player_idx):
        v = self._get(obs, player_idx, "sensors_velocimeter")
        if v is None:
            return np.zeros(3, dtype=np.float32)
        return np.asarray(v, dtype=np.float32).flatten()[:3]

    def _ball_velocity_ego(self, obs, player_idx):
        bv = self._get(obs, player_idx, "ball_ego_linear_velocity")
        if bv is None:
            return np.zeros(3, dtype=np.float32)
        return np.asarray(bv, dtype=np.float32).flatten()[:3]

    # ------------------------------------------------------------------
    # Geometrie-Hilfsmethoden (ego-zentriert)
    # ------------------------------------------------------------------
    @staticmethod
    def _point_line_distance_squared(point, line_start, line_end):
        """Quadratischer Abstand eines Punktes zur Strecke."""
        line_vec = line_end - line_start
        line_len_sq = np.dot(line_vec, line_vec)
        if line_len_sq < 1e-12:
            return np.dot(point - line_start, point - line_start)
        t = np.clip(np.dot(point - line_start, line_vec) / line_len_sq, 0.0, 1.0)
        projection = line_start + t * line_vec
        diff = point - projection
        return np.dot(diff, diff)

    @staticmethod
    def _point_line_projection_ratio(point, line_start, line_end):
        """t der Projektion auf die Strecke (0..1 = dazwischen)."""
        line_vec = line_end - line_start
        line_len_sq = np.dot(line_vec, line_vec)
        if line_len_sq < 1e-12:
            return 0.0
        return np.dot(point - line_start, line_vec) / line_len_sq

    @staticmethod
    def _horizontal_distance(vec):
        """2D-Distanz in der x-y-Ebene (ignoriert z)."""
        if vec is None:
            return 0.0
        v = np.asarray(vec).flatten()[:2]
        return float(np.linalg.norm(v))

    # ------------------------------------------------------------------
    # Zuweisungen
    # ------------------------------------------------------------------
    def _find_ball_chaser(self, obs, team):
        """Index des eigenen Spielers, der dem Ball am nächsten ist."""
        players = list(range(team * self._team_size, (team + 1) * self._team_size))
        dists = {p: self._distance_to_ball(obs, p) for p in players}
        return min(dists, key=dists.get)

    def _find_closest_opponent_to_ball(self, obs, ref_player):
        """Index des Gegners, der dem Ball am nächsten ist (aus Sicht ref_player)."""
        opps = self._opponents(ref_player)
        ball_pos = self._get(obs, ref_player, "ball_ego_position")
        if ball_pos is None:
            return opps[0]
        best = None
        best_dist = float("inf")
        for opp in opps:
            opp_pos = self._opponent_ego_position(obs, ref_player, opp)
            if opp_pos is None:
                continue
            d = self._norm(opp_pos - ball_pos)
            if d < best_dist:
                best_dist = d
                best = opp
        return best if best is not None else opps[0]

    def _compute_marking_assignments(self, obs, team):
        """
        Weist jedem eigenen Spieler außer dem Ball-Chaser eine Linie zu:
          - ('goal', None)       -> Ball -> gegnerisches Tor
          - ('opponent', opp)    -> Ball -> verbleibender Gegner
        Der ball-nächste Gegner wird ignoriert (ist durch Ball-Chaser gedeckt).

        Rückgabe: dict {player_idx: ('goal', None) | ('opponent', opp_idx)}
        """
        team_players = list(range(team * self._team_size, (team + 1) * self._team_size))
        ball_chaser = self._find_ball_chaser(obs, team)
        remaining_teammates = [p for p in team_players if p != ball_chaser]

        if not remaining_teammates:
            return {}

        # Referenz: Ball-Chaser. Im ego-Frame des Ball-Chasers ist der Ball in (0,0,0).
        ref = ball_chaser
        ball_pos = np.zeros(3, dtype=np.float32)

        opp_indices = self._opponents(ref)
        closest_opp_to_ball = self._find_closest_opponent_to_ball(obs, ref)
        remaining_opponents = [o for o in opp_indices if o != closest_opp_to_ball]

        targets = [("goal", None)]
        for opp in remaining_opponents:
            targets.append(("opponent", opp))

        # Falls mehr Spieler als Ziele -> zusätzliche Tor-Linien
        while len(targets) < len(remaining_teammates):
            targets.append(("goal", None))

        cost_matrix = np.zeros((len(remaining_teammates), len(targets)))
        for i, player in enumerate(remaining_teammates):
            # Position des Mitspielers im Frame des Ball-Chasers
            player_pos = self._teammate_ego_position(obs, ref, player)
            if player_pos is None:
                cost_matrix[i, :] = 1e6
                continue
            player_pos = np.asarray(player_pos, dtype=np.float32).flatten()[:3]
            for j, (target_type, target_opp) in enumerate(targets):
                line_end = self._line_end_in_ref_frame(obs, ref, target_type, target_opp)
                if line_end is None:
                    cost = 1e6
                else:
                    cost = self._point_line_distance_squared(player_pos, ball_pos, line_end)
                cost_matrix[i, j] = cost

        if _SCIPY_AVAILABLE and len(remaining_teammates) == len(targets):
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            assignments = {}
            for r, c in zip(row_ind, col_ind):
                assignments[remaining_teammates[r]] = targets[c]
        else:
            assignments = self._greedy_assignment(remaining_teammates, targets, cost_matrix)

        return assignments

    def _line_end_in_ref_frame(self, obs, ref, target_type, target_opp):
        """Endpunkt der Ziel-Linie im ego-Frame des Referenzspielers ref."""
        if target_type == "goal":
            goal = self._get(obs, ref, "opponent_goal_mid")
            if goal is None:
                return None
            return np.asarray(goal, dtype=np.float32).flatten()[:3]
        else:
            opp_pos = self._opponent_ego_position(obs, ref, target_opp)
            if opp_pos is None:
                return None
            return np.asarray(opp_pos, dtype=np.float32).flatten()[:3]

    def _greedy_assignment(self, players, targets, cost_matrix):
        assignments = {}
        used_targets = set()
        rows = list(range(len(players)))
        for _ in range(min(len(players), len(targets))):
            best_cost = float("inf")
            best_row = -1
            best_col = -1
            for r in rows:
                for c in range(len(targets)):
                    if c in used_targets:
                        continue
                    if cost_matrix[r, c] < best_cost:
                        best_cost = cost_matrix[r, c]
                        best_row = r
                        best_col = c
            if best_row < 0:
                break
            assignments[players[best_row]] = targets[best_col]
            used_targets.add(best_col)
            rows.remove(best_row)
        return assignments

    # ------------------------------------------------------------------
    # Branch Reward Funktionen
    # ------------------------------------------------------------------
    def _reward_recovery(self, obs, player_idx, ball_chaser):
        """
        Nur Ball-Chaser bekommt Reward fürs Annähern an den Ball.
        
        WICHTIG: Time-Penalty verhindert Zeit-Schinden!
        """
        if player_idx != ball_chaser:
            return 0.0, 0.0

        dist = self._distance_to_ball(obs, player_idx)
        prev_dist = (
            self._prev_player_dist_to_ball[player_idx]
            if self._prev_player_dist_to_ball is not None
            else dist
        )
        delta = prev_dist - dist
        delta_clip = np.clip(delta, -1.0, 1.0)
        
        # Basis-Reward für Annäherung
        reward = 0.3 * delta_clip
        
        # Speed-Reward: Schnell sein lohnt sich!
        # Velocity in Richtung Ball
        ball_pos = self._get(obs, player_idx, "ball_ego_position")
        if ball_pos is not None:
            ball_dir = -np.asarray(ball_pos).flatten()  # Richtung Ball (im ego-Frame ist Ball bei -pos)
            ball_dist = self._norm(ball_pos)
            if ball_dist > 0.1:  # Nicht zu nah am Ball
                ball_dir = ball_dir / (ball_dist + 1e-8)
                vel = self._velocity(obs, player_idx)[:3]
                speed_towards_ball = np.dot(vel, ball_dir)
                # Bonus für schnelle Bewegung zum Ball (max 0.2 pro Step)
                speed_reward = 0.1 * np.clip(speed_towards_ball, 0.0, 2.0)
                reward += speed_reward
        
        # Bonus bei Erreichen (einmalig, nicht pro Step!)
        if dist < self.possession_radius:
            reward += 0.3  # Einmaliger Bonus, nicht wiederholbar
        
        # Time-Penalty: Jeder Step kostet (verhindert Zeit-Schinden)
        reward -= 0.02  # Kleine Strafe pro Step
        
        phi = -0.1 * np.clip(dist / 10.0, 0.0, 1.0)
        return reward, phi

    def _reward_marking(self, obs, player_idx, ball_chaser, assignment):
        """
        Marking-Reward: Spieler soll auf zugewiesener Linie sein und
        entlang dieser Richtung Ball/Tor laufen.
        Berechnung im Frame des Ball-Chasers.
        """
        if player_idx == ball_chaser or assignment is None:
            return 0.0, 0.0

        ref = ball_chaser
        ball_pos = np.zeros(3, dtype=np.float32)
        target_type, target_opp = assignment
        line_end = self._line_end_in_ref_frame(obs, ref, target_type, target_opp)
        if line_end is None:
            return 0.0, 0.0

        player_pos = self._teammate_ego_position(obs, ref, player_idx)
        if player_pos is None:
            return 0.0, 0.0
        player_pos = np.asarray(player_pos, dtype=np.float32).flatten()[:3]

        dist_sq = self._point_line_distance_squared(player_pos, ball_pos, line_end)
        dist = np.sqrt(dist_sq)

        target_dist = self._norm(player_pos - line_end)
        prev_target_dist = self._prev_marking_target_dist.get(player_idx, target_dist)
        delta_to_target = prev_target_dist - target_dist
        delta_clip = np.clip(delta_to_target, -1.0, 1.0)

        line_reward = -0.3 * np.clip(dist / 5.0, 0.0, 1.0)
        progress_reward = 0.5 * delta_clip
        t = self._point_line_projection_ratio(player_pos, ball_pos, line_end)
        between_bonus = 0.3 if 0.0 <= t <= 1.0 else 0.0

        reward = line_reward + progress_reward + between_bonus
        phi = -0.2 * np.clip(dist / 5.0, 0.0, 1.0)

        self._current_marking_target_dist[player_idx] = target_dist
        return reward, phi

    def _reward_possession(self, obs, player_idx, ball_owner):
        """
        Ball-Besitzer bringt Ball näher ans gegnerische Tor.
        
        WICHTIG: Time-Penalty + Speed-Reward verhindern Zeit-Schinden!
        """
        if player_idx != ball_owner:
            return 0.0, 0.0

        dist_to_goal = self._distance_ball_to_opponent_goal(obs, player_idx)
        prev_dist = self._prev_ball_to_goal if self._prev_ball_to_goal is not None else dist_to_goal
        delta = prev_dist - dist_to_goal
        delta_clip = np.clip(delta, -1.0, 2.0)
        
        # Basis-Reward für Annäherung (reduziert)
        reward = 1.0 * delta_clip
        
        # Speed-Reward: Ball schnell Richtung Tor bewegen
        ball_vel = self._ball_velocity_ego(obs, player_idx)
        goal_pos = self._get(obs, player_idx, "opponent_goal_mid")
        ball_pos = self._get(obs, player_idx, "ball_ego_position")
        if goal_pos is not None and ball_pos is not None:
            goal_dir = np.asarray(goal_pos).flatten() - np.asarray(ball_pos).flatten()
            goal_dist = self._norm(goal_dir)
            if goal_dist > 0.5:
                goal_dir = goal_dir / (goal_dist + 1e-8)
                # Ball-Geschwindigkeit Richtung Tor
                speed_towards_goal = np.dot(ball_vel, goal_dir)
                # Bonus für schnellen Schuss/Pass (max 0.3 pro Step)
                speed_reward = 0.15 * np.clip(speed_towards_goal, 0.0, 3.0)
                reward += speed_reward
        
        # Time-Penalty: Jeder Step kostet (verhindert Dribbling-Zeit-Schinden)
        reward -= 0.03  # Etwas höhere Strafe als bei Recovery
        
        # Kein wiederholbarer "Ball halten" Bonus mehr!
        # Nur noch Fortschritt zählt
        
        phi = -0.2 * np.clip(dist_to_goal / 20.0, 0.0, 1.0)
        return reward, phi

    def _reward_attack_position(self, obs, player_idx, ball_owner, assignments):
        """
        Mitspieler des Ballbesitzers, die auf Tor-Linie zugewiesen sind,
        bekommen Reward für freie Torsicht und offenen Ballzugang.
        """
        if player_idx == ball_owner:
            return 0.0, 0.0

        assignment = assignments.get(player_idx)
        if assignment is None or assignment[0] != "goal":
            return 0.0, 0.0

        player_pos = np.zeros(3, dtype=np.float32)  # Im eigenen Frame
        goal_pos = self._get(obs, player_idx, "opponent_goal_mid")
        ball_pos = self._get(obs, player_idx, "ball_ego_position")
        if goal_pos is None or ball_pos is None:
            return 0.0, 0.0
        goal_pos = np.asarray(goal_pos, dtype=np.float32).flatten()[:3]
        ball_pos = np.asarray(ball_pos, dtype=np.float32).flatten()[:3]

        to_goal = goal_pos - player_pos
        dist_to_goal = self._norm(to_goal)
        if dist_to_goal < 1e-8:
            return 0.0, 0.0
        goal_dir = to_goal / dist_to_goal

        opps = self._opponents(player_idx)
        blocked_width = 0.0
        for opp in opps:
            opp_pos = self._opponent_ego_position(obs, player_idx, opp)
            if opp_pos is None:
                continue
            opp_vec = opp_pos - player_pos
            proj = np.dot(opp_vec, goal_dir)
            if 0 < proj < dist_to_goal:
                perp = opp_vec - proj * goal_dir
                perp_dist = self._norm(perp)
                blocked_width += max(0.0, self.goal_width - perp_dist)

        visible_width = max(0.0, self.goal_width - blocked_width)
        vision_reward = 0.5 * np.clip(visible_width / self.goal_width, 0.0, 1.0)

        optimal_dist = 6.0
        dist_penalty = -0.05 * abs(dist_to_goal - optimal_dist)

        to_ball = ball_pos - player_pos
        dist_to_ball = self._norm(to_ball)
        ball_dir = to_ball / (dist_to_ball + 1e-8) if dist_to_ball > 1e-8 else np.zeros(3)
        ball_blocked = False
        for opp in opps:
            opp_pos = self._opponent_ego_position(obs, player_idx, opp)
            if opp_pos is None:
                continue
            opp_vec = opp_pos - player_pos
            proj = np.dot(opp_vec, ball_dir)
            if 0 < proj < dist_to_ball:
                perp = opp_vec - proj * ball_dir
                if self._norm(perp) < 1.0:
                    ball_blocked = True
                    break
        ball_open_reward = 0.3 if not ball_blocked else -0.1

        reward = vision_reward + dist_penalty + ball_open_reward
        phi = vision_reward
        return reward, phi

    def _reward_shooting(self, obs, player_idx, team, ball_owner):
        """
        Ball fliegt Richtung gegnerisches Tor - ABER NUR wenn eigener Spieler geschossen hat!
        
        WICHTIG: Verhindert Reward für zufällige Ballbewegungen!
        """
        # NUR der Ballbesitzer (oder Teamkollege) bekommt Shooting-Reward
        if ball_owner < 0 or self._team_for_player(ball_owner) != team:
            return 0.0, 0.0
        
        ball_vel = self._ball_velocity_ego(obs, player_idx)
        speed = self._norm(ball_vel)
        
        # Höherer Threshold: Nur echte Schüsse (nicht zufällige Berührungen)
        if speed < self.shot_speed_threshold * 1.5:  # War 2.0, jetzt 3.0
            return 0.0, 0.0

        ball_pos = self._get(obs, player_idx, "ball_ego_position")
        goal_pos = self._get(obs, player_idx, "opponent_goal_mid")
        if ball_pos is None or goal_pos is None:
            return 0.0, 0.0

        ball_pos = np.asarray(ball_pos, dtype=np.float32).flatten()[:3]
        goal_pos = np.asarray(goal_pos, dtype=np.float32).flatten()[:3]
        ball_dir = ball_vel / (speed + 1e-8)
        to_goal = goal_pos - ball_pos
        goal_dist = self._norm(to_goal)
        if goal_dist < 1e-8:
            return 0.0, 0.0
        goal_dir = to_goal / goal_dist

        # Strengere Alignment-Anforderung
        alignment = np.dot(ball_dir, goal_dir)
        if alignment < 0.85:  # War 0.7, jetzt 0.85 (fast perfekt)
            return 0.0, 0.0

        opps = self._opponents(player_idx)
        tor_free = True
        for opp in opps:
            opp_pos = self._opponent_ego_position(obs, player_idx, opp)
            if opp_pos is None:
                continue
            opp_vec = opp_pos - ball_pos
            proj = np.dot(opp_vec, goal_dir)
            if 0 < proj < goal_dist:
                perp = opp_vec - proj * goal_dir
                if self._norm(perp) < self.goal_width:
                    tor_free = False
                    break

        # Viel kleinerer Reward (nicht der Hauptfokus am Anfang!)
        reward = alignment * (0.5 if tor_free else 0.1)  # War 3.0/0.5
        phi = reward
        return reward, phi

    def _reward_blocking(self, obs, player_idx, team):
        """Ball fliegt Richtung eigenes Tor -> Abwehr belohnen."""
        ball_vel = self._ball_velocity_ego(obs, player_idx)
        speed = self._norm(ball_vel)
        if speed < self.shot_speed_threshold:
            return 0.0, 0.0

        ball_pos = self._get(obs, player_idx, "ball_ego_position")
        own_goal = self._get(obs, player_idx, "own_goal_mid")
        if ball_pos is None or own_goal is None:
            return 0.0, 0.0

        ball_pos = np.asarray(ball_pos, dtype=np.float32).flatten()[:3]
        own_goal = np.asarray(own_goal, dtype=np.float32).flatten()[:3]
        ball_dir = ball_vel / (speed + 1e-8)
        to_goal = own_goal - ball_pos
        goal_dist = self._norm(to_goal)
        if goal_dist < 1e-8:
            return 0.0, 0.0
        goal_dir = to_goal / goal_dist

        alignment = np.dot(ball_dir, goal_dir)
        if alignment < 0.7:
            return 0.0, 0.0

        player_pos = np.zeros(3, dtype=np.float32)
        player_vec = player_pos - ball_pos
        proj = np.dot(player_vec, goal_dir)
        perp = player_vec - proj * goal_dir
        between = 0.0 < proj < goal_dist and self._norm(perp) < self.goal_width + 0.5

        reward = 2.0 if between else 0.0
        dist_to_ball = self._norm(player_vec)
        prev_dist = (
            self._prev_player_dist_to_ball[player_idx]
            if self._prev_player_dist_to_ball is not None
            else dist_to_ball
        )
        delta = prev_dist - dist_to_ball
        reward += 0.5 * np.clip(delta, -1.0, 1.0)

        phi = reward
        return reward, phi

    def _reward_goalkeeping(self, obs, player_idx, team, ball_chaser):
        """
        Belohnt Positionierung zwischen Ball und eigenem Tor.
        WICHTIG: Spieler soll AUF DER LINIE Ball->Tor stehen, nicht zum Tor laufen!
        """
        ball_pos = self._get(obs, player_idx, "ball_ego_position")
        own_goal = self._get(obs, player_idx, "own_goal_mid")
        if ball_pos is None or own_goal is None:
            return 0.0, 0.0

        ball_pos = np.asarray(ball_pos, dtype=np.float32).flatten()[:3]
        own_goal = np.asarray(own_goal, dtype=np.float32).flatten()[:3]

        # Vektor von Ball zu Tor
        to_goal = own_goal - ball_pos
        goal_dist = self._norm(to_goal)
        if goal_dist < 1e-8:
            return 0.0, 0.0
        goal_dir = to_goal / goal_dist

        # Spielerposition (im eigenen Frame = 0,0,0)
        player_pos = np.zeros(3, dtype=np.float32)
        
        # Projektion des Spielers auf die Linie Ball->Tor
        player_vec = player_pos - ball_pos
        proj = np.dot(player_vec, goal_dir)  # Wie weit auf der Linie?
        perp = player_vec - proj * goal_dir  # Seitlicher Abstand
        
        # Ideale Position: 30-70% der Distanz von Ball zu Tor (nicht zu nah am Ball!)
        # Wenn Ball weit weg -> näher zum Tor
        # Wenn Ball nah -> näher zum Ball aber nicht zu nah
        ideal_t = np.clip(0.5 - 0.2 * (goal_dist / 20.0), 0.3, 0.7)
        ideal_pos = ball_pos + ideal_t * to_goal
        dist_to_ideal = self._norm(player_pos - ideal_pos)
        
        # Hauptreward: Auf der Linie sein (perpendicular distance)
        line_reward = -0.5 * np.clip(self._norm(perp) / 3.0, 0.0, 1.0)
        
        # Positionsreward: Richtige Distanz auf der Linie
        position_reward = -0.3 * np.clip(dist_to_ideal / 8.0, 0.0, 1.0)
        
        # Bonus: Zwischen Ball und Tor sein (0 < proj < goal_dist)
        between_bonus = 0.0
        if 0.0 < proj < goal_dist:
            between_bonus = 0.3
        
        reward = line_reward + position_reward + between_bonus
        phi = -0.2 * np.clip(dist_to_ideal / 8.0, 0.0, 1.0)
        
        return reward, phi

    # ------------------------------------------------------------------
    # Haupt-Routine
    # ------------------------------------------------------------------
    def _detect_possession(self, obs):
        """Wer hat Ballbesitz? (team, player_idx) oder (None, -1)."""
        best_dist = float("inf")
        owner = -1
        for p in range(self._num_players):
            d = self._distance_to_ball(obs, p)
            if d < best_dist:
                best_dist = d
                owner = p

        if best_dist > self.possession_radius or owner < 0:
            return None, -1
        return self._team_for_player(owner), owner

    def _compute_shaped_reward(self, obs, base_rewards):
        """Berechnet zustandsabhängige Branch-Rewards für alle Spieler."""
        if self._num_players is None:
            self._team_size, self._num_players = self._detect_team_size(obs)

        shaped = np.zeros(self._num_players, dtype=np.float32)
        phi_current = np.zeros(self._num_players, dtype=np.float32)

        possession_team, ball_owner = self._detect_possession(obs)
        self._current_marking_target_dist = {}

        for team in (0, 1):
            ball_chaser = self._find_ball_chaser(obs, team)
            assignments = self._compute_marking_assignments(obs, team)
            team_players = list(range(team * self._team_size, (team + 1) * self._team_size))

            for p in team_players:
                # === ROLLENBASIERTE REWARD-ZUWEISUNG ===
                # Nicht alle Spieler bekommen alle Rewards!
                
                r_rec, phi_rec = 0.0, 0.0
                r_mark, phi_mark = 0.0, 0.0
                r_pos, phi_pos = 0.0, 0.0
                r_att, phi_att = 0.0, 0.0
                r_shoot, phi_shoot = 0.0, 0.0
                r_block, phi_block = 0.0, 0.0
                r_keep, phi_keep = 0.0, 0.0
                
                # 1. BALL-CHASER (nächster Spieler zum Ball)
                if p == ball_chaser:
                    # Nur Recovery + ggf. Possession (wenn er Ball hat)
                    r_rec, phi_rec = self._reward_recovery(obs, p, ball_chaser)
                    if p == ball_owner:
                        r_pos, phi_pos = self._reward_possession(obs, p, ball_owner)
                        # Shooting wenn Ball schnell Richtung Tor
                        r_shoot, phi_shoot = self._reward_shooting(obs, p, team, ball_owner)
                
                # 2. BALL-OWNER (wenn nicht Ball-Chaser)
                elif p == ball_owner:
                    # Possession + Shooting
                    r_pos, phi_pos = self._reward_possession(obs, p, ball_owner)
                    r_shoot, phi_shoot = self._reward_shooting(obs, p, team, ball_owner)
                    r_att, phi_att = self._reward_attack_position(obs, p, ball_owner, assignments)
                
                # 3. ANDERE SPIELER (weder Chaser noch Owner)
                else:
                    # Marking / Positioning
                    r_mark, phi_mark = self._reward_marking(obs, p, ball_chaser, assignments.get(p))
                    r_att, phi_att = self._reward_attack_position(obs, p, ball_owner, assignments)
                    
                    # Goalkeeping NUR wenn defensiv (Ball fliegt aufs eigene Tor)
                    ball_vel = self._ball_velocity_ego(obs, p)
                    if self._norm(ball_vel) > self.shot_speed_threshold:
                        own_goal = self._get(obs, p, "own_goal_mid")
                        ball_pos = self._get(obs, p, "ball_ego_position")
                        if own_goal is not None and ball_pos is not None:
                            to_goal = np.asarray(own_goal).flatten() - np.asarray(ball_pos).flatten()
                            ball_dir = ball_vel / (self._norm(ball_vel) + 1e-8)
                            alignment = np.dot(ball_dir, to_goal) / (self._norm(to_goal) + 1e-8)
                            if alignment > 0.5:  # Ball fliegt aufs eigene Tor
                                r_keep, phi_keep = self._reward_goalkeeping(obs, p, team, ball_chaser)
                    
                    # Blocking wenn Ball aufs eigene Tor fliegt
                    r_block, phi_block = self._reward_blocking(obs, p, team)

                r_total = (
                    self.lambda_recovery * r_rec +
                    self.lambda_marking * r_mark +
                    self.lambda_possession * r_pos +
                    self.lambda_attack_pos * r_att +
                    self.lambda_shooting * r_shoot +
                    self.lambda_blocking * r_block +
                    self.lambda_goalkeeping * r_keep
                )

                phi_total = (
                    self.lambda_recovery * phi_rec +
                    self.lambda_marking * phi_mark +
                    self.lambda_possession * phi_pos +
                    self.lambda_attack_pos * phi_att +
                    self.lambda_shooting * phi_shoot +
                    self.lambda_blocking * phi_block +
                    self.lambda_goalkeeping * phi_keep
                )

                phi_current[p] = phi_total
                prev_phi = self._prev_phi[p] if self._prev_phi is not None else phi_total
                pbrs = self.gamma * phi_total - prev_phi

                shaped[p] = r_total + pbrs
                shaped[p] = np.clip(shaped[p], -2.0, 2.0)

                self._episode_branch_rewards["recovery"] += self.lambda_recovery * r_rec
                self._episode_branch_rewards["marking"] += self.lambda_marking * r_mark
                self._episode_branch_rewards["possession"] += self.lambda_possession * r_pos
                self._episode_branch_rewards["attack_pos"] += self.lambda_attack_pos * r_att
                self._episode_branch_rewards["shooting"] += self.lambda_shooting * r_shoot
                self._episode_branch_rewards["blocking"] += self.lambda_blocking * r_block
                self._episode_branch_rewards["goalkeeping"] += self.lambda_goalkeeping * r_keep

        self._prev_phi = phi_current
        self._prev_marking_target_dist = self._current_marking_target_dist.copy()
        self._prev_ball_to_goal = (
            self._distance_ball_to_opponent_goal(obs, ball_owner)
            if ball_owner >= 0 else None
        )
        self._prev_player_dist_to_ball = np.array([
            self._distance_to_ball(obs, p) for p in range(self._num_players)
        ], dtype=np.float32)
        self._last_branch_rewards = self._episode_branch_rewards.copy()

        return shaped

    # ------------------------------------------------------------------
    # dm_env API
    # ------------------------------------------------------------------
    def reset(self):
        self._reset_branch_stats()
        self._prev_phi = None
        self._prev_marking_target_dist = {}
        self._prev_ball_to_goal = None
        self._prev_player_dist_to_ball = None
        self._num_players = None
        self._team_size = None

        timestep = self.env.reset()
        self._team_size, self._num_players = self._detect_team_size(timestep.observation)
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

    def __getattr__(self, name):
        return getattr(self.env, name)

    # ------------------------------------------------------------------
    # Statistik-API für Training
    # ------------------------------------------------------------------
    def get_branch_rewards(self, reset=True):
        stats = self._episode_branch_rewards.copy()
        if reset:
            self._reset_branch_stats()
        return stats

    def get_last_branch_rewards(self):
        return self._last_branch_rewards.copy()


def make_env_with_dynamic_rewards_v2(
    team_size=2,
    seed=None,
    reward_scale=1.0,
    possession_radius=0.6,
    shot_speed_threshold=2.0,
    goal_width=2.0,
    time_limit=20.0,
    lambda_recovery=1.0,
    lambda_marking=1.0,
    lambda_possession=1.0,
    lambda_shooting=1.0,
    lambda_blocking=1.0,
    lambda_goalkeeping=1.0,
    lambda_attack_pos=0.5,
    **kwargs,
):
    """
    Erstellt Soccer-Environment mit Dynamic Scoring V2 Wrapper.
    
    Parameter:
        time_limit: Episoden-Dauer in Sekunden (Default: 20.0, erhöht für längere Spiele)
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
    return DynamicScoringWrapperV2(
        env,
        reward_scale=reward_scale,
        possession_radius=possession_radius,
        shot_speed_threshold=shot_speed_threshold,
        goal_width=goal_width,
        lambda_recovery=lambda_recovery,
        lambda_marking=lambda_marking,
        lambda_possession=lambda_possession,
        lambda_shooting=lambda_shooting,
        lambda_blocking=lambda_blocking,
        lambda_goalkeeping=lambda_goalkeeping,
        lambda_attack_pos=lambda_attack_pos,
        **kwargs,
    )
