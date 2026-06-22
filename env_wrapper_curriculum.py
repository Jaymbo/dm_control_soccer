"""
Curriculum Learning Wrapper für DM Control Soccer.

Phasen (automatisch oder manuell gesteuert):
  0 - MOVE:      Agenten lernen, sich aufrecht und schnell zu bewegen.
  1 - APPROACH:  Agenten lernen, sich dem Ball zu nähern.
  2 - DRIBBLE:   Agenten lernen, den Ball zu kontrollen und Richtung gegnerisches
                 Tor zu bewegen.
  3 - SHOOT:     Agenten lernen, Tore zu schießen (volles Soccer-Objective).

Jede Phase setzt andere Reward-Schwerpunkte und kann auf andere Weise prüfen,
ob sie gemeistert ist (z. B. durchschnittliche Bewegung, Ballnähe, Tore).
"""
import numpy as np
from env_wrapper_optimized import SoccerRewardWrapperOptimized


class SoccerCurriculumWrapper(SoccerRewardWrapperOptimized):
    """
    Wrapper mit automatischem Curriculum.

    Parameter:
      curriculum_phase:      Startphase (0..3)
      auto_advance:          Ob automatisch aufgestuft werden soll
      phase_episodes:        Minimale Episoden pro Phase (nur bei auto_advance)
      phase_success_rate:    Anteil der letzten Episoden, die Ziel erfüllen müssen
      progress_callback:     optional f(phase) bei Level-Up
    """

    PHASE_NAMES = {
        0: "MOVE",
        1: "APPROACH",
        2: "DRIBBLE",
        3: "SHOOT",
    }

    # Reward-Konfigurationen pro Phase.
    # Werte sind bewusst extrem, damit das jeweilige Ziel dominiert.
    PHASE_CONFIGS = {
        0: {  # MOVE
            "movement_bonus": 3.0,
            "idle_penalty": 1.0,
            "fall_penalty": 1.0,
            "ball_proximity_weight": 0.0,
            "moving_to_ball_weight": 0.0,
            "ball_to_goal_weight": 0.0,
            "possession_bonus": 0.0,
            "shot_to_goal_weight": 0.0,
            "reward_scale": 1.0,
        },
        1: {  # APPROACH
            "movement_bonus": 0.5,
            "idle_penalty": 0.3,
            "fall_penalty": 0.5,
            "ball_proximity_weight": 1.5,
            "moving_to_ball_weight": 2.0,
            "ball_to_goal_weight": 0.0,
            "possession_bonus": 0.0,
            "shot_to_goal_weight": 0.0,
            "reward_scale": 1.0,
        },
        2: {  # DRIBBLE
            "movement_bonus": 0.3,
            "idle_penalty": 0.2,
            "fall_penalty": 0.5,
            "ball_proximity_weight": 0.5,
            "moving_to_ball_weight": 1.0,
            "ball_to_goal_weight": 2.5,
            "possession_bonus": 2.0,
            "shot_to_goal_weight": 0.5,
            "reward_scale": 1.0,
        },
        3: {  # SHOOT (volles optimiertes Shaping)
            "movement_bonus": 0.3,
            "idle_penalty": 0.2,
            "fall_penalty": 0.5,
            "ball_proximity_weight": 0.1,
            "moving_to_ball_weight": 0.8,
            "ball_to_goal_weight": 2.0,
            "possession_bonus": 0.5,
            "shot_to_goal_weight": 1.5,
            "reward_scale": 1.0,
        },
    }

    # KPI-Schwellen für automatisches Aufsteigen.
    # Werte sind bewusst niedrig für sanften Einstieg.
    PHASE_TARGETS = {
        0: {
            "min_avg_steps": 150,       # Agent bleibt nicht sofort liegen
            "min_avg_movement_per_agent": 0.02,  # ~1cm Bewegung pro Agent pro Step
        },
        1: {
            "min_avg_ball_dist": 3.0,   # im Mittel < 3m zum Ball
            "min_avg_moving_to_ball": -0.1,  # darf auch leicht negativ sein am Anfang
        },
        2: {
            "min_avg_possession_time": 2,  # Ballbesitz für 2 Steps
            "min_avg_ball_to_goal": 6.0,   # Ball im Mittel < 6m zum gegnerischen Tor
        },
        3: {
            "min_avg_goals": 0.02,      # mindestens 1 Tor alle 50 Episoden
        },
    }

    def __init__(self, env,
                 curriculum_phase=0,
                 auto_advance=True,
                 phase_episodes=40,
                 phase_success_rate=0.6,
                 progress_callback=None,
                 **base_kwargs):
        # Entferne globale Reward-Parameter, da sie von der Phase diktiert werden
        phase_kwargs = base_kwargs.copy()
        for key in ["movement_bonus", "idle_penalty", "fall_penalty",
                    "ball_proximity_weight", "moving_to_ball_weight",
                    "ball_to_goal_weight", "possession_bonus",
                    "shot_to_goal_weight", "reward_scale"]:
            phase_kwargs.pop(key, None)

        super().__init__(env, **phase_kwargs)
        self._base_reward_scale = base_kwargs.get("reward_scale", 1.0)

        self.curriculum_phase = max(0, min(3, int(curriculum_phase)))
        self.auto_advance = auto_advance
        self.phase_episodes = phase_episodes
        self.phase_success_rate = phase_success_rate
        self.progress_callback = progress_callback

        self._apply_phase_config()

        # Episode-Statistiken für KPIs
        self._episode_stats = []
        self._current_episode_stats = self._new_stats()

    def _new_stats(self):
        return {
            "steps": 0,
            "movement_sum": 0.0,
            "ball_dist_sum": 0.0,
            "moving_to_ball_sum": 0.0,
            "ball_to_goal_sum": 0.0,
            "possession_steps": 0,
            "goals": 0,
        }

    def _apply_phase_config(self):
        cfg = self.PHASE_CONFIGS[self.curriculum_phase]
        for key, value in cfg.items():
            setattr(self, key, value)
        print(f"[Curriculum] Phase {self.curriculum_phase}: {self.PHASE_NAMES[self.curriculum_phase]}")

    def _player_velocity(self, obs, player_idx):
        """
        Gibt die eigene Geschwindigkeit zurück (sensors_velocimeter).
        Da ego-Koordinaten den Spieler immer bei (0,0,0) haben,
        nutzen wir Velocity für Bewegung.
        """
        vel = self._get(obs, player_idx, 'sensors_velocimeter')
        if vel is None:
            return np.zeros(3)
        return np.asarray(vel).flatten()

    def _compute_shaped_reward(self, obs, base_rewards):
        """
        Berechnet Rewards NUR für das aktuelle Phasen-Ziel.
        Jede Phase ignoriert irrelevante Signale komplett.
        
        WICHTIG: Ego-Koordinaten haben Spieler immer bei (0,0,0).
        Bewegung wird über sensors_velocimeter (Velocity) gemessen.
        """
        shaped = np.zeros(self._num_players, dtype=np.float32)
        phase = self.curriculum_phase

        # Stats immer tracken für KPIs (aber Rewards nur phasen-spezifisch)
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

            # Velocity für Bewegung (da ego-Position immer 0)
            velocity = self._player_velocity(obs, p)
            speed = self._norm(velocity)  # Geschwindigkeit in m/s

            # === PHASE 0: MOVE ===
            # Nur: Bewegung belohnen, Sturz bestrafen. KEIN Ball-Bezug!
            if phase == 0:
                # Bewegung belohnen (basierend auf Geschwindigkeit)
                shaped[p] += self.movement_bonus * np.clip(speed / 2.0, 0.0, 1.0)
                self._current_episode_stats["movement_sum"] += float(speed)

                # Sturz bestrafen (body_height < 0.5 = gefallen)
                body_height = self._get(obs, p, 'body_height')
                if body_height is not None:
                    body_height_val = np.asarray(body_height).flatten()[0]
                    if body_height_val < 0.5:
                        shaped[p] -= self.fall_penalty

                # Idle bestrafen (speed < 0.1 m/s = fast stehend)
                if speed < 0.1:
                    shaped[p] -= self.idle_penalty

            # === PHASE 1: APPROACH ===
            # Nur: Zum Ball laufen. Bewegung sekundär.
            elif phase == 1:
                # Annäherung an Ball (Hauptziel)
                if self._prev_ball_dist is not None:
                    delta = self._prev_ball_dist[team] - ball_dist
                    shaped[p] += self.moving_to_ball_weight * np.clip(delta, -1.0, 1.0)
                    self._current_episode_stats["moving_to_ball_sum"] += float(delta)

                # Ballnähe-Bonus (sekundär)
                shaped[p] += self.ball_proximity_weight * np.clip(1.0 - ball_dist / 5.0, 0.0, 1.0)
                self._current_episode_stats["ball_dist_sum"] += float(ball_dist)

                # Leichte Bewegungsbasis (Velocity)
                shaped[p] += self.movement_bonus * np.clip(speed / 2.0, 0.0, 1.0)
                self._current_episode_stats["movement_sum"] += float(speed)

                # Sturz
                body_height = self._get(obs, p, 'body_height')
                if body_height is not None:
                    body_height_val = np.asarray(body_height).flatten()[0]
                    if body_height_val < 0.5:
                        shaped[p] -= self.fall_penalty

            # === PHASE 2: DRIBBLE ===
            # Nur: Ball Richtung Tor bewegen + Besitz
            elif phase == 2:
                # Ball Richtung Tor (Hauptziel)
                if self._prev_ball_to_goal is not None:
                    delta = self._prev_ball_to_goal[team] - ball_to_goal
                    shaped[p] += self.ball_to_goal_weight * np.clip(delta, -1.0, 1.0)
                    self._current_episode_stats["ball_to_goal_sum"] += float(ball_to_goal)

                # Ballbesitz (Hauptziel)
                if self._prev_ball_to_goal is not None and ball_dist < 0.5 and ball_to_goal < self._prev_ball_to_goal[team]:
                    shaped[p] += self.possession_bonus
                    self._current_episode_stats["possession_steps"] += 1

                # Annäherung (sekundär)
                if self._prev_ball_dist is not None:
                    delta = self._prev_ball_dist[team] - ball_dist
                    shaped[p] += self.moving_to_ball_weight * np.clip(delta, -1.0, 1.0)
                    self._current_episode_stats["moving_to_ball_sum"] += float(delta)

                # Ballnähe
                shaped[p] += self.ball_proximity_weight * np.clip(1.0 - ball_dist / 5.0, 0.0, 1.0)
                self._current_episode_stats["ball_dist_sum"] += float(ball_dist)

                # Sturz
                body_height = self._get(obs, p, 'body_height')
                if body_height is not None:
                    body_height_val = np.asarray(body_height).flatten()[0]
                    if body_height_val < 0.5:
                        shaped[p] -= self.fall_penalty

            # === PHASE 3: SHOOT ===
            # Volles Soccer-Reward (alle Signale)
            elif phase == 3:
                # Annäherung
                if self._prev_ball_dist is not None:
                    delta = self._prev_ball_dist[team] - ball_dist
                    shaped[p] += self.moving_to_ball_weight * np.clip(delta, -1.0, 1.0)
                    self._current_episode_stats["moving_to_ball_sum"] += float(delta)

                # Ballnähe
                shaped[p] += self.ball_proximity_weight * np.clip(1.0 - ball_dist / 5.0, 0.0, 1.0)
                self._current_episode_stats["ball_dist_sum"] += float(ball_dist)

                # Bewegung (Velocity)
                shaped[p] += self.movement_bonus * np.clip(speed / 2.0, 0.0, 1.0)
                self._current_episode_stats["movement_sum"] += float(speed)

                # Ball Richtung Tor
                if self._prev_ball_to_goal is not None:
                    delta = self._prev_ball_to_goal[team] - ball_to_goal
                    shaped[p] += self.ball_to_goal_weight * np.clip(delta, -1.0, 1.0)
                    self._current_episode_stats["ball_to_goal_sum"] += float(ball_to_goal)

                # Ballbesitz
                if self._prev_ball_to_goal is not None and ball_dist < 0.5 and ball_to_goal < self._prev_ball_to_goal[team]:
                    shaped[p] += self.possession_bonus
                    self._current_episode_stats["possession_steps"] += 1

                # Schuss
                if self._prev_ball_to_goal is not None:
                    delta = self._prev_ball_to_goal[team] - ball_to_goal
                    shaped[p] += self.shot_to_goal_weight * np.clip(delta, -1.0, 2.0)

                # Sturz
                body_height = self._get(obs, p, 'body_height')
                if body_height is not None:
                    body_height_val = np.asarray(body_height).flatten()[0]
                    if body_height_val < 0.5:
                        shaped[p] -= self.fall_penalty

                # Idle (nur wenn weit vom Ball)
                if ball_dist > 1.0 and speed < 0.1:
                    shaped[p] -= self.idle_penalty

        # Stats updaten
        self._prev_ball_dist = ball_dist_team
        self._prev_ball_to_goal = ball_to_goal_team
        self._current_episode_stats["steps"] += 1

        return shaped

    def step(self, actions):
        timestep = self.env.step(actions)
        base_rewards = np.asarray(timestep.reward, dtype=np.float32)

        # Zähle Tore als große Base-Reward-Spitzen (Heim + Auswärts)
        if base_rewards.max() > 0.5:
            self._current_episode_stats["goals"] += 1

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

    def get_debug_stats(self):
        """Gibt aktuelle Debug-Stats für Phase 0."""
        if len(self._episode_stats) == 0:
            return {}
        last = self._episode_stats[-1]
        steps = max(last["steps"], 1)
        return {
            "steps": last["steps"],
            "movement_sum": last["movement_sum"],
            "movement_per_step": last["movement_sum"] / steps,
            "movement_per_agent_per_step": last["movement_sum"] / steps / 4,
            "total_reward_shaped": last.get("reward_shaped_sum", 0.0),
        }

    def reset(self):
        self._episode_stats.append(self._current_episode_stats)
        self._current_episode_stats = self._new_stats()
        return super().reset()

    def evaluate_phase_progress(self):
        """
        Prüft, ob die letzten phase_episodes Episoden die Ziele der aktuellen
        Phase erreicht haben. Gibt True zurück, wenn aufgestiegen werden soll.
        """
        if not self.auto_advance or self.curriculum_phase >= 3:
            return False

        recent = self._episode_stats[-self.phase_episodes:]
        if len(recent) < self.phase_episodes:
            return False

        target = self.PHASE_TARGETS[self.curriculum_phase]
        successes = 0
        
        # Debug: Durchschnitte berechnen
        avg_steps = np.mean([s["steps"] for s in recent])
        avg_movement = np.mean([s["movement_sum"] / max(s["steps"], 1) / 4 for s in recent])  # /4 für 4 Agenten
        avg_ball_dist = np.mean([s["ball_dist_sum"] / max(s["steps"], 1) for s in recent])
        
        for stats in recent:
            ok = True
            steps = max(stats["steps"], 1)
            num_agents = 4  # Bewegung wird pro Agent gesammelt

            if "min_avg_steps" in target and stats["steps"] < target["min_avg_steps"]:
                ok = False
            # Bewegung pro Agent = movement_sum / steps / 4
            if "min_avg_movement_per_agent" in target and (stats["movement_sum"] / steps / num_agents) < target["min_avg_movement_per_agent"]:
                ok = False
            if "min_avg_ball_dist" in target and (stats["ball_dist_sum"] / steps) > target["min_avg_ball_dist"]:
                ok = False
            if "min_avg_moving_to_ball" in target and (stats["moving_to_ball_sum"] / steps) < target["min_avg_moving_to_ball"]:
                ok = False
            if "min_avg_possession_time" in target and (stats["possession_steps"] / max(stats["steps"], 1)) < target["min_avg_possession_time"]:
                ok = False
            if "min_avg_ball_to_goal" in target and (stats["ball_to_goal_sum"] / steps) > target["min_avg_ball_to_goal"]:
                ok = False
            if "min_avg_goals" in target and (stats["goals"] / max(stats["steps"], 1)) < target["min_avg_goals"]:
                ok = False

            if ok:
                successes += 1

        success_ratio = successes / len(recent)
        print(f"[Curriculum] Phase {self.curriculum_phase} success: {success_ratio:.2f} "
              f"(need {self.phase_success_rate}) | "
              f"avg_steps={avg_steps:.0f}, avg_movement={avg_movement:.4f}, avg_ball_dist={avg_ball_dist:.2f}")
        return success_ratio >= self.phase_success_rate

    def advance_phase(self):
        """Hebt die Schwierigkeit an, falls noch möglich."""
        if self.curriculum_phase < 3:
            self.curriculum_phase += 1
            self._apply_phase_config()
            self._episode_stats.clear()
            if self.progress_callback is not None:
                self.progress_callback(self.curriculum_phase)
            return True
        return False

    @property
    def phase_name(self):
        return self.PHASE_NAMES[self.curriculum_phase]


def make_env_with_curriculum(seed=None, curriculum_phase=0, auto_advance=True,
                             phase_episodes=40, phase_success_rate=0.6,
                             progress_callback=None, **base_kwargs):
    """Erstellt Soccer-Environment mit Curriculum-Learning Wrapper."""
    from dm_control.locomotion import soccer as dm_soccer
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
    return SoccerCurriculumWrapper(
        env,
        curriculum_phase=curriculum_phase,
        auto_advance=auto_advance,
        phase_episodes=phase_episodes,
        phase_success_rate=phase_success_rate,
        progress_callback=progress_callback,
        **base_kwargs,
    )
