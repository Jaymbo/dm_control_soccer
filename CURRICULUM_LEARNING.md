# Curriculum Learning für MAPPO Soccer

## Problem
Das Standard-MAPPO-Training lernt zwar Bewegung, aber Agenten finden nicht systematisch zum Ball oder Tor. **Curriculum Learning** löst dies durch schrittweises Lernen.

## Lösung: 4-Phasen-Curriculum

| Phase | Name | Ziel | Rewards fokussiert auf |
|-------|------|------|------------------------|
| 0 | **MOVE** | Aufrecht bewegen | Bewegung (3.0), Idle-Strafe (1.0) |
| 1 | **APPROACH** | Zum Ball laufen | Ballnähe (1.5), Annäherung (2.0) |
| 2 | **DRIBBLE** | Ball Richtung Tor | Ball-zu-Tor (2.5), Besitz (2.0) |
| 3 | **SHOOT** | Tore schießen | Vollständiges Soccer-Reward |

## Dateien

- `env_wrapper_curriculum.py` - Curriculum-Wrapper mit automatischem Phase-Upgrade
- `train_mappo_curriculum.py` - Training-Skript mit Curriculum-Logik
- `test_mappo_curriculum.py` - Evaluation & Viewer

## Usage

### Training starten (ab Phase 0)
```bash
python train_mappo_curriculum.py --num-episodes 1000 --start-phase 0 --auto-advance
```

### Training ab bestimmter Phase
```bash
# Direkt mit Ballverfolgung starten (Phase 1)
python train_mappo_curriculum.py --num-episodes 800 --start-phase 1

# Nur Torschuss trainieren (Phase 3)
python train_mappo_curriculum.py --num-episodes 500 --start-phase 3
```

### Wichtige Parameter
```bash
--phase-episodes 40        # Min. Episoden pro Phase vor Evaluation
--phase-success-rate 0.6   # 60% Erfolg für Phase-Upgrade nötig
--auto-advance             # Automatisches Upgrade (default: an)
--no-auto-advance          # Manuelles Steuern der Phase
--start-phase 0|1|2|3      # Startphase wählen
```

### Modell evaluieren
```bash
python test_mappo_curriculum.py --checkpoint logs/soccer_mappo_curriculum/best_agent.pt --eval-episodes 20

# Mit Viewer
python test_mappo_curriculum.py --checkpoint logs/soccer_mappo_curriculum/best_agent.pt --viewer
```

## Wie es funktioniert

1. **Phase 0 (MOVE):** Agenten erhalten hohe Rewards für Bewegung und Strafen für Stillstand. Sie lernen, aufrecht zu laufen.

2. **Auto-Upgrade:** Nach 40 Episoden wird geprüft, ob ≥60% der Episoden die KPIs erfüllen (z. B. durchschnittliche Bewegung > 0.15).

3. **Phase 1 (APPROACH):** Reward-Fokus wechselt zu Ballnähe. Agenten, die sich bewegen können, lernen nun, zum Ball zu laufen.

4. **Weiter bis Phase 3:** Jede Phase baut auf der vorherigen auf.

## KPIs pro Phase

| Phase | Erfolgskriterien |
|-------|------------------|
| MOVE | ≥250 Steps, Bewegung > 0.15/Step |
| APPROACH | Ball-Distanz < 2.0m, Annäherung > 0.05/Step |
| DRIBBLE | Ballbesitz ≥5 Steps, Ball-zu-Tor < 4.0m |
| SHOOT | ≥0.05 Tore/Episode (1 Tor alle 20 Episoden) |

## Empfohlenes Training

```bash
# Vollständiges Curriculum (ca. 800-1200 Episoden)
python train_mappo_curriculum.py \
  --num-episodes 1200 \
  --phase-episodes 50 \
  --phase-success-rate 0.5 \
  --log-dir logs/soccer_curriculum_full

# Nur Ballverfolgung (schnell, ca. 300 Episoden)
python train_mappo_curriculum.py \
  --num-episodes 300 \
  --start-phase 1 \
  --phase-episodes 30
```

## Logs & Checkpoints

- `logs/soccer_mappo_curriculum/best_agent.pt` - Bestes Modell (auto-gespeichert)
- `logs/soccer_mappo_curriculum/checkpoint_phase{N}.pt` - Bei Phasenwechsel
- `logs/soccer_mappo_curriculum/checkpoint_ep{N}.pt` - Alle 100 Episoden

## TensorBoard
```bash
tensorboard --logdir logs/soccer_mappo_curriculum
```

## Tipps

1. **Niedrigere `phase-success-rate` (0.4-0.5):** Schnellere Phasenwechsel, aber weniger stabil
2. **Höhere `phase-episodes` (60-80):** Mehr Zeit zum Lernen pro Phase
3. **Start bei Phase 1:** Wenn Bewegung schon funktioniert, spart ~100 Episoden
4. **Entropy anpassen:** `--entropy-coef 0.8` für mehr Exploration in frühen Phasen
