# Dynamic Scoring für MAPPO Soccer

Diese Datei beschreibt den neuen zustandsbasierten Reward-Switcher, der das
alte Phasen-Curriculum ersetzt. Das Curriculum ist damit **immer aktiv** – der
Agent bekommt zu jedem Zeitpunkt genau das Verhalten belohnt, das gerade am
wichtigsten ist.

## Idee

Statt harter Phasen (MOVE → APPROACH → DRIBBLE → SHOOT) entscheidet sich der
Reward dynamisch pro Spieler und pro Zeitschritt. Es gibt vier Branches:

| Branch | Priorität | Wann aktiv? | Ziel |
|--------|-----------|-------------|------|
| **RECOVERY** | 1 (höchste) | Agent gestürzt | Wieder aufstehen |
| **PURSUIT** | 2 | Ball frei | Ball anlaufen + antizipieren |
| **POSSESSION** | 3 | Eigener Team hat Ball | Ball Richtung gegnerisches Tor |
| **DEFENSE** | 4 | Gegner hat Ball | Verteidigen (Presser / Anchor) |

## Branch-Tracking

Nach jedem Batch werden die aufsummierten Rewards pro Branch geloggt:

```text
Reward/branch_recovery
Reward/branch_pursuit
Reward/branch_possession
Reward/branch_defense
Reward/branch_<name>_per_episode
```

Damit siehst du im TensorBoard, welche Fähigkeiten gerade gelernt werden. Am
Anfang dominieren typischerweise **recovery** und **pursuit**; mit Fortschritt
steigen **possession** und **defense**.

## Lokales Training

```bash
python train_mappo_dynamic.py --num-episodes 1000
```

Wichtige Parameter:

```bash
--reward-scale         # Globaler Skalierungsfaktor (default 1.0)
--possession-radius    # Max Ballabstand für Ballbesitz (default 0.6)
--goal-threshold       # Torlinien-Distanz für Shot-Detection (default 6.0)
--lambda-recover       # RECOVERY-Gewicht (default 1.0)
--lambda-pursuit       # PURSUIT-Gewicht (default 1.0)
--lambda-possession    # POSSESSION-Gewicht (default 1.0)
--lambda-defense       # DEFENSE-Gewicht (default 1.0)
```

Empfohlene Startwerte für schnelles Lernen:

```bash
python train_mappo_dynamic.py \
  --num-episodes 1000 \
  --reward-scale 1.0 \
  --lambda-recover 1.0 \
  --lambda-pursuit 1.5 \
  --lambda-possession 2.0 \
  --lambda-defense 1.0 \
  --episodes-per-batch 20
```

## TensorBoard Branch-Verteilung

```bash
tensorboard --logdir logs/soccer_mappo_dynamic
```

Relevante Tags:

- `Reward/branch_*` – Summe pro Branch nach jedem Batch
- `Reward/branch_*_per_episode` – Durchschnitt pro Episode
- `Reward/avg_100` – Gesamtreward über 100 Episoden

## Docker Master-Slave System

### 1. Master starten (PostgreSQL + Optuna Dashboard)

```bash
# .env anpassen (siehe README_DISTRIBUTED.md)
docker-compose -f docker-compose.master.yml up -d
```

### 2. Worker mit Dynamic Scoring

```bash
python worker_entrypoint.py \
  --storage postgresql://optuna:PASSWORD@host:5433/optuna_db \
  --study-name soccer_dynamic_v1 \
  --use-dynamic-rewards \
  --infinite \
  --worker-id gpu-01
```

Oder per Docker Compose (`docker-compose.worker.yml`):

```bash
# .env im Worker-Verzeichnis
OPTUNA_STORAGE=postgresql://optuna:PASSWORD@host:5433/optuna_db
OPTUNA_STUDY_NAME=soccer_dynamic_v1
OPTUNA_USE_DYNAMIC_REWARDS=true
OPTUNA_N_TRIALS=100
```

```bash
docker-compose -f docker-compose.worker.yml up -d
```

### 3. Dashboard

```bash
optuna-dashboard postgresql://optuna:PASSWORD@host:5433/optuna_db
```

bzw. `http://localhost:8080` wenn du Docker Compose Master nutzt.

## Hyperparameter-Search Space

Der Worker optimiert folgende Werte:

- `episodes_per_batch`: 10, 20, 40
- `ppo_epochs`: 4, 6, 8, 10
- `lr`: 1e-5 bis 5e-4 (log)
- `entropy_coef`: 0.01 bis 0.1 (log)
- `entropy_decay`: 0.90 bis 0.99
- `reward_scale`: 0.5 bis 2.0
- `possession_radius`: 0.4 bis 1.0
- `goal_threshold`: 4.0 bis 8.0
- `lambda_recover`: 0.5 bis 2.0
- `lambda_pursuit`: 0.5 bis 2.0
- `lambda_possession`: 0.5 bis 2.0
- `lambda_defense`: 0.5 bis 2.0

## Troubleshooting

### Worker startet nicht

Stelle sicher, dass `env_wrapper_dynamic.py` und `train_mappo_dynamic.py`
fehlerfrei importierbar sind:

```bash
python -m py_compile env_wrapper_dynamic.py train_mappo_dynamic.py worker_entrypoint.py
```

### Branch-Rewards sind alle 0

- Prüfe, ob `reward_scale` korrekt gesetzt ist
- Prüfe im TensorBoard, ob `branch_*` geloggt werden
- Stelle sicher, dass `env_wrapper_dynamic.py` tatsächlich verwendet wird

### Training divergiert

- Reduziere `reward_scale`
- Reduziere `lambda_possession` / `lambda_pursuit`
- Erhöhe `entropy_coef`

## Unterschied zum alten Curriculum

| | Curriculum | Dynamic Scoring |
|---|---|---|
| Aktivierung | Phasen (0→3) | Immer aktiv |
| Reward | Eine Phase dominiert | State-conditioned Branch-Switching |
| Tracking | Phasen-KPIs | Branch-Rewards pro Batch |
| Ziel | Schrittweise Lernen | Gleichzeitiges Lernen aller Skills |

## Next Steps

1. `python train_mappo_dynamic.py --num-episodes 100` testen
2. TensorBoard öffnen und Branch-Verteilung prüfen
3. Docker Worker mit `--use-dynamic-rewards` starten
4. Beste Hyperparameter aus Optuna in finalem Training nutzen
