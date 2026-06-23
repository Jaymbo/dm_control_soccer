# Dynamic Scoring V2 – Rollen-basiertes Reward-System

## Übersicht

Dynamic Scoring V2 ist ein zustandsbasiertes, rollenorientiertes Reward-System für
Multi-Agent Soccer mit **beliebiger Teamgröße N vs N** (2v2, 3v3, 4v4, ...).

Statt fixer Phasen (Curriculum) bekommt jeder Spieler zu jedem Zeitpunkt **eine
klare Rolle** zugewiesen, basierend auf der aktuellen Spielsituation:

| Branch | Wer? | Wann? | Ziel |
|--------|------|-------|------|
| **RECOVERY** | Ball-Chaser (nächster Spieler zum Ball) | Immer | Ball anlaufen |
| **MARKING** | Alle anderen eigenen Spieler | Immer | Gegner decken ODER Angriffsposition einnehmen |
| **POSSESSION** | Ball-Besitzer | Wenn im Besitz | Ball Richtung gegnerisches Tor |
| **ATTACK_POS** | Mitspieler auf Tor-Linie | Wenn eigener Ballbesitz | Freie Torsicht, offen zum Ball |
| **SHOOTING** | Alle | Ball fliegt schnell Richtung gegnerisches Tor | Schuss aufs (freie) Tor belohnen |
| **BLOCKING** | Alle | Ball fliegt schnell Richtung eigenes Tor | Abwehr belohnen |
| **GOALKEEPING** | Alle | Immer | Zwischen Ball und eigenem Tor positionieren |

## Kernideen

### 1. Ball-Chaser Zuweisung
Pro Team wird der Spieler ermittelt, der dem Ball am nächsten ist:
```python
ball_chaser = argmin(distance_to_ball[player] for player in team)
```
**Nur** dieser Spieler bekommt den RECOVERY-Reward für das Anlaufen des Balls.

### 2. Gegner-Marking mit Hungarian Algorithmus
Der ball-nächste Gegner ist implizit durch den Ball-Chaser "gedeckt". Die
verbleibenden eigenen Spieler werden den verbleibenden Gegnern **plus der
Tor-Linie** (Ball → gegnerisches Tor) optimal zugewiesen:

```python
# Kostenmatrix: quadratische Abstände zur Ziel-Linie
cost[i,j] = distance_squared(teammate[i], line(ball → target[j]))

# Hungarian Algorithmus minimiert Gesamtstrecke
assignments = linear_sum_assignment(cost_matrix)
```

Ziele können sein:
- `('goal', None)` → Linie Ball → gegnerisches Tor
- `('opponent', opp_idx)` → Linie Ball → spezifischer Gegner

### 3. Ego-zentrierte Berechnungen
Da DM Control Soccer **keine globalen Koordinaten** liefert, arbeiten alle
Berechnungen im ego-zentrierten Frame des jeweiligen Referenzspielers
(Ball-Chaser für Marking, eigener Frame für andere Rewards).

Verfügbare Observation-Keys:
- `ball_ego_position` – Ballposition relativ zum Spieler
- `teammate_<i>_ego_position` – Mitspieler i relativ zum Spieler
- `opponent_<i>_ego_position` – Gegner i relativ zum Spieler
- `opponent_goal_mid`, `own_goal_mid` – Torpositionen relativ zum Spieler

## Verwendung

### Training (2v2 Default)
```bash
python train_mappo_dynamic_v2.py --num-episodes 1000
```

### Training (3v3)
```bash
python train_mappo_dynamic_v2.py --num-episodes 1000 --team-size 3
```

### Wichtige Parameter
```bash
--team-size N              # Spieler pro Team (Default: 2)
--reward-scale FLOAT       # Globale Reward-Skalierung (Default: 1.0)
--possession-radius FLOAT  # Max. Abstand für Ballbesitz (Default: 0.6)
--shot-speed-threshold     # Min. Ballgeschwindigkeit für Schuss (Default: 2.0)

--lambda-recovery FLOAT    # Gewicht RECOVERY (Default: 1.0)
--lambda-marking FLOAT     # Gewicht MARKING (Default: 1.0)
--lambda-possession FLOAT  # Gewicht POSSESSION (Default: 1.0)
--lambda-shooting FLOAT    # Gewicht SHOOTING (Default: 1.0)
--lambda-blocking FLOAT    # Gewicht BLOCKING (Default: 1.0)
--lambda-goalkeeping FLOAT # Gewicht GOALKEEPING (Default: 1.0)
--lambda-attack-pos FLOAT  # Gewicht ATTACK_POS (Default: 0.5)
```

### Empfohlene Start-Konfiguration
```bash
python train_mappo_dynamic_v2.py \
  --num-episodes 1000 \
  --team-size 2 \
  --reward-scale 1.0 \
  --lambda-recovery 1.0 \
  --lambda-marking 1.0 \
  --lambda-possession 2.0 \
  --lambda-shooting 1.5 \
  --lambda-blocking 1.5 \
  --lambda-goalkeeping 0.5 \
  --lambda-attack-pos 0.5 \
  --episodes-per-batch 20
```

## TensorBoard Logging

Nach jedem Batch werden die Branch-Rewards geloggt:
```bash
tensorboard --logdir logs/soccer_mappo_dynamic_v2
```

Verfügbare Metriken:
- `Reward/branch_<name>` – Summe pro Branch
- `Reward/branch_<name>_per_episode` – Durchschnitt pro Episode
- `Reward/avg_100` – Gesamtreward (letzte 100 Episoden)
- `Loss/policy`, `Loss/value`, `Loss/entropy` – Trainingsverluste

## Reward-Details

### RECOVERY (nur Ball-Chaser)
```python
reward = 0.5 * delta(distance_to_ball) + 0.2 (wenn sehr nah am Ball)
```

### MARKING (alle außer Ball-Chaser)
```python
reward = -0.3 * dist_to_line      # Nah an der Linie sein
       + 0.5 * delta(dist_to_target)  # Richtung Ziel laufen
       + 0.3 (wenn zwischen Ball und Ziel)
```

### POSSESSION (nur Ball-Besitzer)
```python
reward = 2.0 * delta(ball_to_goal_distance) + 0.3 (wenn Ballbesitz)
```

### ATTACK_POS (Mitspieler auf Tor-Linie)
```python
reward = 0.5 * visible_goal_width  # Freie Sicht aufs Tor
       - 0.05 * |dist_to_goal - 6.0|  # Optimale Distanz
       + 0.3 (wenn Ball nicht verdeckt)
```

### SHOOTING (alle, wenn Ball schnell Richtung Tor)
```python
reward = alignment * (3.0 if tor_free else 0.5)
```

### BLOCKING (alle, wenn Ball schnell Richtung eigenes Tor)
```python
reward = 2.0 (wenn zwischen Ball und Tor)
       + 0.5 * delta(dist_to_ball)  # Näher zum Ball laufen
```

### GOALKEEPING (alle)
```python
reward = -0.2 * dist_to_line  # Auf der Linie Ball→Tor
       - 0.3 * dist_to_ideal_position  # Ideale Position (70% der Distanz)
```

## Unterschiede zu V1 (Dynamic Scoring)

| Feature | V1 | V2 |
|---------|----|-----|
| Teamgröße | Fix 2v2 | Beliebig N vs N |
| Marking | Presser/Anchor Rollen | Hungarian Algorithmus |
| Koordinaten | Teilweise global angenommen | Komplett ego-zentriert |
| Branches | 4 (recovery, pursuit, possession, defense) | 7 (detaillierter) |
| Schuss/Block | Implizit in possession/defense | Explizite Branches |

## Troubleshooting

### "Scipy nicht verfügbar"
Der Hungarian Algorithmus fällt automatisch auf Greedy-Zuweisung zurück.
Für optimale Ergebnisse:
```bash
pip install scipy
```

### "Marking-Rewards sind 0"
- Prüfe, ob `team_size` korrekt gesetzt ist
- Bei 2v2 gibt es nur 1 Marking-Spieler pro Team (der nicht Ball-Chaser ist)

### "Training divergiert"
- Reduziere `--reward-scale` auf 0.5
- Erhöhe `--entropy-coef` auf 0.1
- Reduziere `--lambda-possession` und `--lambda-shooting`

### "Agenten laufen nicht zum Ball"
- Erhöhe `--lambda-recovery` auf 1.5–2.0
- Reduziere `--possession-radius` auf 0.4

## Architektur

```
┌─────────────────────────────────────────────────────────────┐
│  DynamicScoringWrapperV2                                     │
├─────────────────────────────────────────────────────────────┤
│  _compute_shaped_reward()                                    │
│    ├─ _detect_possession()                                   │
│    ├─ Für jedes Team:                                        │
│    │   ├─ _find_ball_chaser()                                │
│    │   ├─ _compute_marking_assignments() (Hungarian)         │
│    │   └─ Für jeden Spieler:                                 │
│    │       ├─ _reward_recovery()                             │
│    │       ├─ _reward_marking()                              │
│    │       ├─ _reward_possession()                           │
│    │       ├─ _reward_attack_position()                      │
│    │       ├─ _reward_shooting()                             │
│    │       ├─ _reward_blocking()                             │
│    │       └─ _reward_goalkeeping()                          │
│    └─ PBRS: γΦ(s') - Φ(s)                                    │
└─────────────────────────────────────────────────────────────┘
```

## Nächste Schritte

1. **Baseline testen:**
   ```bash
   python train_mappo_dynamic_v2.py --num-episodes 100 --viewer
   ```

2. **Branch-Verteilung analysieren:**
   ```bash
   tensorboard --logdir logs/soccer_mappo_dynamic_v2
   ```

3. **Hyperparameter optimieren:**
   - Für schnelleres Ball-Anlaufen: `--lambda-recovery 1.5`
   - Für besseres Marking: `--lambda-marking 1.5`
   - Für mehr Tore: `--lambda-possession 2.0 --lambda-shooting 1.5`

4. **Auf 3v3 skalieren:**
   ```bash
   python train_mappo_dynamic_v2.py --num-episodes 2000 --team-size 3
   ```
