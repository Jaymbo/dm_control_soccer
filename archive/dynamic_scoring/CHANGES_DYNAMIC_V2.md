# Änderungen: Umstellung auf Dynamic Scoring V2

## Zusammenfassung

Alle Optimierungs- und Worker-Skripte wurden von der alten **Dynamic Scoring V1** (bzw. Curriculum Learning) auf das neue **Dynamic Scoring V2** Reward-System umgestellt.

## Neue Dateien

| Datei | Beschreibung |
|-------|-------------|
| `env_wrapper_dynamic_v2.py` | Neuer Reward-Wrapper mit 7 Branches (ego-zentriert, für N vs N) |
| `train_mappo_dynamic_v2.py` | Trainingsscript für Dynamic Scoring V2 |
| `optimize_dynamic_v2.py` | Optuna Hyperparameter-Optimierung für V2 |
| `README_DYNAMIC_SCORING_V2.md` | Dokumentation für Dynamic Scoring V2 |
| `WORKER_SETUP_DYNAMIC_V2.md` | Worker-Setup-Anleitung für V2 |
| `CHANGES_DYNAMIC_V2.md` | Diese Datei |

## Geänderte Dateien

### 1. `worker_entrypoint.py`

**Änderungen:**
- Default Study Name: `soccer_dynamic_v1` → `soccer_dynamic_v2`
- Import: `train_mappo_dynamic` → `train_mappo_dynamic_v2`
- Hyperparameter-Space für Dynamic V2 aktualisiert:
  - Alte V1-Parameter: `lambda_recover`, `lambda_pursuit`, `lambda_possession`, `lambda_defense`
  - Neue V2-Parameter: `lambda_recovery`, `lambda_marking`, `lambda_possession`, `lambda_shooting`, `lambda_blocking`, `lambda_goalkeeping`, `lambda_attack_pos`

**Betroffene Funktionen:**
- `create_training_args()`: Neue Parameter für V2
- `objective()`: Importiert `train_mappo_dynamic_v2.train`

### 2. `optimize_curriculum.py` (unverändert)

Das alte Curriculum-Optimierungsskript bleibt erhalten für:
- Legacy-Training mit Phasen-Curriculum
- Vergleichsexperimente V1 vs V2

### 3. `optimize_dynamic_v2.py` (neu)

Neues Optimierungsskript spezifisch für Dynamic Scoring V2:
- Optimiert alle 7 Lambda-Parameter
- Search Space angepasst an V2-Branches
- Speichert Ergebnisse in `logs/optuna_dynamic_v2/`
- Datenbank: `optuna_dynamic_v2.db`

## Vergleich: V1 vs V2 Parameter

| V1 (Curriculum/Dynamic) | V2 (Dynamic Scoring V2) | Beschreibung |
|-------------------------|-------------------------|--------------|
| `lambda_recover` | `lambda_recovery` | Ball-Chaser läuft zum Ball |
| `lambda_pursuit` | ❌ entfernt | In V2 Teil von `recovery` + `marking` |
| `lambda_possession` | `lambda_possession` | Ball zum Tor bringen |
| `lambda_defense` | ❌ entfernt | Aufgeteilt in `marking`, `blocking`, `goalkeeping` |
| ❌ neu | `lambda_marking` | Gegner decken (Hungarian Algorithmus) |
| ❌ neu | `lambda_shooting` | Schuss aufs (freie) Tor |
| ❌ neu | `lambda_blocking` | Schuss abwehren |
| ❌ neu | `lambda_goalkeeping` | Zwischen Ball und Tor positionieren |
| ❌ neu | `lambda_attack_pos` | Freie Torsicht für Mitspieler |

## Migration: Von V1 auf V2

### 1. Training umstellen

**Alt (V1):**
```bash
python train_mappo_dynamic.py --num-episodes 1000
```

**Neu (V2):**
```bash
python train_mappo_dynamic_v2.py --num-episodes 1000
```

### 2. Worker umstellen

**Alt (V1):**
```bash
python worker_entrypoint.py \
  --study-name soccer_dynamic_v1 \
  --use-dynamic-rewards
```

**Neu (V2):**
```bash
python worker_entrypoint.py \
  --study-name soccer_dynamic_v2 \
  --use-dynamic-rewards
```

### 3. Optimierung umstellen

**Alt (V1/Curriculum):**
```bash
python optimize_curriculum.py --n-trials 50
```

**Neu (V2):**
```bash
python optimize_dynamic_v2.py --n-trials 50
```

## Vorteile von V2

1. **Ego-zentriert:** Funktioniert ohne globale Koordinaten (kompatibel mit DM Control)
2. **Skalierbar:** Unterstützt beliebige Teamgrößen (2v2, 3v3, 4v4, ...)
3. **Detaillierter:** 7 spezifische Branches statt 4 generischer
4. **Optimales Marking:** Hungarian Algorithmus für beste Zuweisungen
5. **Bessere Trennung:** Klare Aufgabentrennung (Recovery, Marking, Shooting, etc.)

## Abwärtskompatibilität

- `train_mappo_curriculum.py` bleibt unverändert für Curriculum-Training
- `optimize_curriculum.py` bleibt für Curriculum-Optimierung
- `worker_entrypoint.py` unterstützt beide Modi via `--use-dynamic-rewards` Flag

## Empfohlene Vorgehensweise

1. **Baseline testen:**
   ```bash
   python train_mappo_dynamic_v2.py --num-episodes 200 --viewer
   ```

2. **Hyperparameter optimieren:**
   ```bash
   python optimize_dynamic_v2.py --n-trials 50 --timeout 7200
   ```

3. **Distributed Training:**
   ```bash
   # Master
   docker-compose -f docker-compose.master.yml up -d
   
   # Worker (mehrere)
   python worker_entrypoint.py \
     --storage postgresql://... \
     --study-name soccer_dynamic_v2 \
     --use-dynamic-rewards \
     --infinite
   ```

4. **Ergebnisse analysieren:**
   ```bash
   tensorboard --logdir logs/optuna_dynamic_v2/tensorboard
   optuna-dashboard sqlite:///optuna_dynamic_v2.db
   ```

## Bekannte Einschränkungen

1. **Scipy-Abhängigkeit:** Hungarian Algorithmus benötigt `scipy` (optional, Fallback auf Greedy)
2. **Längere Trainingszeit:** V2 hat mehr Parameter → kann länger brauchen für Konvergenz
3. **Komplexere Rewards:** Mehr Branches → schwieriger zu debuggen

## Nächste Schritte

- [ ] Erste Optimierungsläufe mit `optimize_dynamic_v2.py`
- [ ] Vergleich V1 vs V2 Performance
- [ ] Dokumentation der besten Hyperparameter
- [ ] Ggf. Anpassung der Lambda-Gewichte basierend auf Ergebnissen
