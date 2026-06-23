# Worker Setup für Dynamic Scoring V2

## Übersicht

Dieses Dokument beschreibt die Einrichtung von Optuna Workern für die **Dynamic Scoring V2** Hyperparameter-Optimierung.

## Änderungen gegenüber V1

| Feature | V1 (Curriculum) | V2 (Dynamic Scoring) |
|---------|-----------------|----------------------|
| Study Name | `soccer_dynamic_v1` | `soccer_dynamic_v2` |
| Training Script | `train_mappo_curriculum.py` | `train_mappo_dynamic_v2.py` |
| Lambda-Parameter | 4 (recover, pursuit, possession, defense) | 7 (recovery, marking, possession, shooting, blocking, goalkeeping, attack_pos) |
| Datenbank | `optuna.db` | `optuna_dynamic_v2.db` |

## Quick Start

### 1. Master starten (PostgreSQL + MLflow + Dashboard)

```bash
docker-compose -f docker-compose.master.yml up -d
```

Nach dem Start:
- Optuna Dashboard: http://localhost:8080
- MLflow Tracking: http://localhost:5000
- PostgreSQL: localhost:5433

### 2. Worker mit Dynamic Scoring V2

**Option A: Direkt (Python)**
```bash
python worker_entrypoint.py \
  --storage postgresql://optuna:PASSWORD@localhost:5433/optuna_db \
  --study-name soccer_dynamic_v2 \
  --use-dynamic-rewards \
  --infinite \
  --worker-id gpu-01
```

**Option B: Docker Compose**
```bash
# .env im Worker-Verzeichnis anpassen:
OPTUNA_STORAGE=postgresql://optuna:PASSWORD@host:5433/optuna_db
OPTUNA_STUDY_NAME=soccer_dynamic_v2
OPTUNA_USE_DYNAMIC_REWARDS=true
OPTUNA_N_TRIALS=100

docker-compose -f docker-compose.worker.yml up -d
```

### 3. Lokale Optimierung (SQLite)

Für Testing ohne PostgreSQL:
```bash
python optimize_dynamic_v2.py --n-trials 50 --timeout 3600
```

## Hyperparameter Search Space (V2)

### Branch-Gewichte (Lambda)
| Parameter | Range | Default | Beschreibung |
|-----------|-------|---------|--------------|
| `lambda_recovery` | 0.5–2.0 | 1.0 | Ball-Chaser läuft zum Ball |
| `lambda_marking` | 0.5–2.0 | 1.0 | Gegner decken / Angriffsposition |
| `lambda_possession` | 0.5–3.0 | 1.0 | Ball zum Tor bringen |
| `lambda_shooting` | 0.5–2.0 | 1.0 | Schuss aufs (freie) Tor |
| `lambda_blocking` | 0.5–2.0 | 1.0 | Schuss abwehren |
| `lambda_goalkeeping` | 0.1–1.0 | 0.5 | Zwischen Ball und Tor |
| `lambda_attack_pos` | 0.1–1.0 | 0.5 | Freie Torsicht |

### PPO-Hyperparameter
| Parameter | Range | Default |
|-----------|-------|---------|
| `entropy_coef` | 0.01–0.1 (log) | 0.05 |
| `entropy_decay` | 0.9–0.99 | 0.95 |
| `lr` | 1e-5–5e-4 (log) | 3e-4 |
| `episodes_per_batch` | 10, 20, 40 | 20 |
| `ppo_epochs` | 4–10 | 10 |

### Reward-Shaping
| Parameter | Range | Default |
|-----------|-------|---------|
| `reward_scale` | 0.5–2.0 | 1.0 |
| `possession_radius` | 0.4–1.0 | 0.6 |

## Umstellung von V1 auf V2

### Bestehende Worker updaten

1. **Study Name ändern:**
   ```bash
   # Alt (V1)
   --study-name soccer_dynamic_v1
   
   # Neu (V2)
   --study-name soccer_dynamic_v2
   ```

2. **Flag aktivieren:**
   ```bash
   --use-dynamic-rewards
   ```

3. **Worker neustarten:**
   ```bash
   docker-compose -f docker-compose.worker.yml restart
   ```

### Environment Variables

Für Docker-Worker in `.env`:
```bash
# Study Configuration
OPTUNA_STUDY_NAME=soccer_dynamic_v2
OPTUNA_USE_DYNAMIC_REWARDS=true

# Storage
OPTUNA_STORAGE=postgresql://optuna:PASSWORD@host:5433/optuna_db

# Optional: MLflow
MLFLOW_TRACKING_URI=http://host:5000
```

## Monitoring

### TensorBoard
```bash
tensorboard --logdir logs/optuna_dynamic_v2/tensorboard
```

### Optuna Dashboard
```bash
optuna-dashboard sqlite:///optuna_dynamic_v2.db
# Oder für PostgreSQL:
optuna-dashboard postgresql://optuna:PASSWORD@localhost:5433/optuna_db
```

### MLflow
```bash
mlflow ui --backend-store-uri postgresql://optuna:PASSWORD@localhost:5433/optuna_db
```

## Troubleshooting

### Worker verbindet nicht mit Storage
```bash
# PostgreSQL Connectivity testen
psql -h localhost -p 5433 -U optuna -d optuna_db

# Firewall/Network prüfen
nc -zv localhost 5433
```

### "Study not found"
```bash
# Study manuell erstellen
python -c "
import optuna
study = optuna.create_study(
    study_name='soccer_dynamic_v2',
    storage='postgresql://optuna:PASSWORD@localhost:5433/optuna_db',
    direction='maximize',
    load_if_exists=True
)
print('Study created/loaded successfully')
"
```

### Worker stürzt ab
Logs prüfen:
```bash
# Docker Logs
docker-compose -f docker-compose.worker.yml logs -f

# Lokale Logs
tail -f logs/optuna/workers/worker_*.log
```

## Beste Parameter aus vorherigen Trials verwenden

Nach abgeschlossener Optimierung:
```bash
# Beste Parameter anzeigen
cat logs/optuna_dynamic_v2/best_params.json

# Training mit besten Parametern starten
python train_mappo_dynamic_v2.py \
  --lambda-recovery 1.5 \
  --lambda-marking 1.0 \
  --lambda-possession 2.5 \
  --lambda-shooting 1.5 \
  --lambda-blocking 1.5 \
  --lambda-goalkeeping 0.3 \
  --lambda-attack-pos 0.5 \
  --entropy-coef 0.05 \
  --lr 0.0002 \
  --episodes-per-batch 20
```

## Performance-Tipps

1. **GPU-Worker:**
   - Höhere `episodes_per_batch` (40) für bessere GPU-Auslastung
   - Mixed Precision aktivieren (automatisch in `train_mappo_dynamic_v2.py`)

2. **CPU-Worker:**
   - Niedrigere `episodes_per_batch` (10-20)
   - Weniger `ppo_epochs` (4-6)

3. **Pruning:**
   - MedianPruner stoppt schlechte Trials früh
   - Spart ~70% Rechenzeit

4. **Parallelisierung:**
   - Mehrere Worker mit gleichem `--study-name`
   - Jeder Worker bekommt eigene Trials zugewiesen

## Security

**Wichtig:** PostgreSQL nie direkt ins Internet exponieren!

Empfohlene Absicherung:
1. **Cloudflare Zero Trust** (Access Gateway)
2. **SSH Tunnel** für Worker-Verbindung
3. **WireGuard** VPN zwischen Master und Workern

Beispiel SSH Tunnel:
```bash
# Auf Worker-Maschine
ssh -N -L 5433:localhost:5433 user@master-host
```

Dann im Worker:
```bash
--storage postgresql://optuna:PASSWORD@localhost:5433/optuna_db
```
