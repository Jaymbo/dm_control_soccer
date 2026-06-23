# 🚀 Dezentrale Hyperparameter-Optimierung mit MLflow + Optuna

Diese Anleitung beschreibt das Setup für **dezentrales Training** mit:
- **MLflow**: Experiment Tracking (zentraler Server)
- **Optuna**: Hyperparameter-Optimierung mit Pruning (zentrale PostgreSQL-DB)
- **Worker**: Autonome Slaves, die kommen/gehen können (CPU + GPU parallel)

---

## 🏗️ Architektur

```
┌─────────────────────────────────────────────────────────────┐
│                  MASTER (Server 24/7)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  MLflow      │  │  PostgreSQL  │  │  Dashboards      │   │
│  │  Server      │  │  (Optuna)    │  │  :5000 / :8080   │   │
│  │  :5000       │  │  :5433       │  │                  │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                          ▲
                          │ (nur bei Trial-Start/Ende)
                          │
    ┌─────────────────────┼─────────────────────┐
    │                     │                     │
    ▼                     ▼                     ▼
┌─────────┐         ┌─────────┐           ┌─────────┐
│ Slave 1 │         │ Slave 2 │           │ Slave 3 │
│ Laptop  │         │ Desktop │           │ Server  │
│ (CPU)   │         │ (GPU)   │           │ (CPU)   │
│ kommt   │         │ immer   │           │ geht    │
│ & geht  │         │ an      │           │ offline │
└─────────┘         └─────────┘           └─────────┘
```

**Vorteile:**
- ✅ **Kein Sync-Zwang** während Training (nur Start/Ende)
- ✅ **CPU + GPU parallel** ohne Wartezeit
- ✅ **Worker können offline gehen** ohne System zu stören
- ✅ **Optuna-Pruning** funktioniert zentral (bessere Entscheidungen)
- ✅ **MLflow-Tracking** zentral (einheitliche Historie)

---

## 📦 Installation

### Master-Server

```bash
# Dependencies installieren
pip install -r requirements.txt

# Docker-Stack starten
docker-compose -f docker-compose.master.yml up -d
```

**Zugänge:**
- MLflow UI: http://localhost:5000
- Optuna Dashboard: http://localhost:8080
- PostgreSQL: localhost:5433 (⚠️ nur via VPN/Cloudflare!)

### Worker (Slaves)

```bash
# Dependencies installieren
pip install -r requirements.txt

# PyTorch je nach Hardware:
# AMD GPU:
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
# NVIDIA GPU:
pip install torch --index-url https://download.pytorch.org/whl/cu118
# CPU:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# ⚠️ WICHTIG: MuJoCo Renderer setzen (vermeidet Speicherfehler)
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

---

## 🔧 Konfiguration

### Master: `.env` Datei

```bash
# .env (auf Master-Server)

# PostgreSQL
POSTGRES_USER=optuna
POSTGRES_PASSWORD=CHANGE_ME_TO_SECURE_PASSWORD
POSTGRES_DB=optuna_db
POSTGRES_PORT=5433

# MLflow
MLFLOW_PORT=5000

# Optuna Dashboard
DASHBOARD_PORT=8080
```

### Worker: `.env` Datei

```bash
# .env (auf Worker-Geräten)

# Verbindung zu Master-Server
OPTUNA_STORAGE=postgresql://optuna:PASSWORD@MASTER_IP:5433/optuna_db
MLFLOW_TRACKING_URI=http://MASTER_IP:5000

# Studie
OPTUNA_STUDY_NAME=soccer_dynamic_v1

# Trials pro Worker-Run
OPTUNA_N_TRIALS=10

# Dynamic Scoring aktivieren
OPTUNA_USE_DYNAMIC_REWARDS=true

# Optional: Worker-ID für Logging
OPTUNA_WORKER_ID=laptop-01

# GPU/CPU Auto-Detection (optional)
# PYTORCH_DEVICE=cuda  # oder 'cpu' oder 'mps'
```

---

## 🚀 Quickstart

### 1. Master starten

```bash
docker-compose -f docker-compose.master.yml up -d

# Status prüfen
docker-compose -f docker-compose.master.yml ps

# Logs ansehen
docker-compose -f docker-compose.master.yml logs -f
```

### 2. Worker starten (lokal testen)

```bash
# ⚠️ WICHTIG: Renderer zuerst setzen!
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Ohne Docker (direkt)
python worker_entrypoint.py \
  --storage sqlite:///optuna_local.db \
  --n-trials 2 \
  --use-dynamic-rewards

# Mit Master-Verbindung (Produktion)
python worker_entrypoint.py \
  --storage postgresql://optuna:PASSWORD@MASTER_IP:5433/optuna_db \
  --mlflow-tracking-uri http://MASTER_IP:5000 \
  --n-trials 10 \
  --use-dynamic-rewards
```

### 3. Worker mit Docker starten

```bash
docker-compose -f docker-compose.worker.yml up -d

# Logs
docker-compose -f docker-compose.worker.yml logs -f
```

---

## 📊 Monitoring

### MLflow UI

Öffne http://MASTER_IP:5000 für:
- Experiment-Vergleich
- Metrik-Historie (Rewards, Losses, Branch-Stats)
- Modell-Artifacts (Checkpoints)
- Hyperparameter-Importance

### Optuna Dashboard

Öffne http://MASTER_IP:8080 für:
- Trial-Historie
- Hyperparameter-Optimierung
- Pruning-Statistiken
- Best Trials

### TensorBoard (lokal auf Worker)

```bash
tensorboard --logdir logs/optuna/tensorboard
```

---

## 🎯 Hyperparameter-Suche

### Dynamic Scoring Parameter

Optuna optimiert automatisch:
- `episodes_per_batch`: [10, 20, 40]
- `ppo_epochs`: [4-10]
- `lr`: [1e-5 bis 5e-4]
- `entropy_coef`: [0.01 bis 0.1]
- `entropy_decay`: [0.9 bis 0.99]
- `reward_scale`: [0.5 bis 2.0]
- `possession_radius`: [0.4 bis 1.0]
- `goal_threshold`: [4.0 bis 8.0]
- `lambda_*`: [0.5 bis 2.0] für jeden Branch

### Manuelles Training mit besten Parametern

```bash
python train_mappo_dynamic.py \
  --num-episodes 1000 \
  --lr 3e-4 \
  --entropy-coef 0.05 \
  --possession-radius 0.6 \
  --lambda-pursuit 1.5 \
  --viewer
```

---

## 🔒 Sicherheit

### PostgreSQL schützen

**⚠️ NIEMALS PostgreSQL direkt ins Internet exponieren!**

**Option A: Cloudflare Zero Trust (empfohlen)**
```bash
# Auf Master-Server
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
sudo dpkg -i cloudflared-linux-amd64.deb

# Tunnel erstellen (Cloudflare Dashboard)
cloudflared tunnel create optuna-tunnel

# Route konfigurieren
cloudflared tunnel route dns optuna-tunnel optuna.your-domain.com

# Config erstellen
cat > /etc/cloudflared/config.yml << EOF
tunnel: optuna-tunnel
ingress:
  - hostname: optuna.your-domain.com
    service: tcp://localhost:5433
  - service: http_status:404
EOF

# Starten
sudo systemctl enable cloudflared
sudo systemctl start cloudflared
```

Worker-Connection-String:
```
postgresql://optuna:password@optuna.your-domain.com:443/optuna_db
```

**Option B: SSH-Tunnel (einfach)**
```bash
# Auf Worker
ssh -L 5433:localhost:5433 user@master-server
```

Connection-String:
```
postgresql://optuna:password@localhost:5433/optuna_db
```

**Option C: WireGuard VPN**
- WireGuard auf Master und Workern einrichten
- Private IPs für Kommunikation nutzen

---

## 🛠️ Troubleshooting

### Worker kann sich nicht verbinden

```bash
# Netzwerk prüfen
telnet MASTER_IP 5433
telnet MASTER_IP 5000

# Firewall prüfen (auf Master)
sudo ufw status
sudo iptables -L -n
```

### MLflow loggt nicht

```bash
# MLflow-Verfügbarkeit prüfen
python -c "import mlflow; print(mlflow.__version__)"

# Connection testen
python -c "import mlflow; mlflow.set_tracking_uri('http://MASTER_IP:5000'); mlflow.get_experiment_by_name('test')"
```

### Trials werden nicht verteilt

1. Gleiche `--study-name` auf allen Workern?
2. Gleiche Storage-URL?
3. PostgreSQL erreichbar?

```bash
# Studie prüfen
optuna-dashboard postgresql://optuna:password@MASTER_IP:5433/optuna_db
```

### Out of Memory

```bash
# Batch-Größe reduzieren
OPTUNA_N_TRIALS=5
# Oder in worker_entrypoint.py: num_episodes=200 statt 400
```

### "free(): invalid pointer" oder Speicherfehler

⚠️ **Häufiges Problem mit MuJoCo 3.1.6 + GLIBC 2.43+**

Lösung: Renderer korrekt setzen VOR dem Training:

```bash
# EGL Rendering (empfohlen für headless Server)
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# Alternative: OSMesa (langsamer, aber kompatibler)
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa

# Dann Training starten
python train_mappo_dynamic.py ...
```

In `.env` Datei (permanent):
```bash
MUJOCO_GL=egl
PYOPENGL_PLATFORM=egl
```

---

## 📈 Skalierung

### Mehr Worker hinzufügen

Einfach weitere Worker starten (auf beliebigen Geräten):

```bash
# Worker 1
python worker_entrypoint.py --storage postgresql://... --infinite

# Worker 2 (anderes Gerät)
python worker_entrypoint.py --storage postgresql://... --infinite

# Worker 3 (Docker)
docker-compose -f docker-compose.worker.yml up -d
```

### Ressourcen-Limits (Docker)

```yaml
# docker-compose.worker.yml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 8G
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

---

## 💾 Backup

### PostgreSQL Backup

```bash
docker-compose -f docker-compose.master.yml exec postgres \
  pg_dump -U optuna optuna_db > backup_$(date +%Y%m%d).sql
```

### MLflow Artifacts Backup

```bash
docker cp optuna-postgres:/var/lib/postgresql/data backup_pg_data
docker cp mlflow-server:/mlflow-artifacts backup_mlflow_artifacts
```

---

## 📊 Best Practices

1. **Startup-Trials**: Erste 10-20 Trials mit Random Search (`n_startup_trials=10`)
2. **Pruning**: Median-Pruner nach 3 Batches aktivieren
3. **Dynamic Scoring**: Immer aktivieren für schnelleres Learning
4. **GPU-Nutzung**: Mixed Precision (AMP) automatisch aktiv
5. **Checkpointing**: Bestes Modell automatisch speichern

---

## 🎓 Nächste Schritte

1. Master-Server aufsetzen (PostgreSQL + MLflow)
2. Cloudflare Zero Trust für sicheren Zugang
3. Ersten Worker lokal testen
4. Weitere Worker auf Cluster/Remote-Geräten
5. Fortschritt via Dashboards monitorieren
6. Beste Hyperparameter analysieren
7. Finales Modell mit besten Parametern trainieren

---

## 📚 Referenzen

- [MLflow Documentation](https://mlflow.org/docs/)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Dynamic Scoring Guide](README_DYNAMIC_SCORING.md)
- [Distributed Setup (alt)](README_DISTRIBUTED.md)
