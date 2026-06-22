# ⚽ Multi-Agent Soccer - Deep Reinforcement Learning

**Production-Ready Distributed Training mit MLflow + Optuna**

[![Status](https://img.shields.io/badge/status-production--ready-green)](https://github.com/Jaymbo/dm_control_soccer)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

---

## 🚀 Quickstart

### Master Server einrichten (5 Min)

```bash
ssh dein-server.com
git clone git@github.com:Jaymbo/dm_control_soccer.git
cd dm_control_soccer
./scripts/setup_master.sh
```

### Worker einrichten (3 Min pro Gerät)

```bash
git clone git@github.com:Jaymbo/dm_control_soccer.git
cd dm_control_soccer
./scripts/setup_worker.sh
```

**→ Das ist alles!** Worker verbinden sich automatisch und starten Training.

---

## 📖 Dokumentation

| Dokument | Zweck |
|----------|-------|
| **[SETUP_ANLEITUNG.md](SETUP_ANLEITUNG.md)** | ⭐ **Start here!** Schritt-für-Schritt Guide |
| **[DEPLOYMENT.md](DEPLOYMENT.md)** | Komplette Deploy-Anleitung mit Cloudflare/Tailscale |
| **[QUICKSTART.md](QUICKSTART.md)** | Schnelleinstieg für Training & HPO |
| **[PROJECT_STATUS.md](PROJECT_STATUS.md)** | Aktueller Projekt-Status & Architektur |
| **[README_MLFLOW_MAPO.md](README_MLFLOW_MAPO.md)** | MLflow + Optuna Details |

---

## 🏗️ Architektur

```
┌─────────────────────────────────────────────────────────────┐
│                  MASTER SERVER (24/7)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  MLflow      │  │  PostgreSQL  │  │  Cloudflare      │   │
│  │  :5000       │  │  :5432       │  │  Tunnel          │   │
│  └──────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ Secure Tunnel
                          │
    ┌─────────────────────┼─────────────────────┐
    │                     │                     │
    ▼                     ▼                     ▼
┌─────────┐         ┌─────────┐           ┌─────────┐
│ Worker 1│         │ Worker 2│           │ Worker N│
│ Laptop  │         │ Desktop │           │ Server  │
│ (CPU)   │         │ (GPU)   │           │ Mixed   │
└─────────┘         └─────────┘           └─────────┘
```

---

## ✨ Features

### Training
- ✅ **Dynamic Scoring** - Zustandsbasierte Reward-Branches
- ✅ **Curriculum Learning** - Phasenweises Training
- ✅ **MAPPO** - Multi-Agent PPO mit zentralisiertem Critic
- ✅ **Mixed Precision** - AMP für GPU-Training

### Distributed
- ✅ **MLflow** - Experiment Tracking & Model Registry
- ✅ **Optuna** - Hyperparameter-Optimierung mit Pruning
- ✅ **Cloudflare Zero Trust** - Secure Access ohne VPN
- ✅ **Tailscale** - Alternative Mesh-VPN
- ✅ **Auto-Scaling** - Worker kommen/gehen beliebig

### Monitoring
- ✅ **TensorBoard** - Lokale Metrics
- ✅ **MLflow UI** - Experiment-Vergleich
- ✅ **Optuna Dashboard** - HPO-Status

---

## 🎯 Use Cases

### 1. Lokales Testing
```bash
python train_mappo_dynamic.py --num-episodes 100 --viewer
```

### 2. Hyperparameter-Optimierung (Lokal)
```bash
python optimize_curriculum.py --n-trials 50
```

### 3. Distributed Production
```bash
# Master
./scripts/setup_master.sh

# Worker (beliebig viele)
./scripts/setup_worker.sh
```

---

## 📊 Performance

| Hardware | Steps/sec | Episoden/Min |
|----------|-----------|--------------|
| CPU (8 Core) | ~1500-2000 | ~30-40 |
| GPU (NVIDIA) | ~3000-5000 | ~60-80 |
| Distributed (4 Worker) | ~8000+ | ~150+ |

---

## 🔒 Security

- ✅ **Cloudflare Zero Trust** - PostgreSQL nie öffentlich exponieren
- ✅ **Automatische HTTPS** - Verschlüsselte Kommunikation
- ✅ **Starke Passwörter** - Auto-generiert beim Setup
- ✅ **Access Control** - Optional mit Cloudflare Policies

---

## 🛠️ Tech Stack

| Komponente | Technologie |
|------------|-------------|
| Environment | DM Control Suite (MuJoCo) |
| RL Algorithm | MAPPO (Multi-Agent PPO) |
| Deep Learning | PyTorch |
| Experiment Tracking | MLflow |
| Hyperparameter Opt. | Optuna |
| Database | PostgreSQL |
| Security | Cloudflare Zero Trust / Tailscale |
| Deployment | Docker Compose |

---

## 📁 Repository Struktur

```
dm_control_soccer/
├── scripts/
│   ├── setup_master.sh        # ⭐ Master Setup
│   ├── setup_worker.sh        # ⭐ Worker Setup
│   └── README.md              # Script-Doku
├── train_mappo_dynamic.py     # Dynamic Scoring Training
├── worker_entrypoint.py       # Distributed Worker
├── docker-compose.master.yml  # Master Stack
├── docker-compose.worker.yml  # Worker Container
├── SETUP_ANLEITUNG.md         # ⭐ Schritt-für-Schritt
├── DEPLOYMENT.md              # Complete Deploy Guide
├── QUICKSTART.md              # Quick Reference
└── README.md                  # This file
```

---

## 🎓 Nächste Schritte

1. **[SETUP_ANLEITUNG.md](SETUP_ANLEITUNG.md)** lesen
2. Master auf Server einrichten
3. Ersten Worker starten
4. Training in MLflow/Optuna beobachten
5. Mehr Worker hinzufügen für Speed

---

## 📚 Weitere Ressourcen

- [Dynamic Scoring Guide](README_DYNAMIC_SCORING.md)
- [Curriculum Learning](CURRICULUM_LEARNING.md)
- [Architecture Comparison](MAPPO_VS_PPO.md)
- [Optimizations](OPTIMIZATIONS_SUMMARY.md)

---

## 🤝 Contributing

Issues und PRs sind willkommen!

---

## 📄 License

MIT License - siehe [LICENSE](LICENSE)

---

## 🙏 Acknowledgments

- DM Control Suite (DeepMind)
- MuJoCo (Tencent)
- PyTorch
- MLflow
- Optuna

---

**Viel Erfolg beim Training! ⚽🤖**

[![Deployment](https://img.shields.io/badge/deployment-one--click-green)](SETUP_ANLEITUNG.md)
[![Documentation](https://img.shields.io/badge/docs-complete-blue)](SETUP_ANLEITUNG.md)
