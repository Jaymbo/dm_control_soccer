# 🚀 QUICKSTART - Multi-Agent Soccer DRL

Schnellstart-Anleitung für Training und Hyperparameter-Optimierung.

---

## ⚡ 1-Minuten Setup

### Dependencies installieren

```bash
# Core Dependencies
pip install -r requirements.txt

# PyTorch (wähle deine Hardware):
# AMD GPU:
pip install torch --index-url https://download.pytorch.org/whl/rocm6.0
# NVIDIA GPU:
pip install torch --index-url https://download.pytorch.org/whl/cu118
# CPU:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# ⚠️ WICHTIG: Renderer setzen (vermeidet Speicherfehler!)
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

### Test-Run

```bash
# 50 Episoden Training (ca. 2-3 Minuten)
python train_mappo_dynamic.py --num-episodes 50 --log-dir logs/quickstart
```

---

## 🎯 Training-Optionen

### A. Manuelles Training (Empfohlen für Start)

```bash
# Dynamic Scoring (beste Performance)
python train_mappo_dynamic.py \
  --num-episodes 1000 \
  --reward-scale 1.0 \
  --entropy-coef 0.05 \
  --viewer

# Curriculum Learning (einfachster Einstieg)
python train_mappo_curriculum.py \
  --num-episodes 1000 \
  --start-phase 0 \
  --auto-advance
```

### B. Hyperparameter-Optimierung (Automatisch)

```bash
# Lokale Optimierung (SQLite, 50 Trials)
python optimize_curriculum.py --n-trials 50
```

### C. Distributed Setup (Production) ⭐

**Master Server:**
```bash
git clone git@github.com:Jaymbo/dm_control_soccer.git
cd dm_control_soccer
./scripts/setup_master.sh
```

**Worker (beliebig viele):**
```bash
git clone git@github.com:Jaymbo/dm_control_soccer.git
cd dm_control_soccer
./scripts/setup_worker.sh
```

→ **Das ist alles!** Worker verbinden sich automatisch und starten Training.

**Ausführliche Anleitung:** Siehe [DEPLOYMENT.md](DEPLOYMENT.md)

---

## 📊 Wichtige Parameter

### Training

| Parameter | Empfehlung | Beschreibung |
|-----------|------------|--------------|
| `--num-episodes` | 500-2000 | Gesamte Trainingsepisoden |
| `--episodes-per-batch` | 10-20 | Episoden pro PPO-Update |
| `--ppo-epochs` | 4-10 | PPO-Epochen pro Update |
| `--hidden-dim` | 256-512 | Netzwerk-Größe |
| `--lr` | 1e-4 - 3e-4 | Lernrate |
| `--entropy-coef` | 0.01-0.1 | Exploration-Bonus |
| `--reward-scale` | 0.5-2.0 | Reward-Skalierung |

### Dynamic Scoring Branches

| Branch | Parameter | Empfehlung |
|--------|-----------|------------|
| Recovery | `--lambda-recover` | 0.5-2.0 |
| Pursuit | `--lambda-pursuit` | 1.0-2.0 |
| Possession | `--lambda-possession` | 1.0-2.0 |
| Defense | `--lambda-defense` | 0.5-1.5 |

---

## 🖥️ Monitoring

### TensorBoard

```bash
tensorboard --logdir logs/
# Öffne http://localhost:6006
```

### MLflow UI (Distributed Setup)

```bash
# Nach docker-compose -f docker-compose.master.yml up -d
# Öffne http://localhost:5000
```

### Optuna Dashboard (Distributed Setup)

```bash
# Nach docker-compose -f docker-compose.master.yml up -d
# Öffne http://localhost:8080
```

---

## 🧪 Testing

### Agent testen

```bash
# Checkpoint laden und visualisieren
python test_mappo_dynamic.py \
  --checkpoint logs/soccer_mappo_dynamic/final_agent.pt \
  --viewer
```

### Environment testen

```bash
python -c "
from env_wrapper_dynamic import make_env_with_dynamic_rewards
env = make_env_with_dynamic_rewards(seed=42)
ts = env.reset()
print('Environment OK!')
"
```

---

## 🐛 Troubleshooting

### "free(): invalid pointer"

```bash
# Renderer korrekt setzen!
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

### Training divergiert

```bash
# Lernrate reduzieren, Exploration erhöhen
python train_mappo_dynamic.py \
  --lr 1e-4 \
  --entropy-coef 0.1 \
  --entropy-decay 0.9
```

### Agenten bewegen sich nicht

```bash
# Curriculum Learning verwenden (einfacher zu lernen)
python train_mappo_curriculum.py --auto-advance
```

### Langsames Training

```bash
# GPU nutzen (falls verfügbar)
# PyTorch mit CUDA/ROCm installieren
# Batch-Größe erhöhen
python train_mappo_dynamic.py --episodes-per-batch 40
```

---

## 📁 Wichtige Dateien

| Datei | Zweck |
|-------|-------|
| `train_mappo_dynamic.py` | Dynamic Scoring Training |
| `train_mappo_curriculum.py` | Curriculum Learning Training |
| `worker_entrypoint.py` | Distributed Worker |
| `optimize_curriculum.py` | Lokale HPO |
| `env_wrapper_dynamic.py` | Dynamic Scoring Rewards |
| `README_MLFLOW_MAPO.md` | Distributed Setup Guide |
| `README_DYNAMIC_SCORING.md` | Reward System Details |

---

## 🎓 Nächste Schritte

1. **Erstes Training**: `train_mappo_dynamic.py --num-episodes 100 --viewer`
2. **Hyperparameter-Tuning**: `optimize_curriculum.py --n-trials 20`
3. **Distributed Setup**: `docker-compose -f docker-compose.master.yml up -d`
4. **Production Training**: Worker mit `--infinite` starten
5. **Monitoring**: MLflow + Optuna Dashboards nutzen

---

## 📚 Vollständige Dokumentation

- [MLflow + Optuna Setup](README_MLFLOW_MAPO.md)
- [Dynamic Scoring Guide](README_DYNAMIC_SCORING.md)
- [Curriculum Learning](CURRICULUM_LEARNING.md)
- [Architecture Comparison](MAPPO_VS_PPO.md)
- [Optimizations](OPTIMIZATIONS_SUMMARY.md)

---

**Viel Erfolg beim Training! ⚽🤖**
