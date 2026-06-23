# 🏃 Soccer Training - Unified Script

Alle Training-Features sind jetzt in **einer** Datei: `train.py`

## 🚀 Quick Start

```bash
# Standard Training (1000 Episoden, Reward Shaping, kein Viewer)
python train.py

# Training mit Viewer alle 50 Episoden
python train.py --viewer --viewer-interval 50

# Training ohne Reward Shaping (sparse rewards)
python train.py --no-reward-shaping

# Schnelles Test-Training
python train.py --num-episodes 100 --viewer --eval-at-end
```

## 📋 Alle Optionen

### Training Parameter
| Argument | Default | Beschreibung |
|----------|---------|--------------|
| `--num-episodes` | 1000 | Gesamtanzahl Episoden |
| `--episodes-per-batch` | 10 | Episoden pro PPO Update |
| `--ppo-epochs` | 4 | PPO Epochs pro Batch |

### Model Parameter
| Argument | Default | Beschreibung |
|----------|---------|--------------|
| `--hidden-dim` | 256 | Hidden Layer Dimension |
| `--lr` | 3e-4 | Learning Rate |

### PPO Hyperparameter
| Argument | Default | Beschreibung |
|----------|---------|--------------|
| `--gamma` | 0.99 | Discount Factor |
| `--gae-lambda` | 0.95 | GAE Lambda |
| `--clip-epsilon` | 0.2 | PPO Clip Epsilon |
| `--entropy-coef` | 0.01 | Entropy Coefficient |
| `--value-coef` | 0.5 | Value Loss Coefficient |
| `--max-grad-norm` | 0.5 | Max Gradient Norm |

### Reward Shaping
| Argument | Default | Beschreibung |
|----------|---------|--------------|
| `--no-reward-shaping` | False | Disable Reward Shaping |
| `--reward-scale` | 1.0 | Skalierung für Shaped Rewards |

### Visualisierung 🎬
| Argument | Default | Beschreibung |
|----------|---------|--------------|
| `--viewer` | False | Viewer während Training aktivieren |
| `--viewer-interval` | 50 | Viewer alle N Episoden öffnen |
| `--eval-at-end` | False | Viewer am Ende (3 Episoden) |

### Miscellaneous
| Argument | Default | Beschreibung |
|----------|---------|--------------|
| `--seed` | 42 | Random Seed |
| `--log-dir` | logs/soccer_ppo | Tensorboard Log Directory |
| `--save-interval` | 100 | Checkpoint alle N Episoden |
| `--log-interval` | 10 | Log alle N Episoden |

## 💡 Empfohlene Setups

### 1. Schnelles Testen
```bash
python train.py --num-episodes 200 --viewer --eval-at-end
```

### 2. Volles Training mit Visualisierung
```bash
python train.py --num-episodes 1000 --viewer --viewer-interval 100 --eval-at-end
```

### 3. Headless Training (Server/Cluster)
```bash
python train.py --num-episodes 2000 --no-reward-shaping --save-interval 50
```

### 4. Reward Shaping Tuning
```bash
# Stärkeres Reward Shaping
python train.py --reward-scale 2.0 --viewer-interval 50

# Schwächeres Reward Shaping
python train.py --reward-scale 0.5
```

### 5. Hyperparameter Search
```bash
# Größeres Netzwerk
python train.py --hidden-dim 512 --lr 1e-4

# Mehr Exploration
python train.py --entropy-coef 0.05
```

## 📊 Tensorboard Monitoring

```bash
# Tensorboard starten
tensorboard --logdir logs/soccer_ppo

# Im Browser öffnen: http://localhost:6006
```

## 📁 Gespeicherte Dateien

Nach dem Training findest du im `log-dir`:
- `checkpoint_ep{N}.pt` - Checkpoints alle N Episoden
- `final_agent.pt` - Finaler Agent
- `events.out.tfevents.*` - Tensorboard Logs

## 🎮 Checkpoint laden & testen

```bash
python test.py --checkpoint logs/soccer_ppo/final_agent.pt
```

## ⚠️ Wichtige Hinweise

1. **Viewer blockiert Training**: Wenn `--viewer` aktiv ist, pausiert das Training während der Viewer offen ist
2. **Timeout**: Viewer schließt automatisch nach 60 Sekunden (verhindert Hängenbleiben)
3. **Reward Shaping**: Empfohlen für schnelleres Lernen (default: aktiv)
4. **GPU Support**: Automatische Erkennung von CUDA, ROCm (AMD), MPS (Apple)

## 🔄 Migration von alten Scripts

| Altes Script | Neues Command |
|--------------|---------------|
| `train.py` | `python train.py` |
| `train_periodic_viewer.py` | `python train.py --viewer --viewer-interval 50` |
| `train_auto_viewer.py` | `python train.py --viewer --eval-at-end` |
| `train_with_viewer.py` | ❌ (entfernt - funktionierte nicht) |
