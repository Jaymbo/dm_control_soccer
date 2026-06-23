# Archiv-Index

Dieses Archiv enthält alle vorherigen Versionen des Projekts, sortiert nach Iteration.

## Struktur

### `archive/legacy/` — Erste Iterationen
| Datei | Beschreibung |
|-------|--------------|
| `agent.py` | Zentralisierter PPO-Agent |
| `agent_mappo.py` | Erste MAPPO-Version |
| `agent_mappo_optimized.py` | Optimierte MAPPO mit Parameter Sharing |
| `env_wrapper.py` | Basis Environment Wrapper |
| `env_wrapper_optimized.py` | Optimierter Wrapper |
| `env_wrapper_curriculum.py` | Curriculum Learning Wrapper |
| `train.py` | Zentrales PPO-Training |
| `train_mappo.py` | Erstes MAPPO-Training |
| `train_mappo_optimized.py` | Optimiertes MAPPO-Training |
| `train_mappo_curriculum.py` | Curriculum-Training |
| `optimize_curriculum.py` | Lokale Optuna-HPO |
| `test.py`, `test_mappo.py`, `test_mappo_optimized.py`, `test_mappo_curriculum.py` | Tests |

### `archive/dynamic_scoring/` — Dynamic Scoring Iterationen
| Datei | Beschreibung |
|-------|--------------|
| `env_wrapper_dynamic.py` | Dynamic Scoring Wrapper |
| `env_wrapper_dynamic_v2.py` | Dynamic Scoring V2 |
| `train_mappo_dynamic.py` | Dynamic Scoring Training |
| `train_mappo_dynamic_v2.py` | Dynamic Scoring V2 Training |
| `train_mappo_dynamic_v2_online.py` | Online-Update Variante |
| `optimize_dynamic_v2.py` | HPO für Dynamic Scoring V2 |
| `test_mappo_dynamic.py`, `test_mappo_dynamic_v2.py` | Tests |
| `README_DYNAMIC_SCORING.md`, `README_DYNAMIC_SCORING_V2.md` | Dokumentation |
| `CHANGES_DYNAMIC_V2.md`, `WORKER_SETUP_DYNAMIC_V2.md` | Change-Log & Setup |

### `archive/distributed/` — Distributed Training, MLflow, Optuna
| Datei | Beschreibung |
|-------|--------------|
| `worker_entrypoint.py` | Worker-Slave Entry Point |
| `docker-compose.master.yml` | Master Stack |
| `docker-compose.worker.yml` | Worker Container |
| `Dockerfile`, `.dockerignore` | Docker-Image |
| `README_DISTRIBUTED.md`, `README_MLFLOW_MAPO.md`, `MLFLOW_MIGRATION_SUMMARY.md` | Dokumentation |
| `WORKER_OFFLINE_GUIDE.md`, `WORKER_SETUP_DYNAMIC_V2.md`, `DEPLOYMENT.md` | Deployment |
| `.env.example`, `.env.backup.*` | Konfiguration |

### `archive/simple_ball_origin/` — Originale der aktuellsten Version
| Datei | Beschreibung |
|-------|--------------|
| `env_wrapper_simple_ball.py` | Originaler Simple Ball-Chase Wrapper |
| `train_mappo_simple_ball.py` | Originale Simple Ball-Chase Trainingsdatei |

### `archive/experiments/` — Experimente & Debug
| Datei | Beschreibung |
|-------|--------------|
| `agent_mappo_a3c.py` | A3C-Experiment |
| `debug_agent_behavior.py` | Verhaltens-Debugging |
| `live_checkpoint_viewer.py` | Live-Checkpoint-Viewer |
