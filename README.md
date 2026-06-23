# ⚽ Multi-Agent Soccer RL - Sauberer Start

Dies ist die aufgeräumte Version des Projekts. Alle älteren Iterationen sind sicher im Ordner [`archive/`](archive/) archiviert.

## 🚀 Schnellstart

```bash
# Environment-Check
python -m src.tests.test_env --help

# Training starten (CPU-freundlich, kleiner Testlauf)
python -m src.training.train --num-updates 50 --num-envs 4 --log-dir logs/test_run

# TensorBoard starten
tensorboard --logdir logs/test_run
```

## 📁 Struktur

```
.
├── archive/              # ALTE VERSIONEN - nichts wurde gelöscht
│   ├── legacy/           # Erste Iterationen (PPO, MAPPO v1, Optimized, Curriculum)
│   ├── dynamic_scoring/  # Dynamic Scoring + Dynamic Scoring V2
│   ├── distributed/      # MLflow, Optuna, Docker, Worker
│   └── experiments/      # A3C, Debug-Tools, Viewer
├── src/                  # NEUER SAUBERER START
│   ├── env/              # Environment Wrapper
│   ├── agents/           # MAPPO Agent
│   ├── training/         # Training Scripts
│   └── tests/            # Tests & Visualisierung
├── configs/              # Zukünftige YAML/JSON-Configs
├── docs/                 # Gesammelte Dokumentation
├── logs/                 # Trainings-Logs
├── mlruns/               # MLflow-Tracking
└── requirements.txt      # Abhängigkeiten
```

## 📦 Module

| Datei | Zweck |
|-------|-------|
| `src/env/soccer_env.py` | `SimpleBallChaseWrapper` für 2v2 Soccer |
| `src/agents/mappo_agent.py` | MAPPO-Agent mit Parameter Sharing |
| `src/training/train.py` | Haupt-Trainingsskript |
| `src/tests/test_env.py` | Agent laden & visualisieren |

## 📚 Wichtige alte Dokumente

- `docs/PROJECT_HISTORY_README.md` — Ursprüngliches Haupt-README
- `docs/PROJECT_STATUS.md` — Letzter Projekt-Status
- `docs/SETUP_ANLEITUNG.md` — Distributed Setup
- `archive/ARCHIVE_INDEX.md` — Übersicht aller archivierten Dateien

## ⚠️ Hinweis

Die alten Skripte im `archive/`-Ordner funktionieren weiterhin, verweisen aber eventuell auf Dateien, die nun verschoben sind. Wenn Sie eine alte Version neu starten wollen, kopieren Sie den entsprechenden Ordner am besten zurück ins Hauptverzeichnis oder passen die Imports an.

## 🛠️ Nächste Schritte

1. `python -m src.tests.test_env` → prüfen, ob alles lädt
2. Kleinen Testlauf starten
3. Ergebnisse in TensorBoard prüfen
4. Dann schrittweise erweitern
