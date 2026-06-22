# Hyperparameter-Optimierung mit Optuna

Dieses Dokument beschreibt die effiziente Hyperparameter-Optimierung für Curriculum MAPPO Soccer.

## 🚀 Quick Start

```bash
# 1. Optuna installieren
pip install optuna optuna-integration

# 2. Optimierung starten (50 Trials, max. 2 Stunden)
python optimize_curriculum.py --n-trials 50 --timeout 7200

# 3. Ergebnisse in TensorBoard ansehen
tensorboard --logdir logs/optuna/tensorboard

# 4. (Optional) Interaktives Dashboard
pip install optuna-dashboard
optuna-dashboard sqlite:///optuna.db
```

## 🎯 Warum Optuna?

| Methode | Effizienz | Beschreibung |
|---------|-----------|--------------|
| **Manual Search** | ❌ Schlecht | Manuelles Ausprobieren, keine Systematik |
| **Grid Search** | ❌ Schlecht | Testet alle Kombinationen, sehr teuer |
| **Random Search** | ⚠️ Mittel | Besser als Grid, aber kein Lernen |
| **Optuna (TPE)** | ✅ **State-of-the-Art** | Lernt aus erfolgreichen Trials |

### Vorteile von Optuna

1. **TPE Sampler**: Tree-structured Parzen Estimator lernt welche Hyperparameter-Kombinationen gut funktionieren
2. **Median Pruning**: Stoppt schlechte Trials früh (spart ~70% Rechenzeit)
3. **TensorBoard-Integration**: Direkte Visualisierung der Ergebnisse
4. **Persistente Storage**: Trials werden in SQLite gespeichert, können später fortgesetzt werden

## 📊 Hyperparameter Search Space

| Hyperparameter | Range | Typ | Beschreibung |
|----------------|-------|-----|--------------|
| `phase_episodes` | 20-80 | int | Episoden pro Curriculum-Phase |
| `phase_success_rate` | 0.3-0.8 | float | Erfolgsrate für Phasenwechsel |
| `entropy_coef` | 0.01-0.1 | log | Exploration-Bonus |
| `entropy_decay` | 0.9-0.99 | float | Decay der Exploration |
| `lr` | 1e-5 bis 5e-4 | log | Learning Rate |
| `episodes_per_batch` | 10, 20, 40 | categorical | Batch-Größe |
| `ppo_epochs` | 4-10 | int | PPO Update-Epochen |
| `reward_scale` | 0.5-2.0 | float | Reward-Skalierung |

## 🔧 Konfiguration

### Basis-Optimierung (empfohlen für Start)

```bash
python optimize_curriculum.py \
  --n-trials 30 \
  --timeout 3600 \
  --n-startup-trials 10
```

- **Dauer**: ~1 Stunde
- **Trials**: 30 (davon 10 random, 20 TPE-gelernt)
- **Erwartete Verbesserung**: 20-40% bessere Performance

### Intensive Optimierung (für Production)

```bash
python optimize_curriculum.py \
  --n-trials 100 \
  --timeout 14400 \
  --n-startup-trials 20
```

- **Dauer**: ~4 Stunden
- **Trials**: 100 (davon 20 random, 80 TPE-gelernt)
- **Erwartete Verbesserung**: 40-60% bessere Performance

### Parallele Optimierung (für mehrere GPUs/CPUs)

```bash
# Terminal 1
python optimize_curriculum.py --n-trials 100 --study-name soccer_v1

# Terminal 2 (gleiche Study!)
python optimize_curriculum.py --n-trials 100 --study-name soccer_v1
```

Beide Prozesse teilen sich die SQLite-Datenbank und arbeiten an derselben Study.

## 📈 Ergebnisse interpretieren

### TensorBoard

```bash
tensorboard --logdir logs/optuna/tensorboard
```

Zeigt:
- **avg_reward pro Trial**: Welche Trials funktionieren am besten?
- **Hyperparameter-Importance**: Welche Hyperparameter sind am wichtigsten?
- **Pruning-Statistiken**: Wie viele Trials wurden früh gestoppt?

### Optuna Dashboard (empfohlen!)

```bash
optuna-dashboard sqlite:///optuna.db
# Öffne http://localhost:8080
```

Zeigt interaktiv:
- **Trial History**: Performance über alle Trials
- **Hyperparameter Relationships**: Zusammenhänge zwischen Parametern
- **Parallel Coordinate Plot**: Beste Trials im Vergleich
- **Parameter Importances**: Feature Importance der Hyperparameter

### Beste Parameter finden

Nach der Optimierung:

```bash
cat logs/optuna/best_params.json
```

Training mit besten Parametern:

```bash
python train_mappo_curriculum.py \
  --phase-episodes <wert> \
  --phase-success-rate <wert> \
  --entropy-coef <wert> \
  --lr <wert> \
  ...
```

## 🎓 Wie Pruning funktioniert

**Median Pruning** vergleicht den aktuellen Trial mit vorherigen Trials:

```
Trial 1: Ep 10 → 100, Ep 20 → 150, Ep 30 → 200 ✅ (komplett)
Trial 2: Ep 10 → 50, Ep 20 → 60 ❌ (gepruned bei Ep 20)
Trial 3: Ep 10 → 120, Ep 20 → 180, Ep 30 → 250 ✅ (komplett, besser!)
```

**Ersparte Rechenzeit**: ~70% (schlechte Trials werden nach 20-30 Episoden gestoppt)

## 📝 Beispiel-Workflow

### 1. Erste Exploration (30 Min)

```bash
python optimize_curriculum.py --n-trials 15 --timeout 1800
```

### 2. Ergebnisse ansehen

```bash
tensorboard --logdir logs/optuna/tensorboard
# Oder
optuna-dashboard sqlite:///optuna.db
```

### 3. Vielversprechende Region identifizieren

Beispiel: `entropy_coef` zwischen 0.03-0.05 funktioniert gut

### 4. Vertiefte Suche (1-2 Stunden)

```bash
python optimize_curriculum.py \
  --n-trials 50 \
  --timeout 7200 \
  --study-name soccer_v2  # Neue Study für fokussierte Suche
```

### 5. Finales Training

Beste Parameter aus `logs/optuna/best_params.json` verwenden:

```bash
python train_mappo_curriculum.py \
  --num-episodes 1000 \
  --phase-episodes 40 \
  --phase-success-rate 0.5 \
  --entropy-coef 0.04 \
  --lr 0.0002 \
  ...
```

## 🐛 Troubleshooting

| Problem | Lösung |
|---------|--------|
| `ModuleNotFoundError: No module named 'optuna'` | `pip install optuna optuna-integration` |
| Trials werden nicht gespeichert | Prüfe `sqlite:///optuna.db` existiert |
| TensorBoard zeigt keine HParams | `logs/optuna/tensorboard` verwenden |
| Training divergiert (NaN) | `entropy_coef` erhöhen, `lr` senken |
| Pruning zu aggressiv | `--n-warmup-steps` erhöhen |

## 📚 Weiterführende Links

- [Optuna Dokumentation](https://optuna.readthedocs.io/)
- [TPE Algorithmus Paper](https://papers.nips.cc/paper/2011/hash/86e8f7ab32cfd12577bc2619bc63ae57-Abstract.html)
- [Optuna Dashboard](https://github.com/optuna/optuna-dashboard)

## 🎯 Definition of Done

- [ ] Optuna installiert (`pip install optuna optuna-integration`)
- [ ] Erste Optimierung mit 30 Trials durchgeführt
- [ ] TensorBoard/Optuna Dashboard ausgewertet
- [ ] Beste Parameter identifiziert
- [ ] Finales Training mit besten Parametern durchgeführt
- [ ] Performance dokumentiert (Vergleich vorher/nachher)
