# MPO — Maximum a Posteriori Policy Optimization

Paper-getreue Implementierung von **MPO** (Abdolmaleki et al., ICLR 2018) für
[dm_control](https://github.com/deepmind/dm_control) / MuJoCo-Umgebungen.

## Architektur

| Komponente | Implementierung |
|---|---|
| **Policy** | Tanh-squashed Gaussian (diagonal covariance), 256-256 MLP |
| **Critic** | Twin Q-Networks (256-256), Polyak-averaged targets |
| **E-Step** | Sample K=20 actions per state, target weights `q(a|s) ∝ π(a|s)·exp(Q/η)` |
| **M-Step** | Weighted NLL + decoupled KL constraints (λ_μ für Mean, λ_Σ für Covariance) |
| **Dual Variables** | η (entropy), λ_μ, λ_Σ — multiplicative updates in log-space |

### Schlüssel-Paper-Details
- **Unregularisierte Q-Funktion** (kein SAC-Entropy-Bonus, Eq. 5)
- **Entropie-Schranke E-Step**: `H(q) ≥ log(K) - ε` mit ε=0.1
- **Entkoppelte KL-Constraints M-Step**: ε_μ=0.1, ε_Σ=0.0001 (Appendix D.3, Eq. 27)
- **Mehrere Gradient-Schritte** pro Collection-Runde (Default: 10 Critic + 10 Actor)

## Quick Start

```bash
pip install -r requirements.txt

# Training (z.B. cartpole/balance)
python train.py --domain cartpole --task balance --steps 100000

# Endless Training bis Ctrl+C
python train.py --steps 0

# Visualisiere trainierten Agent
python test.py --domain cartpole --task balance

# Hyperparameter-Optimierung (Optuna)
python hpo.py --trials 20 --steps 20000

# Endless HPO bis Ctrl+C
python hpo.py --trials 0 --steps 50000
```

## Dashboards

```bash
# MLflow: Metrics pro Trial/Run
mlflow ui --backend-store-uri sqlite:///mlflow.db          # Port 5000

# Optuna: HPO-Such-Analyse
optuna-dashboard sqlite:///optuna.db                       # Port 8080
```

## Parallele HPO-Worker

Mehrere `hpo.py`-Prozesse können dieselbe `optuna.db` + `mlflow.db` nutzen.
Optuna verteilt Trials per SQLite-Locking automatisch.

```bash
# Terminal 1
python hpo.py --trials 0 --steps 20000

# Terminal 2
python hpo.py --trials 0 --steps 20000
```

Jeder Trial erhält einen eindeutigen Checkpoint-Pfad (`trial{N}` Tag).

## Resume / Checkpoints

Checkpoints werden automatisch gespeichert und beim nächsten Start geladen
(außer mit `--no-resume`):

```
checkpoints/mpo_{domain}_{task}.pt           # Standard
checkpoints/mpo_{domain}_{task}_{tag}.pt     # Mit --checkpoint_tag
```

Gespeichert werden: Policy, Q-Networks, Target-Networks, Dual-Variablen,
Replay-Buffer, `total_steps`, `best_eval`.

## Datei-Übersicht

| Datei | Beschreibung |
|---|---|
| `agent/mpo.py` | MPO Agent: E-Step, M-Step, Critic, Dual Variables, Save/Load |
| `agent/networks.py` | GaussianPolicy (tanh-squashed) + Twin QNetwork |
| `agent/replay_buffer.py` | Replay Buffer mit state_dict/load_state_dict |
| `train.py` | Trainingsskript mit MLflow-Logging, Resume, CLI-Hyperparametern |
| `test.py` | Testskript mit dm_control Viewer |
| `hpo.py` | Optuna + MLflow Hyperparameter-Optimierung |
| `environments/` | Custom dm_control-Umgebungen (cartpole_ball, one_joint_ball) |
| `tests/` | Unit-Tests (networks, replay_buffer, mpo) |

## Tests

```bash
python -m pytest tests/ -v
```

## Hyperparameter (Defaults)

| Parameter | Default | Paper |
|---|---|---|
| `critic_lr` | 5e-4 | 5e-4 |
| `actor_lr` | 5e-4 | 5e-4 |
| `dual_lr` | 1e-3 | — |
| `gamma` | 0.99 | 0.99 |
| `polyak` | 0.995 | — |
| `num_action_samples` | 20 | 20 |
| `eps_eta` | 0.1 | 0.1 |
| `eps_mu` | 0.1 | 0.1 |
| `eps_sigma` | 1e-4 | 1e-4 |
| `num_critic_updates` | 10 | — |
| `num_actor_updates` | 10 | — |
| `batch_size` | 256 | 256 |
| `update_every` | 100 | — |

## Bekannte Einschränkungen

- **Retrace** nicht implementiert (Paper Section 4) — aktuell 1-Step TD
- **Diagonal covariance** statt full covariance (Paper verwendet Cholesky-Faktor)
- **Netzwerk-Größen**: 256-256 für Policy und Q (Paper: 100-100 Policy, 200-200 Q)
