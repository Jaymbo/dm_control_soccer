"""
Effiziente Hyperparameter-Optimierung mit Optuna für Curriculum MAPPO.

Features:
  - TPE Sampler: Lernt aus vorherigen Trials (smarter als Random Search)
  - Median Pruning: Stoppt schlechte Trials früh (spart ~70% Rechenzeit)
  - TensorBoard-Integration: Direkte Visualisierung ohne TensorFlow-Abhängigkeit

Usage:
  # Optuna installieren:
  pip install optuna

  # Optimierung starten (50 Trials, max 2 Stunden):
  python optimize_curriculum.py --n-trials 50 --timeout 7200

  # TensorBoard für Ergebnisse:
  tensorboard --logdir logs/optuna/tensorboard

  # Optuna Dashboard:
  optuna-dashboard sqlite:///optuna.db

Hyperparameter die optimiert werden:
  - phase_episodes: Wie lange pro Phase (20-80)
  - phase_success_rate: Erfolgsrate für Phasenwechsel (0.3-0.8)
  - entropy_coef: Exploration (0.01-0.1)
  - lr: Learning Rate (1e-5 bis 5e-4)
  - episodes_per_batch: Batch-Größe (10, 20, 40)
  - reward_scale: Reward-Skalierung (0.5-2.0)
"""
import os
import argparse
import json
import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from types import SimpleNamespace

try:
    from torch.utils.tensorboard import SummaryWriter
    _TENSORBOARD_AVAILABLE = True
except Exception:
    _TENSORBOARD_AVAILABLE = False


class TensorBoardStudyCallback:
    """
    Schreibt Optuna-Study-Metriken und Hyperparameter in TensorBoard.
    Nutzt torch.utils.tensorboard statt optuna.integration.TensorBoardCallback,
    um die TensorFlow-Abhängigkeit zu vermeiden.
    """

    def __init__(self, log_dir: str, metric_name: str = "objective"):
        self.metric_name = metric_name
        self.writer = SummaryWriter(log_dir) if _TENSORBOARD_AVAILABLE else None

    def __call__(self, study, trial):
        if self.writer is None:
            return
        self.writer.add_scalar(f"Optuna/{self.metric_name}", trial.value, trial.number)
        self.writer.add_scalar("Optuna/best_value", study.best_value, trial.number)
        for key, value in trial.params.items():
            self.writer.add_scalar(f"Optuna/param/{key}", value, trial.number)
        self.writer.flush()

    def close(self):
        if self.writer is not None:
            self.writer.close()


def objective(trial):
    """Optuna Objective Function - trainiert und evaluiert ein Trial."""
    
    # === HYPERPARAMETER SUGGESTIONS ===
    
    # Curriculum-Parameter
    phase_episodes = trial.suggest_int("phase_episodes", 20, 80, step=10)
    phase_success_rate = trial.suggest_float("phase_success_rate", 0.3, 0.8, step=0.1)
    
    # PPO-Hyperparameter
    entropy_coef = trial.suggest_float("entropy_coef", 0.01, 0.1, log=True)
    entropy_decay = trial.suggest_float("entropy_decay", 0.9, 0.99, step=0.01)
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    episodes_per_batch = trial.suggest_categorical("episodes_per_batch", [10, 20, 40])
    ppo_epochs = trial.suggest_int("ppo_epochs", 4, 10, step=2)
    
    # Reward-Shaping
    reward_scale = trial.suggest_float("reward_scale", 0.5, 2.0, step=0.5)
    
    # === TRAINING CONFIG ===
    num_episodes = 400  # Weniger für schnelle Bewertung
    log_dir = f"logs/optuna/trial_{trial.number}"
    
    # Importiere train-Funktion
    from train_mappo_curriculum import train, set_seed
    
    args = SimpleNamespace(
        num_episodes=num_episodes,
        episodes_per_batch=episodes_per_batch,
        ppo_epochs=ppo_epochs,
        mini_batch_size=256,
        hidden_dim=512,
        actor_layers=2,
        critic_layers=2,
        use_layer_norm=False,
        lr=lr,
        lr_decay=0.9,
        adam_eps=1e-5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=entropy_coef,
        entropy_decay=entropy_decay,
        value_coef=0.5,
        max_grad_norm=0.5,
        start_phase=0,
        auto_advance=True,
        phase_episodes=phase_episodes,
        phase_success_rate=phase_success_rate,
        save_on_phase_change=False,
        reward_scale=reward_scale,
        seed=42,
        log_dir=log_dir,
        save_interval=9999,
        log_interval=10,
    )
    
    try:
        # Training starten
        final_reward = train(args, trial=trial)
        
        # Return: Durchschnittlicher Reward der letzten 100 Episoden
        return final_reward
        
    except optuna.TrialPruned:
        # Trial wurde früh gestoppt
        raise
    except Exception as e:
        # Fehlerhafte Trials bestrafen
        print(f"  Trial {trial.number} failed: {e}")
        return -1000


def main():
    parser = argparse.ArgumentParser(
        description="Optuna Hyperparameter Optimization for Curriculum MAPPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--n-trials", type=int, default=50, 
                        help="Number of trials to run")
    parser.add_argument("--timeout", type=int, default=7200, 
                        help="Timeout in seconds (default: 2h)")
    parser.add_argument("--n-startup-trials", type=int, default=10, 
                        help="Random startup trials before TPE learning")
    parser.add_argument("--storage", type=str, default="sqlite:///optuna.db", 
                        help="Database URL for study storage (sqlite:///optuna.db or postgresql://...)")
    parser.add_argument("--study-name", type=str, default="soccer_curriculum_v1", 
                        help="Name of the study")
    parser.add_argument("--direction", type=str, default="maximize", 
                        choices=["maximize", "minimize"])
    parser.add_argument("--retry", type=int, default=3,
                        help="Number of retries on storage connection failure")
    parser.add_argument("--retry-delay", type=int, default=10,
                        help="Delay between retries in seconds")
    
    args = parser.parse_args()
    
    # Verzeichnisse erstellen
    os.makedirs("logs/optuna", exist_ok=True)
    
    # === OPTUNA KONFIGURATION ===
    
    # Pruner: Median Pruning (stoppt schlechte Trials früh)
    pruner = MedianPruner(
        n_startup_trials=args.n_startup_trials,
        n_warmup_steps=3,
        interval_steps=2,
    )
    
    # Sampler: TPE (Tree-structured Parzen Estimator)
    sampler = TPESampler(
        n_startup_trials=args.n_startup_trials,
        n_ei_candidates=24,
        seed=42,
    )
    
    # Study erstellen mit Retry-Logik
    study = None
    for attempt in range(args.retry + 1):
        try:
            study = optuna.create_study(
                study_name=args.study_name,
                storage=args.storage,
                direction=args.direction,
                pruner=pruner,
                sampler=sampler,
                load_if_exists=True,
            )
            break
        except optuna.exceptions.StorageInternalError as e:
            if attempt < args.retry:
                print(f"[WARN] Storage connection failed (attempt {attempt+1}/{args.retry+1}): {e}")
                print(f"  Retrying in {args.retry_delay}s...")
                import time
                time.sleep(args.retry_delay)
            else:
                print(f"[ERROR] Storage connection failed after {args.retry+1} attempts")
                raise
    
    print(f"[INFO] Connected to study '{args.study_name}' at {args.storage}")
    
    # TensorBoard Callback (eigene Implementierung ohne TensorFlow)
    tb_callback = TensorBoardStudyCallback("logs/optuna/tensorboard", metric_name="avg_reward")

    # === START OPTIMIZATION ===
    print("\n" + "="*70)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION - Curriculum MAPPO Soccer")
    print("="*70)
    print(f"Trials:           {args.n_trials}")
    print(f"Timeout:          {args.timeout}s ({args.timeout/3600:.1f}h)")
    print(f"Pruner:           MedianPruner (n_startup={args.n_startup_trials})")
    print(f"Sampler:          TPE (learns from previous trials)")
    print(f"Storage:          {args.storage}")
    print(f"Study Name:       {args.study_name}")
    print("="*70)
    print("\nHyperparameter Search Space:")
    print("  phase_episodes:     20-80 (step=10)")
    print("  phase_success_rate: 0.3-0.8 (step=0.1)")
    print("  entropy_coef:       0.01-0.1 (log)")
    print("  entropy_decay:      0.9-0.99 (step=0.01)")
    print("  lr:                 1e-5 to 5e-4 (log)")
    print("  episodes_per_batch: 10, 20, 40")
    print("  ppo_epochs:         4-10 (step=2)")
    print("  reward_scale:       0.5-2.0 (step=0.5)")
    print("="*70 + "\n")
    
    # Optimierung starten
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        callbacks=[tb_callback],
        show_progress_bar=True,
    )
    
    # Callback aufräumen
    tb_callback.close()

    # === ERGEBNISSE ===
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"Total Trials:  {len(study.trials)}")
    print(f"Pruned Trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"Complete Trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"\nBest Trial: #{study.best_trial.number}")
    print(f"Best Value: {study.best_value:.2f}")
    print("\nBest Hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print("="*70)

    # Beste Parameter speichern
    best_params_path = "logs/optuna/best_params.json"
    with open(best_params_path, "w") as f:
        json.dump({
            "study_name": args.study_name,
            "best_trial": study.best_trial.number,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "total_trials": len(study.trials),
            "pruned_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        }, f, indent=2)
    print(f"\nBest parameters saved to: {best_params_path}")
    
    # Nächste Schritte
    print("\n" + "="*70)
    print("NEXT STEPS")
    print("="*70)
    print("1. TensorBoard für Visualisierung:")
    print("   tensorboard --logdir logs/optuna/tensorboard")
    print()
    print("2. Optuna Dashboard (interaktiv):")
    print("   pip install optuna-dashboard")
    print("   optuna-dashboard {args.storage}")
    print()
    print("3. Training mit besten Parametern:")
    print(f"   python train_mappo_curriculum.py " + " ".join([
        f"--{k.replace('_', '-')}" + (f" {v}" if not isinstance(v, bool) else "")
        for k, v in study.best_params.items()
    ]))
    print("="*70)


if __name__ == "__main__":
    main()
