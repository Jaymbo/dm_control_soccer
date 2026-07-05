"""Hyperparameter Optimisation for MPO using Optuna + MLflow.

Each trial runs a short training and logs metrics to MLflow.
Optuna uses the **final eval reward** (robust mean over 10 episodes)
as optimisation objective — not the best intermediate eval.

Usage:
    python hpo.py --trials 20 --steps 20000
    python hpo.py --trials 0 --steps 50000          # endless until Ctrl+C
    python hpo.py --domain cartpole --task swingup   # different env

After HPO, view results in two dashboards:
    MLflow (metrics per trial):   mlflow ui --backend-store-uri sqlite:///mlflow.db
    Optuna (HPO search analysis): optuna-dashboard sqlite:///optuna.db
"""
import argparse
import os
import sys
import subprocess
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# SQLite busy timeout (ms) — allows multiple parallel workers to wait
# instead of immediately failing with "database is locked".
SQLITE_BUSY_TIMEOUT_MS = 30000


def _add_sqlite_busy_timeout(uri: str) -> str:
    """Append busy_timeout pragma to a sqlite:/// URI (no-op for other URIs)."""
    if uri.startswith('sqlite:///'):
        sep = '&' if '?' in uri else '?'
        if 'busy_timeout' not in uri:
            uri = f'{uri}{sep}busy_timeout={SQLITE_BUSY_TIMEOUT_MS}'
    return uri


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--trials', type=int, default=20,
                   help='Number of Optuna trials (0 = unlimited, run until Ctrl+C)')
    p.add_argument('--steps', type=int, default=20000,
                   help='Training steps per trial')
    p.add_argument('--domain', type=str, default='cartpole')
    p.add_argument('--task', type=str, default='balance')
    p.add_argument('--study_name', type=str, default='mpo_hpo')
    p.add_argument('--optuna_storage', type=str, default='sqlite:///optuna.db',
                   help='Optuna storage URI (for optuna-dashboard)')
    p.add_argument('--mlflow_tracking_uri', type=str, default='sqlite:///mlflow.db')
    p.add_argument('--mlflow_experiment', type=str, default=None,
                   help='MLflow experiment name (default: <domain>_<task>_hpo)')
    p.add_argument('--seed', type=int, default=42)
    return p


def make_objective(args):
    """Create Optuna objective that runs train.py with sampled hyperparams."""
    import optuna

    def objective(trial: optuna.Trial) -> float:
        # --- Sample hyperparameters ---
        hp = {
            'critic_lr': trial.suggest_float('critic_lr', 1e-4, 1e-3, log=True),
            'actor_lr': trial.suggest_float('actor_lr', 1e-4, 1e-3, log=True),
            'dual_lr': trial.suggest_float('dual_lr', 1e-4, 1e-2, log=True),
            'num_action_samples': trial.suggest_categorical('num_action_samples', [10, 20, 30]),
            'eps_eta': trial.suggest_float('eps_eta', 0.05, 0.2),
            'eps_mu': trial.suggest_float('eps_mu', 0.05, 0.2),
            'eps_sigma': trial.suggest_float('eps_sigma', 1e-5, 1e-3, log=True),
            'num_critic_updates': trial.suggest_int('num_critic_updates', 5, 20),
            'num_actor_updates': trial.suggest_int('num_actor_updates', 5, 20),
            'polyak': trial.suggest_float('polyak', 0.99, 0.999),
        }

        # Build CLI args for train.py
        cmd = [
            sys.executable, os.path.join(os.path.dirname(__file__), 'train.py'),
            '--domain', args.domain,
            '--task', args.task,
            '--steps', str(args.steps),
            '--seed', str(args.seed + trial.number),
            '--eval_every', str(args.steps // 4),  # 4 evals per trial
            '--print_every', str(args.steps),       # print once at end
            '--no-resume',                           # HPO trials start fresh
            '--checkpoint_tag', f'trial{trial.number}',  # unique ckpt per trial
        ]
        cmd += ['--mlflow_tracking_uri', _add_sqlite_busy_timeout(args.mlflow_tracking_uri)]
        cmd += ['--mlflow_experiment', args.mlflow_experiment]
        cmd += ['--mlflow_run_name', f'trial_{trial.number}']

        for k, v in hp.items():
            cmd += [f'--{k}', str(v)]

        # Run training as subprocess (clean process each trial)
        print(f"\n{'='*60}")
        print(f"Trial {trial.number} | steps={args.steps} | params: {hp}")
        print(f"{'='*60}")

        result = subprocess.run(cmd, capture_output=True, text=True,
                                cwd=os.path.dirname(os.path.abspath(__file__)))
        print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
        if result.returncode != 0:
            print(f"STDERR: {result.stderr[-500:]}")
            return -1e9

        # Parse FINAL_EVAL (robust final evaluation, not best intermediate)
        final_eval = -1e9
        for line in result.stdout.splitlines():
            if 'FINAL_EVAL' in line and 'mean=' in line:
                try:
                    final_eval = float(line.split('mean=')[1].split()[0])
                except (ValueError, IndexError):
                    pass

        # Store steps as trial metadata for cross-run comparison
        trial.set_user_attr('steps', args.steps)
        trial.set_user_attr('final_eval', final_eval)

        print(f"Trial {trial.number} final_eval = {final_eval:.3f} (steps={args.steps})")
        return final_eval

    return objective


def main():
    args = build_parser().parse_args()

    # Default experiment name: domain_task_hpo
    if args.mlflow_experiment is None:
        args.mlflow_experiment = f'{args.domain}_{args.task}_hpo'

    import optuna
    import mlflow

    # Append SQLite busy_timeout to URIs for parallel worker safety.
    mlflow_uri = _add_sqlite_busy_timeout(args.mlflow_tracking_uri)
    optuna_uri = args.optuna_storage
    if optuna_uri.startswith('sqlite:///'):
        optuna_uri = _add_sqlite_busy_timeout(optuna_uri)

    # MLflow DB init can race when multiple HPO workers start simultaneously.
    # Retry with backoff to handle "table already exists" errors.
    for attempt in range(5):
        try:
            mlflow.set_tracking_uri(mlflow_uri)
            mlflow.set_experiment(args.mlflow_experiment)
            break
        except Exception as e:
            if attempt < 4:
                print(f"MLflow init attempt {attempt+1} failed ({e}), retrying...")
                time.sleep(2 ** attempt)
            else:
                raise

    # RDBStorage with connect_args timeout for safe concurrent access.
    storage = optuna.storages.RDBStorage(
        optuna_uri,
        engine_kwargs={'connect_args': {'timeout': SQLITE_BUSY_TIMEOUT_MS / 1000}},
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction='maximize',
        # Vary sampler seed per worker (PID) so parallel workers explore
        # different regions. --seed is still used for training reproducibility.
        sampler=optuna.samplers.TPESampler(seed=args.seed + os.getpid()),
        load_if_exists=True,
    )

    objective = make_objective(args)
    n_trials = args.trials if args.trials > 0 else None

    if n_trials is None:
        print("Running unlimited HPO. Press Ctrl+C to stop.\n")

    # Use MLflow pruning callback (optional — requires mlflow-optuna integration)
    try:
        from mlflow_optuna import MLflowCallback
        mlflow_callback = MLflowCallback(
            tracking_uri=mlflow_uri,
            metric_name='final_eval_reward',
        )
        study.optimize(objective, n_trials=n_trials,
                       callbacks=[mlflow_callback])
    except ImportError:
        # Fall back without MLflow pruning callback
        print("mlflow-optuna not installed, running without pruning callback")
        study.optimize(objective, n_trials=n_trials)

    print("\n" + "=" * 60)
    print("HPO COMPLETE")
    print("=" * 60)
    print(f"Best trial value: {study.best_value:.3f}")
    print(f"Best params: {study.best_params}")
    print(f"\nView results:")
    print(f"  MLflow:   mlflow ui --backend-store-uri {args.mlflow_tracking_uri}")
    print(f"  Optuna:   optuna-dashboard {args.optuna_storage}")


if __name__ == '__main__':
    main()
