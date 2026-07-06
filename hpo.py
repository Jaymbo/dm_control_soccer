"""Hyperparameter Optimisation for MPO using Optuna + MLflow.

Each trial runs a short training and logs metrics to MLflow.
Optuna uses the **penalised final eval reward** as objective:

    objective = final_eval_reward - step_penalty * (total_steps / 1000)

This encourages the search to prefer trials that reach good performance
in fewer steps.  Intermediate eval results are reported to Optuna for
pruning (MedianPruner), so trials that lag behind are killed early.

The number of training steps per trial is itself sampled (log-uniform)
between --min_steps and --steps (upper bound).

Usage:
    python hpo.py --trials 20 --steps 50000
    python hpo.py --trials 0 --steps 50000          # endless until Ctrl+C
    python hpo.py --domain cartpole --task swingup   # different env
    python hpo.py --steps 50000 --step_penalty 2.0   # stronger time penalty

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
    p.add_argument('--steps', type=int, default=50000,
                   help='Max training steps per trial (upper bound for sampling)')
    p.add_argument('--min_steps', type=int, default=1000,
                   help='Min training steps per trial (lower bound for sampling)')
    p.add_argument('--eval_every', type=int, default=1000,
                   help='Eval interval (steps) for intermediate pruning signals')
    p.add_argument('--step_penalty', type=float, default=5.0,
                   help='Penalty per 1000 steps subtracted from objective '
                        '(encourages fewer steps)')
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

        # Sample total steps (log-uniform) — the optimiser can explore
        # short and long training runs.  The step_penalty in the objective
        # discourages unnecessarily long trials.
        trial_steps = trial.suggest_int(
            'steps', args.min_steps, args.steps, log=True
        )

        # Build CLI args for train.py
        cmd = [
            sys.executable, os.path.join(os.path.dirname(__file__), 'train.py'),
            '--domain', args.domain,
            '--task', args.task,
            '--steps', str(trial_steps),
            '--seed', str(args.seed + trial.number),
            '--eval_every', str(args.eval_every),
            '--print_every', str(trial_steps),       # print once at end
            '--no-resume',                           # HPO trials start fresh
            '--checkpoint_tag', f'trial{trial.number}',  # unique ckpt per trial
        ]
        cmd += ['--mlflow_tracking_uri', _add_sqlite_busy_timeout(args.mlflow_tracking_uri)]
        cmd += ['--mlflow_experiment', args.mlflow_experiment]
        cmd += ['--mlflow_run_name', f'trial_{trial.number}']

        for k, v in hp.items():
            cmd += [f'--{k}', str(v)]

        # Run training as subprocess (clean process each trial).
        # We read stdout line-by-line so we can parse intermediate EVAL
        # results and report them to Optuna for pruning decisions.
        # The penalty is applied to intermediate values too, so that
        # pruning decisions account for the time cost.
        print(f"\n{'='*60}")
        print(f"Trial {trial.number} | steps={trial_steps} | eval_every={args.eval_every} | "
              f"step_penalty={args.step_penalty} | params: {hp}")
        print(f"{'='*60}")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )

        eval_step = 0
        final_eval = -1e9

        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                print(line, flush=True)

            # Parse intermediate EVAL lines and report to Optuna.
            # We apply the step penalty to the intermediate value so that
            # the pruner sees the penalised objective, consistent with the
            # final objective calculation.
            if 'EVAL @' in line and 'mean=' in line:
                try:
                    mean_val = float(line.split('mean=')[1].split()[0])
                    # Extract step count from "EVAL @ <N> steps"
                    eval_at_steps = int(line.split('EVAL @')[1].split('steps')[0].strip())
                except (ValueError, IndexError):
                    continue
                eval_step += 1
                penalised = mean_val - args.step_penalty * (eval_at_steps / 1000)
                trial.report(penalised, step=eval_step)
                if trial.should_prune():
                    proc.kill()
                    proc.wait()
                    raise optuna.TrialPruned(
                        f"Pruned at eval_step={eval_step} "
                        f"(raw={mean_val:.3f}, penalised={penalised:.3f}, "
                        f"steps={eval_at_steps})"
                    )

            # Parse FINAL_EVAL
            if 'FINAL_EVAL' in line and 'mean=' in line:
                try:
                    final_eval = float(line.split('mean=')[1].split()[0])
                except (ValueError, IndexError):
                    pass

        proc.wait()
        stderr_output = proc.stderr.read() if proc.stderr else ''
        if proc.returncode != 0:
            print(f"STDERR: {stderr_output[-500:]}")
            return -1e9

        # Penalised objective: reward minus time cost
        penalty = args.step_penalty * (trial_steps / 1000)
        objective_value = final_eval - penalty

        # Store metadata for cross-run comparison
        trial.set_user_attr('steps', trial_steps)
        trial.set_user_attr('final_eval', final_eval)
        trial.set_user_attr('penalty', penalty)
        trial.set_user_attr('objective_value', objective_value)

        print(f"Trial {trial.number} | final_eval={final_eval:.3f} | "
              f"penalty={penalty:.3f} | objective={objective_value:.3f} "
              f"(steps={trial_steps})")
        return objective_value

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
        # Prune trials that perform poorly compared to median of completed
        # trials at the same intermediate step.  Needs at least
        # n_warmup_steps completed trials and n_min_trials reported
        # intermediate values before pruning kicks in.
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=2,
        ),
        load_if_exists=True,
    )

    objective = make_objective(args)
    n_trials = args.trials if args.trials > 0 else None

    if n_trials is None:
        print("Running unlimited HPO. Press Ctrl+C to stop.\n")

    # Use MLflow callback from optuna.integration (logs trial params/metrics
    # to MLflow automatically).  Pruning is handled by the MedianPruner
    # configured above via trial.report() / trial.should_prune() in the
    # objective function.
    try:
        from optuna.integration.mlflow import MLflowCallback
        mlflow_callback = MLflowCallback(
            tracking_uri=mlflow_uri,
            metric_name='final_eval_reward',
        )
        study.optimize(objective, n_trials=n_trials,
                       callbacks=[mlflow_callback])
    except ImportError:
        # Fall back without MLflow callback
        print("optuna.integration.mlflow not available, "
              "running without MLflow callback")
        study.optimize(objective, n_trials=n_trials)

    print("\n" + "=" * 60)
    print("HPO COMPLETE")
    print("=" * 60)
    print(f"Best trial value (penalised): {study.best_value:.3f}")
    best = study.best_trial
    print(f"  final_eval:  {best.user_attrs.get('final_eval', '?'):.3f}")
    print(f"  steps:       {best.user_attrs.get('steps', '?')}")
    print(f"  penalty:     {best.user_attrs.get('penalty', '?'):.3f}")
    print(f"  params:      {best.params}")
    print(f"\nView results:")
    print(f"  MLflow:   mlflow ui --backend-store-uri {args.mlflow_tracking_uri}")
    print(f"  Optuna:   optuna-dashboard {args.optuna_storage}")


if __name__ == '__main__':
    main()
