"""Hyperparameter Optimisation for PPO using Optuna + MLflow.

Each trial runs a short training and logs metrics to MLflow.
Optuna uses the **penalised final eval reward** as objective:

    objective = final_eval_reward - step_penalty * (total_steps / 1000)

Intermediate eval results are reported to Optuna for pruning (MedianPruner).

Usage:
    python hpo_ppo.py --trials 20 --steps 50000
    python hpo_ppo.py --trials 0 --steps 50000          # endless until Ctrl+C
"""
import argparse
import os
import sys
import subprocess
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SQLITE_BUSY_TIMEOUT_MS = 30000


def _add_sqlite_busy_timeout(uri: str) -> str:
    if uri.startswith('sqlite:///'):
        sep = '&' if '?' in uri else '?'
        if 'busy_timeout' not in uri:
            uri = f'{uri}{sep}busy_timeout={SQLITE_BUSY_TIMEOUT_MS}'
    return uri


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--trials', type=int, default=0,
                   help='Number of Optuna trials (0 = unlimited)')
    p.add_argument('--steps', type=int, default=50000,
                   help='Max training steps per trial')
    p.add_argument('--min_steps', type=int, default=1000,
                   help='Min training steps per trial')
    p.add_argument('--eval_every', type=int, default=1000,
                   help='Eval interval for intermediate pruning')
    p.add_argument('--step_penalty', type=float, default=5.0,
                   help='Penalty per 1000 steps subtracted from objective')
    p.add_argument('--domain', type=str, default='cartpole')
    p.add_argument('--task', type=str, default='balance')
    p.add_argument('--study_name', type=str, default='ppo_hpo')
    p.add_argument('--optuna_storage', type=str, default='sqlite:///optuna.db')
    p.add_argument('--mlflow_tracking_uri', type=str, default='sqlite:///mlflow.db')
    p.add_argument('--mlflow_experiment', type=str, default=None,
                   help='MLflow experiment name (default: <domain>_<task>_hpo)')
    p.add_argument('--seed', type=int, default=42)
    return p


def make_objective(args):
    """Create Optuna objective that runs train_ppo.py with sampled hyperparams."""
    import optuna

    def objective(trial: optuna.Trial) -> float:
        hp = {
            'actor_lr': trial.suggest_float('actor_lr', 1e-5, 1e-2, log=True),
            'critic_lr': trial.suggest_float('critic_lr', 1e-5, 1e-2, log=True),
            'clip_eps': trial.suggest_float('clip_eps', 0.1, 0.3),
            'lam': trial.suggest_float('lam', 0.9, 0.99),
            'entropy_coef': trial.suggest_float('entropy_coef', 1e-4, 0.1, log=True),
            'rollout_size': trial.suggest_categorical('rollout_size', [1024, 2048, 4096]),
            'update_epochs': trial.suggest_int('update_epochs', 3, 15),
            'num_minibatches': trial.suggest_categorical('num_minibatches', [16, 32, 64]),
        }

        trial_steps = trial.suggest_int('steps', args.min_steps, args.steps, log=True)

        # Cost factor: rollout_size scales wall-clock time
        cost_factor = hp['rollout_size'] / 2048.0

        cmd = [
            sys.executable, os.path.join(os.path.dirname(__file__), 'train_ppo.py'),
            '--domain', args.domain,
            '--task', args.task,
            '--steps', str(trial_steps),
            '--seed', str(args.seed + trial.number),
            '--eval_every', str(args.eval_every),
            '--print_every', str(trial_steps),
            '--no-resume',
            '--checkpoint_tag', f'trial{trial.number}',
        ]
        cmd += ['--mlflow_tracking_uri', _add_sqlite_busy_timeout(args.mlflow_tracking_uri)]
        cmd += ['--mlflow_experiment', args.mlflow_experiment]
        cmd += ['--mlflow_run_name', f'trial_{trial.number}']

        for k, v in hp.items():
            cmd += [f'--{k}', str(v)]

        print(f"\n{'='*60}")
        print(f"Trial {trial.number} | steps={trial_steps} | eval_every={args.eval_every} | "
              f"step_penalty={args.step_penalty} | cost_factor={cost_factor:.2f} "
              f"| params: {hp}")
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

            if 'EVAL @' in line and 'mean=' in line:
                try:
                    mean_val = float(line.split('mean=')[1].split()[0])
                    eval_at_steps = int(line.split('EVAL @')[1].split('steps')[0].strip())
                except (ValueError, IndexError):
                    continue
                eval_step += 1
                penalised = mean_val - args.step_penalty * cost_factor * (eval_at_steps / 1000)
                trial.report(penalised, step=eval_step)
                if trial.should_prune():
                    proc.kill()
                    proc.wait()
                    raise optuna.TrialPruned(
                        f"Pruned at eval_step={eval_step} "
                        f"(raw={mean_val:.3f}, penalised={penalised:.3f}, "
                        f"steps={eval_at_steps})"
                    )

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

        penalty = args.step_penalty * cost_factor * (trial_steps / 1000)
        objective_value = final_eval - penalty

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

    if args.mlflow_experiment is None:
        args.mlflow_experiment = f'{args.domain}_{args.task}_hpo'

    import optuna
    import mlflow

    mlflow_uri = _add_sqlite_busy_timeout(args.mlflow_tracking_uri)
    optuna_uri = args.optuna_storage
    if optuna_uri.startswith('sqlite:///'):
        optuna_uri = _add_sqlite_busy_timeout(optuna_uri)

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

    storage = optuna.storages.RDBStorage(
        optuna_uri,
        engine_kwargs={'connect_args': {'timeout': SQLITE_BUSY_TIMEOUT_MS / 1000}},
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=args.seed + os.getpid()),
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

    try:
        from optuna.integration.mlflow import MLflowCallback
        mlflow_callback = MLflowCallback(
            tracking_uri=mlflow_uri,
            metric_name='final_eval_reward',
        )
        study.optimize(objective, n_trials=n_trials,
                       callbacks=[mlflow_callback])
    except ImportError:
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
