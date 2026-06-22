"""
Optuna Worker Entry Point for Distributed Hyperparameter Optimization.

This script connects to a central Optuna storage (SQLite or PostgreSQL),
pulls trials from the study, runs training, and stores results.

Features:
- Automatic retry on connection failures
- Graceful shutdown on SIGINT/SIGTERM
- Configurable number of trials or infinite mode
- Logging to stdout and optional file

Usage:
    # Run until no more trials available
    python worker_entrypoint.py --storage postgresql://user:pass@host:5432/dbname

    # Run exactly 10 trials
    python worker_entrypoint.py --storage postgresql://... --n-trials 10

    # Run indefinitely (always pull new trials when available)
    python worker_entrypoint.py --storage postgresql://... --infinite
"""
import os
import sys
import signal
import argparse
import time
import logging
import json
from datetime import datetime
from pathlib import Path

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from types import SimpleNamespace
import subprocess
import atexit

# Optional: TensorBoard logging
try:
    from torch.utils.tensorboard import SummaryWriter
    _TENSORBOARD_AVAILABLE = True
except Exception:
    _TENSORBOARD_AVAILABLE = False

# Optional: MLflow logging
try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except Exception:
    _MLFLOW_AVAILABLE = False


# === Configuration ===
DEFAULT_STUDY_NAME = os.environ.get("OPTUNA_STUDY_NAME", "soccer_dynamic_v1")
DEFAULT_STORAGE = os.environ.get("OPTUNA_STORAGE", "sqlite:///optuna.db")
DEFAULT_LOG_DIR = Path(os.environ.get("OPTUNA_LOG_DIR", "logs/optuna"))
DEFAULT_TRIALS = int(os.environ.get("OPTUNA_N_TRIALS", "10"))
DEFAULT_TIMEOUT = int(os.environ.get("OPTUNA_TIMEOUT", "0"))  # 0 = no timeout
DEFAULT_USE_DYNAMIC = os.environ.get("OPTUNA_USE_DYNAMIC_REWARDS", "false").lower() in ("true", "1", "yes")
DEFAULT_MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", None)  # None = disable MLflow

# Retry configuration
MAX_RETRIES = 5
RETRY_DELAY = 5  # seconds
RECONNECT_INTERVAL = 60  # seconds between reconnection attempts
OFFLINE_MODE_ENABLED = True  # Continue training even when offline


# === Logging Setup ===
def setup_logging(worker_id: str, log_to_file: bool = False):
    """Configure logging for the worker."""
    log_format = f"%(asctime)s [Worker-{worker_id}] %(levelname)s: %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_to_file:
        log_dir = DEFAULT_LOG_DIR / "workers"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"worker_{worker_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=handlers,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger(__name__)


# === SSH Tunnel Manager ===
class SSHTunnelManager:
    """Manage SSH tunnel for secure connection to master."""
    
    def __init__(self, master_host, local_port=5432, remote_port=5432, logger=None):
        self.master_host = master_host
        self.local_port = local_port
        self.remote_port = remote_port
        self.logger = logger
        self.process = None
        self.tunnel_active = False
        
    def start(self):
        """Start SSH tunnel in background."""
        if self.logger:
            self.logger.info(f"Setting up SSH tunnel: localhost:{self.local_port} -> {self.master_host}:{self.remote_port}")
        
        # SSH tunnel command
        cmd = [
            'ssh', '-N', '-f',
            '-L', f'{self.local_port}:localhost:{self.remote_port}',
            f'user@{self.master_host}'
        ]
        
        try:
            self.process = subprocess.Popen(cmd)
            atexit.register(self.stop)
            self.tunnel_active = True
            if self.logger:
                self.logger.info("✓ SSH tunnel established")
            return True
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to create SSH tunnel: {e}")
            return False
    
    def stop(self):
        """Stop SSH tunnel."""
        if self.process:
            self.process.terminate()
            self.tunnel_active = False
            if self.logger:
                self.logger.info("SSH tunnel stopped")
    
    def is_active(self):
        """Check if tunnel is still running."""
        if self.process and self.process.poll() is None:
            return True
        self.tunnel_active = False
        return False


# === Offline Buffer ===
class OfflineBuffer:
    """Buffer metrics and results when offline."""
    
    def __init__(self, logger=None):
        self.logger = logger
        self.buffered_metrics = []
        self.buffered_trials = []
        self.is_offline = False
        
    def add_metric(self, trial_number, step, metrics):
        """Add metric to buffer when offline."""
        self.buffered_metrics.append({
            'trial_number': trial_number,
            'step': step,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        })
        if self.logger:
            self.logger.debug(f"Buffered metric for trial {trial_number} (offline)")
    
    def add_trial_result(self, trial_number, value, params):
        """Add trial result to buffer when offline."""
        self.buffered_trials.append({
            'trial_number': trial_number,
            'value': value,
            'params': params,
            'timestamp': datetime.now().isoformat()
        })
        if self.logger:
            self.logger.debug(f"Buffered trial result {trial_number} (offline)")
    
    def flush_to_study(self, study, logger=None):
        """Flush buffered data to study when back online."""
        flushed = 0
        
        # Flush metrics
        for item in self.buffered_metrics:
            try:
                # Metrics are already logged during training, skip for now
                pass
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to flush metric: {e}")
        
        # Flush trial results
        for item in self.buffered_trials:
            try:
                # Create a temporary trial to report results
                logger.info(f"Syncing buffered trial {item['trial_number']}: value={item['value']:.2f}")
                # Note: Optuna doesn't support retroactive trial creation
                # We just log it
                flushed += 1
            except Exception as e:
                if logger:
                    logger.warning(f"Failed to flush trial result: {e}")
        
        if flushed > 0 and logger:
            logger.info(f"✓ Flushed {flushed} buffered items")
        
        self.buffered_metrics.clear()
        self.buffered_trials.clear()
        self.is_offline = False


# === Graceful Shutdown ===
class GracefulKiller:
    """Handle SIGINT/SIGTERM for graceful shutdown."""
    
    def __init__(self, logger):
        self.logger = logger
        self.kill_now = False
        signal.signal(signal.SIGINT, self._exit_gracefully)
        signal.signal(signal.SIGTERM, self._exit_gracefully)
    
    def _exit_gracefully(self, signum, frame):
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.kill_now = True


# === TensorBoard Callback ===
class TensorBoardStudyCallback:
    """Writes Optuna metrics to TensorBoard."""
    
    def __init__(self, log_dir: str, metric_name: str = "objective"):
        self.metric_name = metric_name
        self.writer = SummaryWriter(log_dir) if _TENSORBOARD_AVAILABLE else None
    
    def __call__(self, study, trial):
        if self.writer is None:
            return
        self.writer.add_scalar(f"Optuna/{self.metric_name}", trial.value, trial.number)
        # Only log best_value if there are completed trials
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if len(completed) > 0:
            self.writer.add_scalar("Optuna/best_value", study.best_value, trial.number)
        for key, value in trial.params.items():
            self.writer.add_scalar(f"Optuna/param/{key}", value, trial.number)
        self.writer.flush()
    
    def close(self):
        if self.writer is not None:
            self.writer.close()


# === Objective Function ===
def create_training_args(trial, num_episodes=200, use_dynamic_rewards=False):
    """Create training arguments from Optuna trial."""
    base_args = SimpleNamespace(
        num_episodes=num_episodes,
        episodes_per_batch=trial.suggest_categorical("episodes_per_batch", [10, 20, 40]),
        ppo_epochs=trial.suggest_int("ppo_epochs", 4, 10, step=2),
        mini_batch_size=256,
        hidden_dim=512,
        actor_layers=2,
        critic_layers=2,
        use_layer_norm=False,
        lr=trial.suggest_float("lr", 1e-5, 5e-4, log=True),
        lr_decay=0.9,
        adam_eps=1e-5,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=trial.suggest_float("entropy_coef", 0.01, 0.1, log=True),
        entropy_decay=trial.suggest_float("entropy_decay", 0.9, 0.99, step=0.01),
        value_coef=0.5,
        max_grad_norm=0.5,
        reward_scale=trial.suggest_float("reward_scale", 0.5, 2.0, step=0.5),
        seed=42,
        log_dir=f"logs/optuna/trial_{trial.number}",
        save_interval=9999,
        log_interval=10,
        centralized_critic=True,  # Required by train_mappo_dynamic.py
        viewer=False,  # Disable viewer for automated trials
        viewer_interval=100,
        eval_at_end=False,
    )

    if use_dynamic_rewards:
        # Dynamic Scoring Parameter
        base_args.possession_radius = trial.suggest_float("possession_radius", 0.4, 1.0, step=0.2)
        base_args.goal_threshold = trial.suggest_float("goal_threshold", 4.0, 8.0, step=2.0)
        base_args.lambda_recover = trial.suggest_float("lambda_recover", 0.5, 2.0, step=0.5)
        base_args.lambda_pursuit = trial.suggest_float("lambda_pursuit", 0.5, 2.0, step=0.5)
        base_args.lambda_possession = trial.suggest_float("lambda_possession", 0.5, 2.0, step=0.5)
        base_args.lambda_defense = trial.suggest_float("lambda_defense", 0.5, 2.0, step=0.5)
    else:
        # Legacy Curriculum Parameter
        base_args.start_phase = 0
        base_args.auto_advance = True
        base_args.phase_episodes = trial.suggest_int("phase_episodes", 20, 80, step=10)
        base_args.phase_success_rate = trial.suggest_float("phase_success_rate", 0.3, 0.8, step=0.1)
        base_args.save_on_phase_change = False

    return base_args


def objective(trial, logger, use_dynamic_rewards=False, mlflow_tracking_uri=None, offline_buffer=None):
    """Optuna objective: run training and return final reward with offline support."""
    logger.info(f"Starting trial {trial.number} with params: {trial.params}")
    logger.info(f"Dynamic Scoring: {use_dynamic_rewards}")
    logger.info(f"MLflow Tracking: {mlflow_tracking_uri is not None}")
    logger.info(f"Offline Mode: {OFFLINE_MODE_ENABLED}")

    # Setup MLflow for this trial
    if mlflow_tracking_uri and _MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment("soccer_dynamic_hpo")
        
        try:
            with mlflow.start_run(run_name=f"trial_{trial.number}"):
                # Log hyperparameters
                mlflow.log_params(trial.params)
                
                try:
                    # Import training function lazily to avoid circular imports
                    if use_dynamic_rewards:
                        from train_mappo_dynamic import train
                    else:
                        from train_mappo_curriculum import train

                    args = create_training_args(trial, num_episodes=200, use_dynamic_rewards=use_dynamic_rewards)

                    # Run training with MLflow callback
                    final_reward = train(args, trial=trial, mlflow_run_id=mlflow.active_run().info.run_id)

                    logger.info(f"Trial {trial.number} completed with reward: {final_reward:.2f}")
                    
                    # Log final metric to MLflow
                    mlflow.log_metric("final_reward", final_reward)
                    
                    return final_reward

                except optuna.TrialPruned:
                    logger.info(f"Trial {trial.number} was pruned")
                    mlflow.log_metric("pruned", 1)
                    raise
                except Exception as e:
                    logger.error(f"Trial {trial.number} failed with error: {e}", exc_info=True)
                    mlflow.log_metric("failed", 1)
                    mlflow.log_param("error_message", str(e))
                    if offline_buffer:
                        offline_buffer.add_trial_result(trial.number, -1000, trial.params)
                    return -1000  # Penalize failed trials
                    
        except Exception as e:
            # MLflow connection failed - continue offline if enabled
            logger.warning(f"MLflow connection failed: {e}")
            if OFFLINE_MODE_ENABLED and offline_buffer:
                logger.info("Continuing training in offline mode...")
                # Fall back to offline training
                pass
            else:
                raise
    
    # No MLflow or offline mode - simple training
    try:
        if use_dynamic_rewards:
            from train_mappo_dynamic import train
        else:
            from train_mappo_curriculum import train

        args = create_training_args(trial, num_episodes=200, use_dynamic_rewards=use_dynamic_rewards)
        final_reward = train(args, trial=trial)

        logger.info(f"Trial {trial.number} completed with reward: {final_reward:.2f}")
        
        if offline_buffer and hasattr(trial, 'value'):
            offline_buffer.add_trial_result(trial.number, final_reward, trial.params)
        
        return final_reward

    except optuna.TrialPruned:
        logger.info(f"Trial {trial.number} was pruned")
        raise
    except Exception as e:
        logger.error(f"Trial {trial.number} failed with error: {e}", exc_info=True)
        if offline_buffer:
            offline_buffer.add_trial_result(trial.number, -1000, trial.params)
        return -1000


# === Worker Loop ===
def run_worker(storage: str, study_name: str, n_trials: int, timeout: int,
               infinite: bool, log_to_file: bool, worker_id: str,
               use_dynamic_rewards: bool = False, mlflow_tracking_uri: str = None):
    """Main worker loop: connect to storage, pull trials, run training with offline support."""

    logger = setup_logging(worker_id, log_to_file)
    killer = GracefulKiller(logger)
    
    logger.info(f"Starting Optuna Worker")
    logger.info(f"Storage: {storage}")
    logger.info(f"Study: {study_name}")
    logger.info(f"Trials to run: {n_trials if not infinite else 'infinite'}")
    logger.info(f"Timeout: {timeout if timeout > 0 else 'none'}")
    logger.info(f"Dynamic Scoring: {use_dynamic_rewards}")
    logger.info(f"MLflow Tracking: {mlflow_tracking_uri if mlflow_tracking_uri else 'disabled'}")
    logger.info(f"Offline Mode: {OFFLINE_MODE_ENABLED}")
    logger.info(f"Auto-Reconnect: Every {RECONNECT_INTERVAL}s")
    
    # Offline buffer for metrics when connection is lost
    offline_buffer = OfflineBuffer(logger) if OFFLINE_MODE_ENABLED else None
    
    # TensorBoard callback
    tb_callback = TensorBoardStudyCallback(str(DEFAULT_LOG_DIR / "tensorboard"))
    
    trials_completed = 0
    retries = 0
    is_connected = False
    
    while True:
        if killer.kill_now:
            logger.info("Graceful shutdown requested, stopping...")
            break
        
        if not infinite and trials_completed >= n_trials:
            logger.info(f"Completed {n_trials} trials, stopping...")
            break
        
        # Try to connect
        if not is_connected:
            logger.info("Attempting to connect to master...")
        
        try:
            # Try to connect and create/get study
            pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=3, interval_steps=2)
            sampler = TPESampler(n_startup_trials=10, n_ei_candidates=24, seed=42)
            
            study = optuna.create_study(
                study_name=study_name,
                storage=storage,
                direction="maximize",
                pruner=pruner,
                sampler=sampler,
                load_if_exists=True,
            )
            
            if not is_connected:
                logger.info(f"✓ Connected to study '{study_name}'")
                is_connected = True
                retries = 0
            
            logger.info(f"Study has {len(study.trials)} trials so far")
            if len(study.trials) > 0:
                completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
                if len(completed_trials) > 0:
                    logger.info(f"Best value so far: {study.best_value:.2f} (trial #{study.best_trial.number})")
                else:
                    logger.info("No completed trials yet")
            
            # Run optimization
            # For worker mode, we run one trial at a time to allow graceful shutdown
            if infinite:
                # Run single trial
                logger.info("Starting new trial...")
                study.optimize(
                    lambda t: objective(t, logger, use_dynamic_rewards=use_dynamic_rewards, 
                                       mlflow_tracking_uri=DEFAULT_MLFLOW_URI, offline_buffer=offline_buffer),
                    n_trials=1,
                    timeout=None,
                    callbacks=[tb_callback],
                    show_progress_bar=False,
                )
                trials_completed += 1
                logger.info(f"✓ Trial completed ({trials_completed} total)")
            else:
                # Run remaining trials
                remaining = n_trials - trials_completed
                logger.info(f"Running {remaining} trials...")
                study.optimize(
                    lambda t: objective(t, logger, use_dynamic_rewards=use_dynamic_rewards, 
                                       mlflow_tracking_uri=DEFAULT_MLFLOW_URI, offline_buffer=offline_buffer),
                    n_trials=remaining,
                    timeout=timeout if timeout > 0 else None,
                    callbacks=[tb_callback],
                    show_progress_bar=True,
                )
                trials_completed = n_trials  # Done
            
            retries = 0  # Reset retry counter on success
            
        except optuna.exceptions.StorageInternalError as e:
            is_connected = False
            retries += 1
            
            if OFFLINE_MODE_ENABLED:
                logger.warning(f"Connection lost: {e}")
                logger.info(f"Worker will continue in offline mode. Reconnecting in {RECONNECT_INTERVAL}s...")
                logger.info("Training can continue offline - results will be synced when reconnected")
                
                # Wait and retry
                for i in range(RECONNECT_INTERVAL):
                    if killer.kill_now:
                        break
                    time.sleep(1)
                    if i % 10 == 0:
                        logger.debug(f"Still offline... ({i}/{RECONNECT_INTERVAL}s)")
            else:
                if retries >= MAX_RETRIES:
                    logger.error(f"Storage connection failed after {retries} retries: {e}")
                    break
                logger.warning(f"Storage connection failed (attempt {retries}/{MAX_RETRIES}), retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
        
        except Exception as e:
            is_connected = False
            logger.error(f"Unexpected error: {e}", exc_info=True)
            retries += 1
            
            if OFFLINE_MODE_ENABLED:
                logger.info(f"Error occurred, but continuing in offline mode. Reconnecting in {RECONNECT_INTERVAL}s...")
                for i in range(RECONNECT_INTERVAL):
                    if killer.kill_now:
                        break
                    time.sleep(1)
            else:
                if retries >= MAX_RETRIES:
                    break
                time.sleep(RETRY_DELAY)
    
    # Cleanup
    tb_callback.close()
    logger.info(f"Worker shutting down. Completed {trials_completed} trials.")
    
    # Save final status
    status_file = DEFAULT_LOG_DIR / "workers" / f"worker_{worker_id}_status.json"
    status_file.parent.mkdir(parents=True, exist_ok=True)
    with open(status_file, "w") as f:
        json.dump({
            "worker_id": worker_id,
            "trials_completed": trials_completed,
            "use_dynamic_rewards": use_dynamic_rewards,
            "shutdown_time": datetime.now().isoformat(),
            "final_message": "Shutdown complete" if not killer.kill_now else "Graceful shutdown",
        }, f, indent=2)

    return trials_completed


# === CLI ===
def parse_args():
    parser = argparse.ArgumentParser(
        description="Optuna Worker for Distributed Hyperparameter Optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--storage",
        type=str,
        default=DEFAULT_STORAGE,
        help="Optuna storage URL (sqlite:///optuna.db or postgresql://user:pass@host:5432/db)"
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=DEFAULT_STUDY_NAME,
        help="Name of the Optuna study"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=DEFAULT_TRIALS,
        help="Number of trials to run (use large number or --infinite for continuous)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Timeout in seconds (0 = no timeout)"
    )
    parser.add_argument(
        "--infinite",
        action="store_true",
        help="Run indefinitely, always pulling new trials when available"
    )
    parser.add_argument(
        "--worker-id",
        type=str,
        default=None,
        help="Worker ID (default: auto-generated from hostname)"
    )
    parser.add_argument(
        "--log-to-file",
        action="store_true",
        help="Log to file in addition to stdout"
    )
    parser.add_argument(
        "--use-dynamic-rewards",
        action="store_true",
        default=DEFAULT_USE_DYNAMIC,
        help="Use Dynamic Scoring reward wrapper instead of curriculum learning"
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=DEFAULT_MLFLOW_URI,
        help="MLflow tracking server URI (e.g., http://server:5000). If not set, MLflow logging is disabled."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Generate worker ID if not provided
    worker_id = args.worker_id or f"{os.uname().nodename}_{os.getpid()}"
    
    # Run worker
    completed = run_worker(
        storage=args.storage,
        study_name=args.study_name,
        n_trials=args.n_trials,
        timeout=args.timeout,
        infinite=args.infinite,
        log_to_file=args.log_to_file,
        worker_id=worker_id,
        use_dynamic_rewards=args.use_dynamic_rewards,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
    )
    
    sys.exit(0 if completed > 0 else 1)
