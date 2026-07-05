"""Launch parallel HPO workers with CPU pinning and thread limiting.

Spawns N independent hpo.py processes, one per CPU core (or --workers).
Each worker is pinned to a dedicated CPU core and limited to 1 thread
to avoid oversubscription and context-switch overhead.

Usage:
    python launch_hpo.py --trials 20 --steps 20000
    python launch_hpo.py --workers 8 --trials 0 --steps 50000   # 8 of 16 cores
    python launch_hpo.py --workers 16 --trials 0                 # all cores, endless

All --trials, --steps, --domain, --task etc. are forwarded to each worker.
Optuna coordinates trial distribution automatically across workers.
"""
import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--workers', type=int, default=0,
                   help='Number of parallel workers (0 = auto-detect CPU count)')
    p.add_argument('--cpu_offset', type=int, default=0,
                   help='Pin workers to cores starting at this index (default: 0)')
    p.add_argument('--nice', type=int, default=10,
                   help='Nice priority for workers (default: 10, lower priority than normal)')
    # Unknown args are forwarded to hpo.py via parse_known_args
    return p


# Env vars that force all BLAS/torch backends to single-threaded.
THREAD_ENV = {
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'OPENBLAS_NUM_THREADS': '1',
    'NUMEXPR_NUM_THREADS': '1',
    'TORCH_NUM_THREADS': '1',
}


def pin_cpu(pid: int, core: int):
    """Pin a process to a specific CPU core via sched_setaffinity."""
    try:
        os.sched_setaffinity(pid, {core})
    except (OSError, AttributeError):
        # Fallback: not supported on this platform
        pass


def set_nice(pid: int, niceness: int = 10):
    """Lower CPU scheduling priority (higher nice = lower priority).

    Default nice=10 means other processes get CPU time first.
    Range: -20 (highest) to 19 (lowest). Regular users can raise but not lower.
    """
    try:
        os.nice(0)  # no-op, just to ensure os.nice is available
        # os.nice only affects the calling process, so use sched_setscheduler
        # via /proc or psutil. Simpler: use renice via subprocess.
        import ctypes
        libc = ctypes.CDLL('libc.so.6', use_errno=True)
        # setpriority(PRIO_PROCESS, pid, nice)
        PRIO_PROCESS = 0
        ret = libc.setpriority(PRIO_PROCESS, pid, niceness)
        if ret != 0:
            errno = ctypes.get_errno()
            print(f"  WARNING: setpriority failed (errno={errno}), "
                  f"worker {pid} runs at default priority")
    except Exception:
        pass


def main():
    parser = build_parser()
    args, hpo_args = parser.parse_known_args()

    n_workers = args.workers if args.workers > 0 else os.cpu_count()
    n_cpus = os.cpu_count() or 1
    if n_workers > n_cpus:
        print(f"WARNING: {n_workers} workers requested but only {n_cpus} CPUs available. "
              f"Limiting to {n_cpus}.")
        n_workers = n_cpus

    print(f"Starting {n_workers} parallel HPO workers on {n_cpus} CPUs")
    print(f"Each worker pinned to a dedicated core (offset={args.cpu_offset})")
    print(f"Thread limit: 1 per worker (no oversubscription)")
    print(f"Nice priority: {args.nice} (lower priority than normal processes)")
    if hpo_args:
        print(f"HPO args: {' '.join(hpo_args)}")
    print()

    hpo_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hpo.py')
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hpo_logs')
    os.makedirs(log_dir, exist_ok=True)

    # Build environment for workers: single-threaded + inherit current env
    worker_env = os.environ.copy()
    worker_env.update(THREAD_ENV)

    processes = []
    try:
        for i in range(n_workers):
            core = args.cpu_offset + i
            log_path = os.path.join(log_dir, f'worker_{i}.log')
            cmd = [sys.executable, hpo_script] + hpo_args
            with open(log_path, 'w') as f:
                proc = subprocess.Popen(
                    cmd,
                    env=worker_env,
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                    stdout=f,
                    stderr=subprocess.STDOUT,
                )
            pin_cpu(proc.pid, core)
            set_nice(proc.pid, args.nice)
            processes.append((i, core, proc, log_path))
            print(f"  Worker {i} -> PID {proc.pid}, CPU core {core}, nice {args.nice}, log -> {log_path}")
            # Stagger start to reduce DB init contention
            time.sleep(1)

        print(f"\nAll {n_workers} workers started. Press Ctrl+C to stop all.")
        print(f"Logs: tail -f {log_dir}/worker_*.log\n")

        # Wait for all workers to finish
        while any(proc.poll() is None for _, _, proc, _ in processes):
            for idx, core, proc, log_path in processes:
                rc = proc.poll()
                if rc is not None and rc != 0 and not getattr(proc, '_reported', False):
                    proc._reported = True
                    print(f"  Worker {idx} (core {core}) exited with code {rc} -> {log_path}")
            time.sleep(5)

        print("\nAll workers finished.")

    except KeyboardInterrupt:
        print("\nCtrl+C received. Terminating all workers...")
        for idx, core, proc, _ in processes:
            if proc.poll() is None:
                proc.terminate()
        # Give them 5s to clean up
        for idx, core, proc, _ in processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                print(f"  Worker {idx} (core {core}) killed")
        print("All workers stopped.")


if __name__ == '__main__':
    main()
