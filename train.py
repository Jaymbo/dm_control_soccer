"""Dispatcher: route to train_mpo.py or train_ppo.py based on --algo.

Usage:
    python train.py --algo mpo --domain cartpole --task balance --steps 100000
    python train.py --algo ppo --domain cartpole --task balance --steps 100000
    python train.py --algo ppo --steps 0   # endless

Any extra args after --algo are forwarded to the selected training script.
"""
import argparse
import os
import subprocess
import sys


def main():
    # Parse only --algo, forward everything else
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--algo', type=str, default='mpo', choices=['mpo', 'ppo'],
                        help='Algorithm: mpo or ppo (default: mpo)')
    algo_args, forward_args = parser.parse_known_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if algo_args.algo == 'ppo':
        script = os.path.join(script_dir, 'train_ppo.py')
    else:
        script = os.path.join(script_dir, 'train_mpo.py')

    cmd = [sys.executable, script] + forward_args
    print(f"Dispatching to: {script}")
    result = subprocess.run(cmd, cwd=script_dir)
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
