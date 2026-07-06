"""Dispatcher: route to hpo_mpo.py or hpo_ppo.py based on --algo.

Usage:
    python hpo.py --algo mpo --trials 20 --steps 50000
    python hpo.py --algo ppo --trials 20 --steps 50000
    python hpo.py --algo ppo --trials 0 --steps 50000   # endless

Any extra args after --algo are forwarded to the selected HPO script.
"""
import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--algo', type=str, default='mpo', choices=['mpo', 'ppo'],
                        help='Algorithm: mpo or ppo (default: mpo)')
    algo_args, forward_args = parser.parse_known_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if algo_args.algo == 'ppo':
        script = os.path.join(script_dir, 'hpo_ppo.py')
    else:
        script = os.path.join(script_dir, 'hpo_mpo.py')

    cmd = [sys.executable, script] + forward_args
    result = subprocess.run(cmd, cwd=script_dir)
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
