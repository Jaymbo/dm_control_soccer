"""Dispatcher: route to test_mpo.py or test_ppo.py based on --algo.

Usage:
    python test.py --algo mpo --domain cartpole --task balance
    python test.py --algo ppo --domain cartpole --task balance --no-viewer
    python test.py --algo ppo --checkpoint checkpoints/ppo_cartpole_balance.pt

Any extra args after --algo are forwarded to the selected test script.
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
        script = os.path.join(script_dir, 'test_ppo.py')
    else:
        script = os.path.join(script_dir, 'test_mpo.py')

    cmd = [sys.executable, script] + forward_args
    result = subprocess.run(cmd, cwd=script_dir)
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
