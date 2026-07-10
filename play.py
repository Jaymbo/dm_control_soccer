"""Manually control any dm_control environment with the keyboard.

For each action dimension, two keys are assigned (positive / negative).
Held keys produce +1 / -1 in the respective dimension; releasing returns to 0.

Usage:
    python play.py --domain cartpole --task balance
    python play.py --domain cartpole_ball --task kick
    python play.py --domain cheetah --task run

The key mapping is printed at startup.
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dm_control.viewer import application, user_input


# ---------------------------------------------------------------------------
# Key layout: pairs of (positive_key, negative_key) for each action dimension.
# Uses GLFW key constants from dm_control.viewer.user_input.
# Row 1:  1-0 (positive)  /  Q-P (negative)   -> dims 0-9
# Row 2:  A-L (positive)   /  Z-' (negative)    -> dims 10-20
# ---------------------------------------------------------------------------
_KEY_PAIRS = [
    # dim 0-9:  number row (positive) / QWERTY row (negative)
    (user_input.KEY_1,    user_input.KEY_Q),
    (user_input.KEY_2,    user_input.KEY_W),
    (user_input.KEY_3,    user_input.KEY_E),
    (user_input.KEY_4,    user_input.KEY_R),
    (user_input.KEY_5,    user_input.KEY_T),
    (user_input.KEY_6,    user_input.KEY_Y),
    (user_input.KEY_7,    user_input.KEY_U),
    (user_input.KEY_8,    user_input.KEY_I),
    (user_input.KEY_9,    user_input.KEY_O),
    (user_input.KEY_0,    user_input.KEY_P),
    # dim 10-20:  ASDF row (positive) / ZXCV row (negative)
    (user_input.KEY_A,           user_input.KEY_Z),
    (user_input.KEY_S,           user_input.KEY_X),
    (user_input.KEY_D,           user_input.KEY_C),
    (user_input.KEY_F,           user_input.KEY_V),
    (user_input.KEY_G,           user_input.KEY_B),
    (user_input.KEY_H,           user_input.KEY_N),
    (user_input.KEY_J,           user_input.KEY_M),
    (user_input.KEY_K,           user_input.KEY_COMMA),
    (user_input.KEY_L,           user_input.KEY_PERIOD),
    (user_input.KEY_SEMICOLON,   user_input.KEY_SLASH),
    (user_input.KEY_APOSTROPHE,  user_input.KEY_RIGHT_BRACKET),
]

# Human-readable names for printing
_KEY_NAMES = {
    user_input.KEY_1: '1', user_input.KEY_2: '2', user_input.KEY_3: '3',
    user_input.KEY_4: '4', user_input.KEY_5: '5', user_input.KEY_6: '6',
    user_input.KEY_7: '7', user_input.KEY_8: '8', user_input.KEY_9: '9',
    user_input.KEY_0: '0',
    user_input.KEY_Q: 'Q', user_input.KEY_W: 'W', user_input.KEY_E: 'E',
    user_input.KEY_R: 'R', user_input.KEY_T: 'T', user_input.KEY_Y: 'Z',  # QWERTZ
    user_input.KEY_U: 'U', user_input.KEY_I: 'I', user_input.KEY_O: 'O',
    user_input.KEY_P: 'P',
    user_input.KEY_A: 'A', user_input.KEY_S: 'S', user_input.KEY_D: 'D',
    user_input.KEY_F: 'F', user_input.KEY_G: 'G', user_input.KEY_H: 'H',
    user_input.KEY_J: 'J', user_input.KEY_K: 'K', user_input.KEY_L: 'L',
    user_input.KEY_Z: 'Y',  # QWERTZ: GLFW KEY_Z = German Y key
    user_input.KEY_X: 'X', user_input.KEY_C: 'C',
    user_input.KEY_V: 'V', user_input.KEY_B: 'B', user_input.KEY_N: 'N',
    user_input.KEY_M: 'M',
    user_input.KEY_COMMA: ',', user_input.KEY_PERIOD: '.',
    user_input.KEY_SEMICOLON: ';', user_input.KEY_SLASH: '/',
    user_input.KEY_APOSTROPHE: "'", user_input.KEY_RIGHT_BRACKET: ']',
}

# GLFW action constants (same values as glfw.PRESS / glfw.RELEASE)
_PRESS = 1
_RELEASE = 0


def make_env(domain, task):
    """Load environment from dm_control suite or custom suite."""
    builtin_domains = (
        'cartpole', 'cheetah', 'hopper', 'walker', 'pendulum', 'fish',
        'humanoid', 'point_mass', 'reacher', 'finger', 'manipulator',
        'acrobot', 'ball_in_cup', 'dog', 'humanoid_CMU', 'lqr',
    )
    if domain in builtin_domains:
        from dm_control import suite
        return suite.load(domain, task)
    else:
        import environments.suite as suite
        return suite.load(domain, task)


class ManualController:
    """Tracks held keys and produces actions for the policy callback."""

    def __init__(self, act_dim, act_min, act_max, keyboard):
        self._act_dim = act_dim
        self._act_min = act_min
        self._act_max = act_max
        self._held = set()  # GLFW key codes currently held down
        self._pairs = _KEY_PAIRS[:act_dim]

        # Subscribe to raw keyboard events (press AND release)
        keyboard.on_key += self._on_key

    def _on_key(self, key, scancode, action, modifiers):
        """Callback for keyboard events from GlfwKeyboard."""
        if action == _PRESS:
            self._held.add(key)
        elif action == _RELEASE:
            self._held.discard(key)

    def __call__(self, time_step):
        """Policy callback: returns action array based on held keys."""
        action = np.zeros(self._act_dim, dtype=np.float32)
        for i, (pos_key, neg_key) in enumerate(self._pairs):
            if pos_key in self._held:
                action[i] += 1.0
            if neg_key in self._held:
                action[i] -= 1.0
        return np.clip(action, self._act_min, self._act_max)

    def reset(self):
        self._held.clear()


def print_key_mapping(act_dim):
    """Print a human-readable table of key -> action dimension -> direction."""
    print("\n" + "=" * 55)
    print("  MANUAL CONTROL - Key Mapping")
    print("=" * 55)
    print(f"  {'Dim':>3}  {'+ (positive)':^14}  {'- (negative)':^14}")
    print("-" * 55)
    for i in range(act_dim):
        pos_key, neg_key = _KEY_PAIRS[i]
        pos_name = _KEY_NAMES.get(pos_key, f'KEY_{pos_key}')
        neg_name = _KEY_NAMES.get(neg_key, f'KEY_{neg_key}')
        print(f"  {i:>3}  {pos_name:^14}  {neg_name:^14}")
    print("-" * 55)
    print("  Hold a key to apply +1/-1 in that dimension.")
    print("  Release to return to 0 (neutral).")
    print("  Press multiple keys simultaneously for combined actions.")
    print("  ESC closes the viewer.")
    print("=" * 55 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Manually control a dm_control environment with keyboard.")
    parser.add_argument('--domain', type=str, default='cartpole_ball',
                        help='Environment domain (default: cartpole_ball)')
    parser.add_argument('--task', type=str, default='kick',
                        help='Environment task (default: kick)')
    args = parser.parse_args()

    env = make_env(args.domain, args.task)

    spec = env.action_spec()
    act_dim = int(np.prod(spec.shape))
    act_min = np.broadcast_to(spec.minimum, spec.shape).astype(np.float32)
    act_max = np.broadcast_to(spec.maximum, spec.shape).astype(np.float32)

    if act_dim > len(_KEY_PAIRS):
        print(f"WARNING: Environment has {act_dim} action dimensions but only "
              f"{len(_KEY_PAIRS)} key pairs are defined. "
              f"Only the first {len(_KEY_PAIRS)} dimensions will be controllable.")

    print_key_mapping(min(act_dim, len(_KEY_PAIRS)))

    # Build the application manually so we can access the keyboard
    app = application.Application(title=f'Play: {args.domain}/{args.task}')
    controller = ManualController(act_dim, act_min, act_max,
                                  keyboard=app._window.keyboard)
    app.launch(environment_loader=env, policy=controller)


if __name__ == '__main__':
    main()
