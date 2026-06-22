"""Debug: Was ist in der Soccer-Observation?"""
import numpy as np
from dm_control.locomotion import soccer as dm_soccer

env = dm_soccer.load(
    team_size=2,
    time_limit=10.0,
    disable_walker_contacts=False,
    enable_field_box=True,
    terminate_on_goal=False,
    walker_type=dm_soccer.WalkerType.BOXHEAD
)

timestep = env.reset()
print("=== OBSERVATION STRUCTURE ===")
print(f"Number of players: {len(timestep.observation)}")

for i, player_obs in enumerate(timestep.observation):
    print(f"\n--- Player {i} ---")
    for key, val in player_obs.items():
        arr = np.asarray(val)
        print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
        if arr.size <= 10:
            print(f"    values: {arr.flatten()}")

# Test: Ein Step mit zufälligen Actions
print("\n=== TAKING RANDOM STEP ===")
actions = np.random.randn(12).astype(np.float32)  # 4 agents * 3 actions
timestep = env.step(actions)

print("\nNach dem Step:")
for i, player_obs in enumerate(timestep.observation):
    walker_pos = player_obs.get('walker_ego_position')
    if walker_pos is not None:
        print(f"  Player {i} walker_ego_position: {np.asarray(walker_pos).flatten()}")
    else:
        print(f"  Player {i} walker_ego_position: None")
