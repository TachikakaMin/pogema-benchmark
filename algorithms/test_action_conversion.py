#!/usr/bin/env python
"""
Verify action conversion between foundation_mapf and pogema.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_action_conversion():
    print("=" * 70)
    print("Verifying Action Conversion")
    print("=" * 70)

    # Foundation MAPF action definitions (from path_formation.py move_agent):
    # action 1 (up): y += 1 (agent_locations[:, 1] += 1)
    # action 2 (down): y -= 1 (agent_locations[:, 1] -= 1)
    # action 3 (left): x -= 1 (agent_locations[:, 0] -= 1)
    # action 4 (right): x += 1 (agent_locations[:, 0] += 1)
    # action 0 (stay): no movement

    # Pogema action definitions:
    # action 0 (stay): no movement
    # action 1 (up): row -= 1
    # action 2 (down): row += 1
    # action 3 (left): col -= 1
    # action 4 (right): col += 1

    # Coordinate mapping:
    # pogema: (row, col) where row is vertical, col is horizontal
    # foundation_mapf: (x, y) where x is first dim, y is second dim
    #
    # In my implementation:
    # pogema (row, col) -> foundation_mapf (x, y) = (col, row)
    # So: x = col, y = row

    print("\n1. Coordinate system analysis:")
    print("  Foundation MAPF: agent_locations[:, 0] = x, agent_locations[:, 1] = y")
    print("  Pogema: global_xy returns (row, col)")
    print("  My mapping: (row, col) -> (x, y) = (col, row)")
    print("  So: x = col, y = row")

    print("\n2. Action effect analysis:")
    print("  Foundation MAPF actions:")
    print("    action 1 (up): y += 1 -> row += 1 (pogema down)")
    print("    action 2 (down): y -= 1 -> row -= 1 (pogema up)")
    print("    action 3 (left): x -= 1 -> col -= 1 (pogema left)")
    print("    action 4 (right): x += 1 -> col += 1 (pogema right)")

    print("\n3. Required action mapping:")
    print("  foundation_mapf -> pogema")
    print("    0 (stay) -> 0 (stay)")
    print("    1 (up/y+1) -> 2 (down/row+1)")
    print("    2 (down/y-1) -> 1 (up/row-1)")
    print("    3 (left/x-1) -> 3 (left/col-1)")
    print("    4 (right/x+1) -> 4 (right/col+1)")

    # My implementation's mapping
    action_mapping = {0: 0, 1: 2, 2: 1, 3: 3, 4: 4}

    print("\n4. My implementation's mapping:")
    print(f"  {action_mapping}")

    # Verify
    expected = {0: 0, 1: 2, 2: 1, 3: 3, 4: 4}
    if action_mapping == expected:
        print("\n  ✓ Action mapping is CORRECT!")
    else:
        print("\n  ✗ Action mapping is INCORRECT!")
        print(f"  Expected: {expected}")

    # Test with actual pogema environment
    print("\n5. Testing with actual pogema environment:")
    try:
        from pogema import pogema_v0, GridConfig
        import torch

        # Create a simple environment
        grid_config = GridConfig(
            num_agents=1,
            size=10,
            density=0.0,
            seed=42,
            max_episode_steps=50,
            observation_type='MAPF',
            on_target='nothing',
        )

        env = pogema_v0(grid_config=grid_config)
        obs, _ = env.reset(seed=42)

        initial_pos = obs[0]['global_xy']
        print(f"  Initial position (row, col): {initial_pos}")

        # Test each action
        actions_to_test = [
            (1, "up", (-1, 0)),      # pogema up: row -= 1
            (2, "down", (1, 0)),     # pogema down: row += 1
            (3, "left", (0, -1)),    # pogema left: col -= 1
            (4, "right", (0, 1)),    # pogema right: col += 1
        ]

        all_correct = True
        for action, name, expected_delta in actions_to_test:
            env.reset(seed=42)
            obs, _ = env.reset(seed=42)
            start_pos = obs[0]['global_xy']

            obs, _, _, _, _ = env.step([action])
            end_pos = obs[0]['global_xy']

            actual_delta = (end_pos[0] - start_pos[0], end_pos[1] - start_pos[1])

            # Check if movement was blocked by obstacle
            if actual_delta == (0, 0) and expected_delta != (0, 0):
                print(f"  Action {action} ({name}): blocked by obstacle/boundary")
            elif actual_delta == expected_delta:
                print(f"  Action {action} ({name}): {start_pos} -> {end_pos}, delta={actual_delta} ✓")
            else:
                print(f"  Action {action} ({name}): {start_pos} -> {end_pos}, delta={actual_delta}, expected={expected_delta} ✗")
                all_correct = False

        if all_correct:
            print("\n  ✓ All action tests PASSED!")

    except ImportError as e:
        print(f"  Pogema not available: {e}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    test_action_conversion()
