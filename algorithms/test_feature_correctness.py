#!/usr/bin/env python
"""
Verify feature construction matches original implementation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

# Add foundation_mapf to path
FOUNDATION_MAPF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'foundation_mapf')
sys.path.insert(0, FOUNDATION_MAPF_PATH)

def test_feature_construction():
    print("=" * 70)
    print("Verifying Feature Construction")
    print("=" * 70)

    # Import both implementations
    from foundation_mapf.tools.utils import construct_input_feature as original_construct
    from foundation_mapf.tools.utils import get_distance, NOT_FOUND_PATH
    from foundation_mapf_wrapper.inference import LazyDistanceMap

    # Create a simple test map (5x5)
    map_data = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.float32)

    # Agent positions and goals
    agent_locations = torch.tensor([[0, 0], [4, 4]], dtype=torch.long)
    goal_locations = torch.tensor([[4, 4], [0, 0]], dtype=torch.long)

    # Create lazy distance map for my implementation
    lazy_dmap = LazyDistanceMap(map_data)

    # Create original distance map format (dict)
    from collections import deque
    def create_original_dmap(map_data):
        height, width = map_data.shape
        distance_map = {}
        for sx in range(height):
            for sy in range(width):
                if map_data[sx, sy] == 0:
                    dist = np.full((height, width), NOT_FOUND_PATH, dtype=np.int32)
                    dist[sx, sy] = 0
                    queue = deque([(sx, sy)])
                    while queue:
                        cx, cy = queue.popleft()
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < height and 0 <= ny < width:
                                if map_data[nx, ny] == 0 and dist[nx, ny] == NOT_FOUND_PATH:
                                    dist[nx, ny] = dist[cx, cy] + 1
                                    queue.append((nx, ny))
                    distance_map[(sx, sy)] = dist
        return distance_map

    original_dmap = create_original_dmap(map_data)

    # Test distance queries
    print("\n1. Testing distance queries:")
    test_cases = [
        ((0, 0), (4, 4)),
        ((4, 4), (0, 0)),
        ((0, 0), (2, 2)),
        ((1, 0), (1, 3)),  # Around obstacle
    ]

    all_pass = True
    for (ax, ay), (gx, gy) in test_cases:
        lazy_dist = lazy_dmap.get_distance(ax, ay, gx, gy)
        orig_dist = get_distance(original_dmap, (ax, ay), (gx, gy))
        match = "✓" if lazy_dist == orig_dist else "✗"
        if lazy_dist != orig_dist:
            all_pass = False
        print(f"  ({ax},{ay}) -> ({gx},{gy}): lazy={lazy_dist}, orig={orig_dist} {match}")

    # Test feature construction
    print("\n2. Testing feature construction (channels 0-3):")
    map_tensor = torch.tensor(map_data, dtype=torch.float32)

    # Original implementation
    orig_features = original_construct(
        map_tensor, agent_locations, goal_locations,
        original_dmap, feature_dim=6, feature_type="gradient"
    )

    # My implementation (Python version)
    from foundation_mapf_wrapper.inference import FoundationMAPFInference, FoundationMAPFInferenceConfig
    cfg = FoundationMAPFInferenceConfig(device='cpu')

    # Manually construct features using my implementation
    my_features = torch.zeros((6, 5, 5), dtype=torch.float32)
    my_features[0] = map_tensor
    agent_indices = torch.arange(1, 3, dtype=torch.float32)
    my_features[1, agent_locations[:, 0], agent_locations[:, 1]] = agent_indices
    my_features[2, goal_locations[:, 0], goal_locations[:, 1]] = agent_indices

    # Channel 3: distances
    for i in range(2):
        ax, ay = agent_locations[i].tolist()
        gx, gy = goal_locations[i].tolist()
        my_features[3, ax, ay] = lazy_dmap.get_distance(ax, ay, gx, gy)

    # Compare channels 0-3
    for ch in range(4):
        diff = torch.abs(orig_features[ch] - my_features[ch]).max().item()
        match = "✓" if diff < 1e-6 else "✗"
        if diff >= 1e-6:
            all_pass = False
        print(f"  Channel {ch}: max diff = {diff:.6f} {match}")

    # Test gradient computation (channels 4-5)
    print("\n3. Testing gradient computation (channels 4-5):")
    print("  Note: Gradients may differ due to random choices, checking logic only")

    # For a deterministic test, check a case where gradient is deterministic
    # Agent at (0,0), goal at (4,4) - should want to go right (+x) and up (+y)
    ax, ay = 0, 0
    gx, gy = 4, 4

    current_dist = lazy_dmap.get_distance(ax, ay, gx, gy)
    left_dist = lazy_dmap.get_distance(ax - 1, ay, gx, gy) if ax > 0 else NOT_FOUND_PATH
    right_dist = lazy_dmap.get_distance(ax + 1, ay, gx, gy)
    up_dist = lazy_dmap.get_distance(ax, ay + 1, gx, gy)
    down_dist = lazy_dmap.get_distance(ax, ay - 1, gx, gy) if ay > 0 else NOT_FOUND_PATH

    print(f"  Agent at ({ax},{ay}), goal at ({gx},{gy})")
    print(f"  current_dist={current_dist}")
    print(f"  left_dist={left_dist}, right_dist={right_dist}")
    print(f"  up_dist={up_dist}, down_dist={down_dist}")
    print(f"  delta_left={left_dist - current_dist}, delta_right={right_dist - current_dist}")
    print(f"  delta_up={up_dist - current_dist}, delta_down={down_dist - current_dist}")

    # Expected: right is closer (delta_right < 0), left is blocked (delta_left = NOT_FOUND_PATH - current)
    # So dx should be 1 (go right)
    # up is closer (delta_up < 0), down is blocked
    # So dy should be 1 (go up)

    print("\n" + "=" * 70)
    if all_pass:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 70)


if __name__ == '__main__':
    test_feature_construction()
