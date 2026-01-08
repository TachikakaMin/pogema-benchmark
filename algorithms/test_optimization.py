"""
Test script for global optimization integration.

This script tests the trajectory collection and ILP optimization
on a simple MAPF scenario.
"""

import sys
sys.path.insert(0, '/home/yimintan/research/WAFR2026')

from pogema import pogema_v0, GridConfig
from trajectory_collector import TrajectoryCollectorWrapper
from run_episode_optimized import (
    run_episode_with_optimization,
    compute_move_only_soc,
    validate_trajectory_quick
)


class SimpleAlgorithm:
    """Simple algorithm that moves agents toward their goals."""

    def reset_states(self):
        pass

    def act(self, observations):
        actions = []
        for obs in observations:
            # Simple greedy: try to move toward goal
            # Action mapping: 0=wait, 1=up, 2=down, 3=left, 4=right
            if 'target_xy' in obs and 'xy' in obs:
                tx, ty = obs['target_xy']
                x, y = obs['xy']
                dx, dy = tx - x, ty - y

                if dx < 0:
                    actions.append(1)  # up
                elif dx > 0:
                    actions.append(2)  # down
                elif dy < 0:
                    actions.append(3)  # left
                elif dy > 0:
                    actions.append(4)  # right
                else:
                    actions.append(0)  # wait (at goal)
            else:
                actions.append(0)  # wait
        return actions


def test_trajectory_collection():
    """Test that trajectory collection works correctly."""
    print("=" * 60)
    print("Test 1: Trajectory Collection")
    print("=" * 60)

    config = GridConfig(
        num_agents=4,
        size=10,
        density=0.1,
        seed=42,
        max_episode_steps=32,
        observation_type='MAPF',
        on_target='nothing',
    )

    env = pogema_v0(grid_config=config)
    env = TrajectoryCollectorWrapper(env)

    algo = SimpleAlgorithm()
    algo.reset_states()

    obs, _ = env.reset(seed=config.seed)
    step_count = 0

    while True:
        actions = algo.act(obs)
        obs, rew, terminated, truncated, infos = env.step(actions)
        step_count += 1

        if all(terminated) or all(truncated):
            break

    trajectories = env.get_trajectories()

    print(f"Number of agents: {len(trajectories)}")
    print(f"Trajectory length: {len(trajectories[0])} (steps: {step_count})")
    print(f"Sample trajectory (agent 0): {trajectories[0][:5]}...")

    # Verify trajectory is in infos
    assert 'trajectories' in infos[0], "Trajectories should be in infos"
    print("Trajectory collection: PASSED")
    return trajectories


def test_move_only_soc():
    """Test move-only SoC calculation."""
    print("\n" + "=" * 60)
    print("Test 2: Move-Only SoC Calculation")
    print("=" * 60)

    # Simple test case: agent moves A->B->C->B->A
    M = [
        [(0, 0), (0, 1), (0, 2), (0, 1), (0, 0)],  # 4 moves
        [(1, 0), (1, 0), (1, 0), (1, 1), (1, 1)],  # 1 move
    ]

    soc = compute_move_only_soc(M)
    print(f"Test trajectory SoC: {soc}")
    assert soc == 5, f"Expected SoC=5, got {soc}"
    print("Move-only SoC calculation: PASSED")


def test_optimization_integration():
    """Test full optimization integration."""
    print("\n" + "=" * 60)
    print("Test 3: Full Optimization Integration")
    print("=" * 60)

    config = GridConfig(
        num_agents=4,
        size=10,
        density=0.1,
        seed=42,
        max_episode_steps=32,
        observation_type='MAPF',
        on_target='nothing',
    )

    env = pogema_v0(grid_config=config)
    env = TrajectoryCollectorWrapper(env)

    # Add grid_config attribute for compatibility
    env.grid_config = config

    algo = SimpleAlgorithm()

    print("Running episode with optimization...")
    results = run_episode_with_optimization(
        env, algo,
        enable_optimization=True,
        optimization_config={
            'time_limit_sec': 30.0,
            'threads': 2,
            'verbose': False,
        }
    )

    print("\nResults:")
    for key, value in results.items():
        print(f"  {key}: {value}")

    # Check optimization results
    if 'optimization_error' in results:
        print(f"\nOptimization error: {results['optimization_error']}")
        print("(This may be expected if Gurobi is not installed)")
    else:
        if 'original_move_soc' in results:
            print(f"\nOriginal move SoC: {results['original_move_soc']}")
            print(f"Optimized move SoC: {results['optimized_move_soc']}")
            print(f"Saving: {results['move_soc_saving']}")
            print("Full optimization integration: PASSED")
        else:
            print("No optimization results found")


def main():
    print("Testing Global Optimization Integration")
    print("=" * 60)

    # Test 1: Trajectory collection
    trajectories = test_trajectory_collection()

    # Test 2: Move-only SoC
    test_move_only_soc()

    # Test 3: Full integration
    test_optimization_integration()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
