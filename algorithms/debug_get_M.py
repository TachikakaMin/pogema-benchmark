"""
Debug script to:
1. Run a simple episode to get an unoptimized trajectory M
2. Save M to a file
3. Test opt_main.py optimization with the saved M
"""

import sys
sys.path.insert(0, '/home/yimintan/research/WAFR2026')

import json
import pickle
from pogema import pogema_v0, GridConfig
from trajectory_collector import TrajectoryCollectorWrapper


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


def create_synthetic_M_with_backtracking():
    """
    Create a synthetic trajectory M that has backtracking (inefficient movement).
    This is designed to test the optimization - agents that go A->B->C->B->A can be
    optimized to stay at A.
    """
    print("=" * 60)
    print("Creating synthetic M with backtracking for optimization testing")
    print("=" * 60)

    # Create trajectories with backtracking patterns
    # Agent 0: goes (0,0) -> (0,1) -> (0,2) -> (0,1) -> (0,0) -> stays at (0,0)
    # Agent 1: goes (2,0) -> (2,1) -> (2,2) -> (2,3) -> (2,2) -> (2,1) -> stays at (2,1)
    # Agent 2: goes (4,0) -> (4,1) -> (4,0) -> (4,1) -> (4,0) -> stays at (4,0)
    # Agent 3: goes (6,0) -> (6,1) -> (6,2) -> (6,3) -> (6,4) -> stays at (6,4) (no backtracking)

    T = 20  # trajectory length

    M = []

    # Agent 0: backtrack from (0,2) to (0,0)
    traj0 = [(0, 0), (0, 1), (0, 2), (0, 1), (0, 0)]
    traj0 += [(0, 0)] * (T + 1 - len(traj0))
    M.append(traj0)

    # Agent 1: backtrack from (2,3) to (2,1)
    traj1 = [(2, 0), (2, 1), (2, 2), (2, 3), (2, 2), (2, 1)]
    traj1 += [(2, 1)] * (T + 1 - len(traj1))
    M.append(traj1)

    # Agent 2: oscillate between (4,0) and (4,1)
    traj2 = [(4, 0), (4, 1), (4, 0), (4, 1), (4, 0)]
    traj2 += [(4, 0)] * (T + 1 - len(traj2))
    M.append(traj2)

    # Agent 3: no backtracking, just moves forward
    traj3 = [(6, 0), (6, 1), (6, 2), (6, 3), (6, 4)]
    traj3 += [(6, 4)] * (T + 1 - len(traj3))
    M.append(traj3)

    # Create a simple obstacle map (no obstacles)
    import numpy as np
    obstacles = np.zeros((10, 10), dtype=np.int32)

    print(f"Number of agents: {len(M)}")
    print(f"Trajectory length: {len(M[0])}")
    for i, traj in enumerate(M):
        print(f"Agent {i}: {traj[:10]}...")

    return M, obstacles


def create_synthetic_M_with_interactions():
    """
    Create a synthetic trajectory M where agents interact (pass through same vertices).
    This tests the dependency constraints in the ILP.
    """
    print("=" * 60)
    print("Creating synthetic M with agent interactions")
    print("=" * 60)

    T = 15  # trajectory length

    M = []

    # Agent 0: goes through (1,1) at times 2,4, then backtracks to (0,0)
    # (0,0) -> (0,1) -> (1,1) -> (1,2) -> (1,1) -> (0,1) -> (0,0)
    traj0 = [(0, 0), (0, 1), (1, 1), (1, 2), (1, 1), (0, 1), (0, 0)]
    traj0 += [(0, 0)] * (T + 1 - len(traj0))
    M.append(traj0)

    # Agent 1: also goes through (1,1) at time 3
    # (2, 1) -> (2, 1) -> (2, 1) -> (1, 1) -> (1, 0) -> (1, 0)
    traj1 = [(2, 1), (2, 1), (2, 1), (1, 1), (1, 0)]
    traj1 += [(1, 0)] * (T + 1 - len(traj1))
    M.append(traj1)

    # Agent 2: goes through (1,1) at times 5,6
    # (1, 2) -> (1, 2) -> (1, 2) -> (1, 2) -> (1, 2) -> (1, 1) -> (1, 1) -> (2, 1) -> (2, 2)
    traj2 = [(1, 2), (1, 2), (1, 2), (1, 2), (1, 2), (1, 1), (1, 1), (2, 1), (2, 2)]
    traj2 += [(2, 2)] * (T + 1 - len(traj2))
    M.append(traj2)

    # Create a simple obstacle map (no obstacles)
    import numpy as np
    obstacles = np.zeros((10, 10), dtype=np.int32)

    print(f"Number of agents: {len(M)}")
    print(f"Trajectory length: {len(M[0])}")
    for i, traj in enumerate(M):
        print(f"Agent {i}: {traj}")

    return M, obstacles


def get_unoptimized_M():
    """Run an episode and get the unoptimized trajectory M."""
    print("=" * 60)
    print("Step 1: Getting unoptimized trajectory M")
    print("=" * 60)

    # Use a more complex scenario with more agents and higher density
    config = GridConfig(
        num_agents=16,
        size=20,
        density=0.2,
        seed=123,
        max_episode_steps=128,
        observation_type='MAPF',
        on_target='nothing',
    )

    env = pogema_v0(grid_config=config)
    env = TrajectoryCollectorWrapper(env)

    algo = SimpleAlgorithm()
    algo.reset_states()

    obs, _ = env.reset(seed=config.seed)

    # Get obstacles for later validation
    obstacles = env.get_obstacles()

    step_count = 0
    while True:
        actions = algo.act(obs)
        obs, rew, terminated, truncated, infos = env.step(actions)
        step_count += 1

        if all(terminated) or all(truncated):
            break

    M = env.get_trajectories()

    print(f"Number of agents: {len(M)}")
    print(f"Trajectory length: {len(M[0])} (steps: {step_count})")
    print(f"Sample trajectory (agent 0, first 5 positions): {M[0][:5]}")
    print(f"Sample trajectory (agent 0, last 5 positions): {M[0][-5:]}")

    return M, obstacles


def save_M(M, obstacles, filename='debug_M.pkl'):
    """Save M and obstacles to a pickle file."""
    print(f"\nSaving M to {filename}...")
    with open(filename, 'wb') as f:
        pickle.dump({'M': M, 'obstacles': obstacles}, f)
    print(f"Saved successfully!")
    return filename


def compute_move_only_soc(M):
    """Compute Sum of Costs where only moves count (waits are free)."""
    total_moves = 0
    for agent_trajectory in M:
        for k in range(len(agent_trajectory) - 1):
            if agent_trajectory[k] != agent_trajectory[k + 1]:
                total_moves += 1
    return total_moves


def validate_trajectory(M, obstacles):
    """Validate trajectory feasibility."""
    N = len(M)
    if N == 0:
        return True, []

    T = len(M[0]) - 1
    VALID_MOVES = {(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)}
    errors = []

    for k in range(T + 1):
        positions_at_k = {}
        for i in range(N):
            pos = tuple(M[i][k])

            # Vertex collision check
            if pos in positions_at_k:
                errors.append(f"Vertex collision at time {k}: agents {positions_at_k[pos]} and {i} at {pos}")
            positions_at_k[pos] = i

            # Obstacle check
            x, y = pos
            try:
                if obstacles[x][y] == 1:
                    errors.append(f"Agent {i} at obstacle position {pos} at time {k}")
            except (IndexError, TypeError):
                pass

            # Adjacency check (for k > 0)
            if k > 0:
                prev_pos = M[i][k - 1]
                dx = pos[0] - prev_pos[0]
                dy = pos[1] - prev_pos[1]
                if (dx, dy) not in VALID_MOVES:
                    errors.append(f"Agent {i} invalid move from {prev_pos} to {pos} at time {k}")

        # Edge collision check
        if k < T:
            for i in range(N):
                for j in range(i + 1, N):
                    if (tuple(M[i][k]) == tuple(M[j][k + 1]) and
                        tuple(M[j][k]) == tuple(M[i][k + 1]) and
                        tuple(M[i][k]) != tuple(M[i][k + 1])):
                        errors.append(f"Edge collision at time {k}: agents {i} and {j}")

    return len(errors) == 0, errors


def test_optimization(M, obstacles):
    """Test opt_main.py optimization with the given M."""
    print("\n" + "=" * 60)
    print("Step 2: Testing optimization with opt_main.py")
    print("=" * 60)

    # Compute original SoC
    original_soc = compute_move_only_soc(M)
    print(f"Original move-only SoC: {original_soc}")

    # Validate original trajectory
    is_valid, errors = validate_trajectory(M, obstacles)
    print(f"Original trajectory valid: {is_valid}")
    if not is_valid:
        print(f"Errors: {errors[:5]}...")  # Show first 5 errors

    # Import and run optimization
    try:
        from opt_main import solve_collapsed_mapf_from_M, apply_actions_to_trajectory

        print("\nRunning ILP optimization...")
        summary, chosen_actions = solve_collapsed_mapf_from_M(
            M,
            time_limit_sec=30.0,
            threads=4,
            verbose=True
        )

        print("\n=== Optimization Summary ===")
        for k, v in summary.items():
            print(f"  {k}: {v}")

        print(f"\n=== Chosen Actions ({len(chosen_actions)} total) ===")
        for action in chosen_actions[:10]:  # Show first 10
            print(f"  {action}")
        if len(chosen_actions) > 10:
            print(f"  ... and {len(chosen_actions) - 10} more")

        # Apply actions to get optimized trajectory
        print("\n=== Applying actions to trajectory ===")
        M_opt = apply_actions_to_trajectory(M, chosen_actions)

        # Compute optimized SoC
        optimized_soc = compute_move_only_soc(M_opt)
        print(f"Optimized move-only SoC: {optimized_soc}")
        print(f"Expected from ILP: {summary['best_min_cost']}")
        print(f"SoC match: {optimized_soc == summary['best_min_cost']}")

        # Validate optimized trajectory
        is_valid_opt, errors_opt = validate_trajectory(M_opt, obstacles)
        print(f"\nOptimized trajectory valid: {is_valid_opt}")
        if not is_valid_opt:
            print(f"Validation errors ({len(errors_opt)} total):")
            for err in errors_opt[:20]:  # Show first 20 errors
                print(f"  {err}")
            if len(errors_opt) > 20:
                print(f"  ... and {len(errors_opt) - 20} more")

        # Compare trajectories
        print("\n=== Trajectory Comparison (Agent 0) ===")
        print(f"Original first 10: {M[0][:10]}")
        print(f"Optimized first 10: {M_opt[0][:10]}")

        return M_opt, summary, chosen_actions

    except ImportError as e:
        print(f"Import error: {e}")
        return None, None, None
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        return None, None, None


def main():
    print("Debug: Get M and test optimization")
    print("=" * 60)

    # Test 1: Simple backtracking (no interactions)
    print("\n\n" + "=" * 80)
    print("TEST 1: Simple backtracking (no agent interactions)")
    print("=" * 80)
    M1, obstacles1 = create_synthetic_M_with_backtracking()
    save_M(M1, obstacles1, 'debug_M_simple.pkl')
    M_opt1, summary1, chosen1 = test_optimization(M1, obstacles1)

    # Test 2: With agent interactions
    print("\n\n" + "=" * 80)
    print("TEST 2: With agent interactions (dependency constraints)")
    print("=" * 80)
    M2, obstacles2 = create_synthetic_M_with_interactions()
    save_M(M2, obstacles2, 'debug_M_interactions.pkl')
    M_opt2, summary2, chosen2 = test_optimization(M2, obstacles2)

    print("\n" + "=" * 60)
    print("Debug complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()
