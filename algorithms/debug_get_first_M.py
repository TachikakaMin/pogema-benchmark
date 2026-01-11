"""
Debug script to run Follower on the first testcase and extract the trajectory M.

M structure:
- M is a list of lists
- M[i][k] = (x, y) position of agent i at timestep k
- M[i] is the full trajectory of agent i
- len(M) = number of agents
- len(M[i]) = trajectory length (T+1 positions for T steps)
"""

import sys
sys.path.insert(0, '/home/yimintan/research/WAFR2026/pogema-benchmark/algorithms')
sys.path.insert(0, '/home/yimintan/research/WAFR2026')

import pickle
from gymnasium import Wrapper
from pogema import pogema_v0, GridConfig
from trajectory_collector import TrajectoryCollectorWrapper
from follower.follower_python.inference import FollowerInference, FollowerInferenceConfig
from follower.follower_python.preprocessing import follower_preprocessor


class ProvideGlobalObstaclesWrapper(Wrapper):
    """Wrapper to provide global_obstacles in observations (needed by Follower)."""

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        # Add global_obstacles from the grid (as numpy array, Follower expects .astype())
        global_obstacles = self.env.unwrapped.grid.get_obstacles()
        observations[0]['global_obstacles'] = global_obstacles
        observations[0]['after_reset'] = True
        observations[0]['max_episode_steps'] = self.env.unwrapped.grid_config.max_episode_steps
        # Add global positions
        global_agents_xy = self.env.unwrapped.grid.get_agents_xy()
        global_targets_xy = self.env.unwrapped.grid.get_targets_xy()
        for idx, obs in enumerate(observations):
            obs['global_xy'] = global_agents_xy[idx]
            obs['global_target_xy'] = global_targets_xy[idx]
        return observations, infos


def run_first_testcase():
    """Run Follower on a simple testcase and extract M."""
    print("=" * 60)
    print("Running Follower on first testcase")
    print("=" * 60)

    # Create a simple test environment config
    grid_config = GridConfig(
        num_agents=4,
        size=10,
        density=0.1,
        seed=42,
        max_episode_steps=64,
        observation_type='POMAPF',  # Follower uses POMAPF
        on_target='nothing',
    )

    # Initialize Follower algorithm first (need config for preprocessor)
    algo_config = FollowerInferenceConfig(
        seed=42,
        device='cpu',
    )
    algo = FollowerInference(algo_config)

    # Create environment with all necessary wrappers:
    # pogema_v0 -> ProvideGlobalObstaclesWrapper -> TrajectoryCollectorWrapper -> follower_preprocessor
    base_env = pogema_v0(grid_config=grid_config)
    env_with_global = ProvideGlobalObstaclesWrapper(base_env)
    env_with_traj = TrajectoryCollectorWrapper(env_with_global)
    env = follower_preprocessor(env_with_traj, algo_config)

    # Reset environment and algorithm
    obs, info = env.reset(seed=grid_config.seed)
    algo.reset_states()

    print(f"Number of agents: {env_with_traj.get_num_agents()}")
    print(f"Initial positions: {env_with_traj.get_trajectories()[0] if env_with_traj.get_trajectories() else 'N/A'}")

    # Run episode
    step_count = 0
    while True:
        # Get actions from Follower (obs is already preprocessed)
        actions = algo.act(obs)

        # Step environment
        obs, rew, terminated, truncated, infos = env.step(actions)
        step_count += 1

        if all(terminated) or all(truncated):
            break

    # Get the trajectory M from the trajectory collector wrapper
    M = env_with_traj.get_trajectories()
    obstacles = env_with_traj.get_obstacles()

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Episode finished in {step_count} steps")
    print(f"Number of agents: {len(M)}")
    print(f"Trajectory length: {len(M[0])} (T+1 positions)")

    print("\n--- M structure ---")
    print(f"type(M) = {type(M)}")
    print(f"type(M[0]) = {type(M[0])}")
    print(f"type(M[0][0]) = {type(M[0][0])}")

    print("\n--- Agent trajectories ---")
    for i, traj in enumerate(M):
        print(f"\nAgent {i}:")
        print(f"  Start: {traj[0]}")
        print(f"  End:   {traj[-1]}")
        print(f"  First 10 positions: {traj[:10]}")
        if len(traj) > 10:
            print(f"  Last 5 positions:  {traj[-5:]}")

        # Count moves vs waits
        moves = sum(1 for k in range(len(traj)-1) if traj[k] != traj[k+1])
        waits = len(traj) - 1 - moves
        print(f"  Moves: {moves}, Waits: {waits}")

    # Save M to file
    output_file = 'debug_first_M.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump({'M': M, 'obstacles': obstacles, 'config': {
            'num_agents': grid_config.num_agents,
            'size': grid_config.size,
            'density': grid_config.density,
            'seed': grid_config.seed,
        }}, f)
    print(f"\nSaved M to {output_file}")

    return M, obstacles


if __name__ == '__main__':
    M, obstacles = run_first_testcase()
