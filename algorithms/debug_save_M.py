"""
Debug script to run a single episode and save M for analysis.
Simplified version that directly uses the eval.py setup.
"""
import sys
import pickle
import yaml
from pathlib import Path
from functools import partial

sys.path.insert(0, '/home/yimintan/research/WAFR2026')

from pogema_toolbox.registry import ToolboxRegistry
from pogema_toolbox.results_holder import ResultsHolder
from create_env import create_env_base
from trajectory_collector import TrajectoryCollectorWrapper
from follower.follower_python.inference import FollowerInference, FollowerInferenceConfig
from follower.follower_python.preprocessing import follower_preprocessor
from pogema_toolbox.create_env import Environment


def run_and_save_M():
    # Load config
    config_path = Path('experiments/01-random/01-random-mapf.yaml')
    with open(config_path) as f:
        evaluation_config = yaml.safe_load(f)

    maps_path = Path('experiments/01-random/maps.yaml')
    with open(maps_path, 'r') as f:
        maps = yaml.safe_load(f)
    ToolboxRegistry.register_maps(maps)

    # Register environment
    ToolboxRegistry.register_env('Environment', create_env_base, Environment)

    # Register algorithm
    ToolboxRegistry.register_algorithm('Follower', FollowerInference, FollowerInferenceConfig, follower_preprocessor)

    # Get environment config
    env_cfg = evaluation_config.get('environment', {})

    # Create environment with trajectory collection
    env_cfg_copy = dict(env_cfg)
    env_cfg_copy['map_name'] = 'validation-random-seed-000'
    env_cfg_copy['num_agents'] = 64

    print(f"Creating environment with N={env_cfg_copy['num_agents']}, map={env_cfg_copy['map_name']}")

    env = create_env_base(env_cfg_copy)
    env = TrajectoryCollectorWrapper(env)

    # Create algorithm config
    algo_cfg = FollowerInferenceConfig()

    # Wrap environment with preprocessor
    env = follower_preprocessor(env, algo_cfg)

    # Create algorithm
    algo = FollowerInference(algo_cfg)

    # Run episode
    algo.reset_states()
    obs, _ = env.reset(seed=env.grid_config.seed)

    step = 0
    while True:
        obs, rew, terminated, truncated, infos = env.step(algo.act(obs))
        step += 1
        if step % 20 == 0:
            print(f"Step {step}...")
        if all(terminated) or all(truncated):
            break

    print(f"Episode finished after {step} steps")

    # Get M from infos - need to access the underlying TrajectoryCollectorWrapper
    # The infos come from the outermost wrapper, so we need to find trajectories
    M = None
    if infos and len(infos) > 0 and 'trajectories' in infos[0]:
        M = infos[0]['trajectories']
    else:
        # Try to get from the wrapped env
        current_env = env
        while hasattr(current_env, 'env'):
            if isinstance(current_env, TrajectoryCollectorWrapper):
                M = current_env.trajectories
                break
            current_env = current_env.env

    if M is not None:
        N = len(M)
        T = len(M[0]) - 1 if M else 0
        print(f"M size: N={N}, T={T}")

        # Save M
        output_file = '/home/yimintan/research/WAFR2026/debug_M.pkl'
        with open(output_file, 'wb') as f:
            pickle.dump({'M': M, 'N': N, 'T': T}, f)
        print(f"Saved M to {output_file}")

        # Also run ILP to see action count
        from opt_main import build_ilp_artifacts
        art = build_ilp_artifacts(M)
        print(f"Number of actions: {len(art.actions)}")

        return M
    else:
        print("No trajectories found!")
        return None


if __name__ == '__main__':
    run_and_save_M()
