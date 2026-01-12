import os
from pathlib import Path
from typing import Literal
import json

os.environ.setdefault("RAY_DISABLE_DASHBOARD", "1")
os.environ.setdefault("RAY_DASHBOARD_ENABLED", "0")

import yaml
from pogema_toolbox.evaluator import evaluation

from pogema_toolbox.registry import ToolboxRegistry

from create_env import create_env_base
from rhcr_cpp.rhcr import RHCRInference, RHCRConfig
from pogema_toolbox.create_env import Environment
from scrimp.inference import SCRIMPInference, SCRIMPInferenceConfig
from follower.follower_python.inference import FollowerInference, FollowerInferenceConfig
from follower.follower_python.preprocessing import follower_preprocessor
# from mamba.inference.utils import MAMBAInference, MAMBAInferenceConfig
from lacam.inference import LacamInference, LacamInferenceConfig
from mats_lp.inference import MATS_LPConfig, MATS_LPInference
from dcc.inference import DCCInference, DCCInferenceConfig
from sillm.inference import SILLMInference, SILLMInferenceConfig
from mamba.inference.inference_config import MAMBAInferenceConfig
from mamba.inference.utils import MAMBAInference
from mamba.preprocessing import mamba_preprocessor

# Global optimization imports
from functools import partial
from trajectory_collector import TrajectoryCollectorWrapper
from run_episode_optimized import run_episode_with_optimization


PROJECT_NAME = 'Benchmark'
BASE_PATH = Path('experiments')
MODE: Literal["mapf", "lmapf"] = 'mapf'

# Global optimization configuration
ENABLE_GLOBAL_OPTIMIZATION = True
OPTIMIZATION_CONFIG = {
    'time_limit_sec': 5.0,
    'threads': 8,
    'verbose': True,
}


def create_env_with_trajectory(config):
    """
    Modified env creation that includes trajectory collection for global optimization.
    Wraps the base environment with TrajectoryCollectorWrapper.
    """
    env = create_env_base(config)
    if ENABLE_GLOBAL_OPTIMIZATION:
        env = TrajectoryCollectorWrapper(env)
    return env


def save_evaluation_results(eval_dir):
    """
    Save aggregated evaluation results to a summary file.
    """
    results_dir = Path(eval_dir)
    all_results = []

    for json_file in results_dir.glob("*.json"):
        with open(json_file, 'r') as f:
            results = json.load(f)
            all_results.extend(results)

    if all_results:
        summary_path = results_dir / "evaluation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        ToolboxRegistry.info(f"Saved evaluation summary to {summary_path}")

def main():
    env_cfg_name = 'Environment'

    # Register environment with trajectory collection if optimization is enabled
    if ENABLE_GLOBAL_OPTIMIZATION:
        ToolboxRegistry.register_env(env_cfg_name, create_env_with_trajectory, Environment)
        # Register the optimized run_episode function with config
        optimized_run_func = partial(
            run_episode_with_optimization,
            optimization_config=OPTIMIZATION_CONFIG
        )
        ToolboxRegistry.register_run_func('optimized', optimized_run_func)
        ToolboxRegistry.info("Global optimization enabled - using trajectory collection and optimized run_episode")
    else:
        ToolboxRegistry.register_env(env_cfg_name, create_env_base, Environment)

    ToolboxRegistry.register_algorithm('RHCR', RHCRInference, RHCRConfig)
    ToolboxRegistry.register_algorithm('SCRIMP', SCRIMPInference, SCRIMPInferenceConfig)
    ToolboxRegistry.register_algorithm('Follower', FollowerInference, FollowerInferenceConfig, follower_preprocessor)
    ToolboxRegistry.register_algorithm('LaCAM', LacamInference, LacamInferenceConfig)
    ToolboxRegistry.register_algorithm('MATS-LP', MATS_LPInference, MATS_LPConfig)
    ToolboxRegistry.register_algorithm('DCC', DCCInference, DCCInferenceConfig)
    ToolboxRegistry.register_algorithm('SILLM', SILLMInference, SILLMInferenceConfig)
    ToolboxRegistry.register_algorithm("MAMBA", MAMBAInference, MAMBAInferenceConfig, mamba_preprocessor)

    folder_names = [
        '01-random',
        '02-mazes',
        '03-warehouse',
        '04-movingai',
        '05-puzzles',
    ]

    # if MODE == "mapf":
    #     folder_names += ['06-pathfinding']

    for folder in folder_names:
        maps_path = BASE_PATH / folder / "maps.yaml"
        with open(maps_path, 'r') as f:
            maps = yaml.safe_load(f)
        ToolboxRegistry.register_maps(maps)

        config_path = BASE_PATH / folder / f"{Path(folder).name}-{MODE}.yaml"
        with open(config_path) as f:
            evaluation_config = yaml.safe_load(f)

        # If optimization is enabled, modify algorithm configs to use optimized run_episode
        if ENABLE_GLOBAL_OPTIMIZATION:
            for algo_cfg in evaluation_config.get('algorithms', {}).values():
                algo_cfg['run_episode_func'] = 'optimized'

        eval_dir = BASE_PATH / folder
        evaluation(evaluation_config, eval_dir=eval_dir)
        save_evaluation_results(eval_dir)


if __name__ == '__main__':
    main()
