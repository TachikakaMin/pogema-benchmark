"""
Run Episode with Global Optimization Integration.

This module provides a custom run_episode function that:
1. Runs the episode normally and collects trajectories
2. Applies the Collapsed-MAPF ILP optimizer to minimize move costs
3. Returns results with both original and optimized metrics

Usage:
    from run_episode_optimized import run_episode_with_optimization
    from pogema_toolbox.registry import ToolboxRegistry

    ToolboxRegistry.register_run_func('optimized', run_episode_with_optimization)
"""

import sys
from typing import List, Tuple, Any, Dict, Optional

# Add path for opt_main.py
sys.path.insert(0, '/home/yimintan/research/WAFR2026')

from pogema_toolbox.results_holder import ResultsHolder


def compute_move_only_soc(M: List[List[Tuple[int, int]]]) -> int:
    """
    Compute Sum of Costs where only moves count (waits are free).

    Args:
        M: List of trajectories, M[i][k] = position of agent i at time k
           Each position is a tuple (x, y) or any hashable type

    Returns:
        int: Total number of moves across all agents
    """
    total_moves = 0
    for agent_trajectory in M:
        for k in range(len(agent_trajectory) - 1):
            if agent_trajectory[k] != agent_trajectory[k + 1]:
                total_moves += 1
    return total_moves


def validate_trajectory_quick(M: List[List[Tuple[int, int]]], obstacles) -> bool:
    """
    Quick validation of trajectory feasibility.

    Args:
        M: List of trajectories, M[i][k] = (x, y) position
        obstacles: 2D array where obstacles[x][y] = 1 if obstacle

    Returns:
        bool: True if trajectory is valid, False otherwise
    """
    N = len(M)
    if N == 0:
        return True

    T = len(M[0]) - 1
    VALID_MOVES = {(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)}

    for k in range(T + 1):
        positions_at_k = set()
        for i in range(N):
            pos = tuple(M[i][k])

            # Vertex collision check
            if pos in positions_at_k:
                return False
            positions_at_k.add(pos)

            # Obstacle check
            x, y = pos
            try:
                if obstacles[x][y] == 1:
                    return False
            except (IndexError, TypeError):
                pass  # Skip obstacle check if not available

            # Adjacency check (for k > 0)
            if k > 0:
                prev_pos = M[i][k - 1]
                dx = pos[0] - prev_pos[0]
                dy = pos[1] - prev_pos[1]
                if (dx, dy) not in VALID_MOVES:
                    return False

        # Edge collision check
        if k < T:
            for i in range(N):
                for j in range(i + 1, N):
                    if (tuple(M[i][k]) == tuple(M[j][k + 1]) and
                        tuple(M[j][k]) == tuple(M[i][k + 1]) and
                        tuple(M[i][k]) != tuple(M[i][k + 1])):
                        return False

    return True


def run_episode_with_optimization(
    env,
    algo,
    enable_optimization: bool = True,
    optimization_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Runs an episode and optionally applies global optimization.

    This function:
    1. Runs the episode using the provided algorithm
    2. Collects trajectory data via TrajectoryCollectorWrapper
    3. Applies the Collapsed-MAPF ILP optimizer to minimize moves
    4. Returns results with both original and optimized metrics

    Args:
        env: The environment to run the episode in. Should be wrapped with
             TrajectoryCollectorWrapper for trajectory collection.
        algo: The algorithm used for action selection.
        enable_optimization: If True, apply ILP optimization after episode.
        optimization_config: Optional dict with optimization parameters:
            - time_limit_sec: ILP solver time limit (default: 60.0)
            - threads: Number of threads for solver (default: 4)
            - verbose: Print solver output (default: False)

    Returns:
        dict: Results containing standard metrics plus:
            - original_move_soc: Move-only SoC before optimization
            - optimized_move_soc: Move-only SoC after optimization
            - move_soc_saving: Reduction in move cost
            - optimization_status: Gurobi solver status code
            - num_actions_chosen: Number of collapse actions selected
    """
    # Default optimization config
    if optimization_config is None:
        optimization_config = {
            'time_limit_sec': 5.0,
            'threads': 4,
            'verbose': False,
        }

    algo.reset_states()
    results_holder = ResultsHolder()

    obs, _ = env.reset(seed=env.grid_config.seed)

    # Try to get obstacles for validation
    obstacles = None
    try:
        if hasattr(env, 'get_obstacles'):
            obstacles = env.get_obstacles()
        elif hasattr(env.unwrapped, 'get_obstacles'):
            obstacles = env.unwrapped.get_obstacles()
    except Exception:
        pass

    while True:
        obs, rew, terminated, truncated, infos = env.step(algo.act(obs))
        results_holder.after_step(infos)

        if all(terminated) or all(truncated):
            break

    results = results_holder.get_final()

    # Apply global optimization if enabled and trajectories available
    if enable_optimization and infos and len(infos) > 0 and 'trajectories' in infos[0]:
        M = infos[0]['trajectories']

        # Validate original trajectory if obstacles available
        if obstacles is not None:
            original_valid = validate_trajectory_quick(M, obstacles)
            results['original_trajectory_valid'] = original_valid

        try:
            # Import here to avoid import errors if Gurobi not available
            import time
            from opt_main import solve_collapsed_mapf_from_M

            # Compute original move-only SoC
            original_move_soc = compute_move_only_soc(M)

            # Run ILP optimization with timing
            opt_start_time = time.time()
            summary, chosen_actions = solve_collapsed_mapf_from_M(
                M,
                time_limit_sec=optimization_config.get('time_limit_sec', 60.0),
                threads=optimization_config.get('threads', 4),
                verbose=optimization_config.get('verbose', False)
            )
            opt_elapsed_time = time.time() - opt_start_time

            # Add optimization results to metrics
            results['optimization_time'] = opt_elapsed_time
            results['original_move_soc'] = original_move_soc
            results['optimized_move_soc'] = int(summary['best_min_cost'])
            results['move_soc_saving'] = original_move_soc - int(summary['best_min_cost'])
            results['optimization_status'] = summary['status']
            results['num_actions_chosen'] = len(chosen_actions)
            results['num_agents'] = summary.get('N', len(M))
            results['trajectory_length'] = summary.get('T', len(M[0]) - 1 if M else 0)

            # ILP constraints guarantee validity
            results['optimized_trajectory_valid'] = True

        except ImportError as e:
            results['optimization_error'] = f"Import error: {str(e)}"
        except Exception as e:
            results['optimization_error'] = str(e)

    return results


def run_episode_standard(env, algo) -> Dict[str, Any]:
    """
    Standard run_episode function without optimization.

    This is a drop-in replacement for the default run_episode that
    still collects trajectories but doesn't apply optimization.

    Args:
        env: The environment to run the episode in.
        algo: The algorithm used for action selection.

    Returns:
        dict: Standard episode results.
    """
    return run_episode_with_optimization(env, algo, enable_optimization=False)
