"""
Trajectory Collector Wrapper for Global Optimization Integration.

This module provides a Gymnasium wrapper that records agent positions at each timestep
during episode execution. The collected trajectories can then be used for post-processing
optimization via the Collapsed-MAPF ILP solver.

Usage:
    env = pogema_v0(grid_config=config)
    env = TrajectoryCollectorWrapper(env)

    obs, info = env.reset()
    while not done:
        obs, rew, terminated, truncated, infos = env.step(actions)

    # After episode ends, trajectories are available in infos[0]['trajectories']
    # or via env.get_trajectories()
"""

from gymnasium import Wrapper
from typing import List, Tuple, Any, Optional


class TrajectoryCollectorWrapper(Wrapper):
    """
    A Gymnasium wrapper that records agent positions at each timestep.

    This wrapper collects trajectory data in the format required by the
    Collapsed-MAPF ILP optimizer (opt_main.py):
    - M[i][k] = position of agent i at time k
    - Positions are stored as tuples (x, y)

    The trajectories are attached to the info dict when the episode ends,
    and can also be accessed via get_trajectories() at any time.

    Attributes:
        trajectories: List of trajectories, one per agent. Each trajectory
                     is a list of (x, y) position tuples.
        _episode_active: Flag indicating if an episode is currently running.
    """

    def __init__(self, env):
        """
        Initialize the TrajectoryCollectorWrapper.

        Args:
            env: The environment to wrap. Must have get_agents_xy() and
                 get_num_agents() methods.
        """
        super().__init__(env)
        self.trajectories: Optional[List[List[Tuple[int, int]]]] = None
        self._episode_active: bool = False

    def reset(self, **kwargs) -> Tuple[Any, dict]:
        """
        Reset the environment and initialize trajectory collection.

        Args:
            **kwargs: Arguments passed to the underlying environment's reset.

        Returns:
            Tuple of (observations, info) from the underlying environment.
        """
        obs, info = self.env.reset(**kwargs)

        # Get number of agents
        num_agents = self.get_num_agents()

        # Initialize trajectory storage
        self.trajectories = [[] for _ in range(num_agents)]
        self._episode_active = True

        # Record initial positions
        positions = self._get_agent_positions()
        for i, pos in enumerate(positions):
            self.trajectories[i].append(tuple(pos))

        return obs, info

    def step(self, actions) -> Tuple[Any, Any, list, list, list]:
        """
        Execute one step and record agent positions.

        Args:
            actions: Actions for all agents.

        Returns:
            Tuple of (observations, rewards, terminated, truncated, infos)
            from the underlying environment. When the episode ends,
            infos[0]['trajectories'] contains the collected trajectories.
        """
        obs, rew, terminated, truncated, infos = self.env.step(actions)

        # Record positions after step
        if self._episode_active:
            positions = self._get_agent_positions()
            for i, pos in enumerate(positions):
                self.trajectories[i].append(tuple(pos))

        # On episode end, attach trajectories to info
        if all(terminated) or all(truncated):
            self._episode_active = False
            if infos and len(infos) > 0:
                infos[0]['trajectories'] = self.trajectories

        return obs, rew, terminated, truncated, infos

    def get_trajectories(self) -> Optional[List[List[Tuple[int, int]]]]:
        """
        Get the collected trajectories.

        Returns:
            List of trajectories M where M[i][k] = (x, y) position of agent i
            at timestep k. Returns None if no episode has been run.
        """
        return self.trajectories

    def get_num_agents(self) -> int:
        """
        Get the number of agents in the environment.

        Returns:
            Number of agents.
        """
        # Try different methods to get num_agents
        if hasattr(self.env, 'get_num_agents'):
            return self.env.get_num_agents()
        elif hasattr(self.env, 'grid_config') and hasattr(self.env.grid_config, 'num_agents'):
            return self.env.grid_config.num_agents
        else:
            # Fallback: try to get from unwrapped env
            unwrapped = self.env.unwrapped
            if hasattr(unwrapped, 'get_num_agents'):
                return unwrapped.get_num_agents()
            elif hasattr(unwrapped, 'grid_config'):
                return unwrapped.grid_config.num_agents
            raise AttributeError("Cannot determine number of agents from environment")

    def _get_agent_positions(self) -> List[Tuple[int, int]]:
        """
        Get current positions of all agents.

        Returns:
            List of (x, y) positions for each agent.
        """
        # Try different methods to get agent positions
        if hasattr(self.env, 'get_agents_xy'):
            return self.env.get_agents_xy()
        else:
            # Try to get from unwrapped env
            unwrapped = self.env.unwrapped
            if hasattr(unwrapped, 'get_agents_xy'):
                return unwrapped.get_agents_xy()
            raise AttributeError("Cannot get agent positions from environment")

    def get_obstacles(self, ignore_borders: bool = False):
        """
        Get the obstacle map from the environment.

        Args:
            ignore_borders: If True, exclude artificial borders from the map.

        Returns:
            2D numpy array where 1 = obstacle, 0 = free.
        """
        if hasattr(self.env, 'get_obstacles'):
            return self.env.get_obstacles(ignore_borders=ignore_borders)
        else:
            unwrapped = self.env.unwrapped
            if hasattr(unwrapped, 'get_obstacles'):
                return unwrapped.get_obstacles(ignore_borders=ignore_borders)
            raise AttributeError("Cannot get obstacles from environment")
