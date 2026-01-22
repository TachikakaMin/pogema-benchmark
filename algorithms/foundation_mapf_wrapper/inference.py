"""
Foundation MAPF (RAILGUN) inference wrapper for pogema-benchmark.

This wrapper adapts the Foundation MAPF algorithm to work with the pogema-benchmark
evaluation framework. Foundation MAPF is a centralized learning-based method that
uses a UNet architecture to generate actions for all agents simultaneously.

IMPORTANT: Coordinate system differences:
- pogema: global_xy returns (row, col) = (y, x)
  - action 1 (up) -> row-1 (y-1)
  - action 2 (down) -> row+1 (y+1)
  - action 3 (left) -> col-1 (x-1)
  - action 4 (right) -> col+1 (x+1)

- foundation_mapf: agent_locations[:, 0] = x (col), agent_locations[:, 1] = y (row)
  - action 1 (up) -> y+1 ([:, 1] += 1)
  - action 2 (down) -> y-1 ([:, 1] -= 1)
  - action 3 (left) -> x-1 ([:, 0] -= 1)
  - action 4 (right) -> x+1 ([:, 0] += 1)

So the coordinate systems are DIFFERENT:
- pogema: (row, col) = (y, x), up=row-1, down=row+1
- foundation_mapf: (x, y) = (col, row), up=y+1, down=y-1

This means:
- pogema up (row-1) = foundation_mapf down (y-1) when y=row
- pogema down (row+1) = foundation_mapf up (y+1) when y=row
"""

import sys
import os

# Add foundation_mapf to path
FOUNDATION_MAPF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     '..', 'foundation_mapf')
if FOUNDATION_MAPF_PATH not in sys.path:
    sys.path.insert(0, FOUNDATION_MAPF_PATH)

import numpy as np
import torch
import random
from typing import Optional, List, Dict, Any
from typing_extensions import Literal
from pydantic import Extra
from pogema_toolbox.algorithm_config import AlgoBase
from collections import deque

# Try to import C++ extension
try:
    from . import mapf_features_cpp
    USE_CPP = True
    print("Using C++ acceleration for feature construction")
except ImportError:
    USE_CPP = False
    print("C++ extension not available, using Python implementation")


class FoundationMAPFInferenceConfig(AlgoBase, extra=Extra.forbid):
    """Configuration for Foundation MAPF inference."""
    name: Literal['FoundationMAPF'] = 'FoundationMAPF'
    model_path: str = "foundation_mapf/model_checkpoint_epoch_4.pth"
    feature_dim: int = 6
    feature_type: str = "gradient"
    action_dim: int = 5
    first_layer_channels: int = 64
    bilinear: bool = False
    action_choice: str = "sample"  # "sample" or "max"
    device: str = 'cuda'


NOT_FOUND_PATH = 2048


class LazyDistanceMap:
    """
    Lazy distance map that only computes BFS from goal points on demand.
    Much faster than precomputing all pairs for dynamic maps.
    """
    def __init__(self, map_data: np.ndarray):
        self.map_data = map_data
        self.height, self.width = map_data.shape
        self._cache = {}  # goal -> distance array

    def _bfs_from_goal(self, gx: int, gy: int) -> np.ndarray:
        """BFS from goal to compute distances to all reachable points."""
        dist = np.full((self.height, self.width), NOT_FOUND_PATH, dtype=np.int32)
        if self.map_data[gx, gy] != 0:
            return dist

        dist[gx, gy] = 0
        queue = deque([(gx, gy)])

        while queue:
            cx, cy = queue.popleft()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.height and 0 <= ny < self.width:
                    if self.map_data[nx, ny] == 0 and dist[nx, ny] == NOT_FOUND_PATH:
                        dist[nx, ny] = dist[cx, cy] + 1
                        queue.append((nx, ny))

        return dist

    def get_distance(self, ax: int, ay: int, gx: int, gy: int) -> int:
        """Get distance from (ax, ay) to goal (gx, gy)."""
        goal_key = (gx, gy)
        if goal_key not in self._cache:
            self._cache[goal_key] = self._bfs_from_goal(gx, gy)

        if 0 <= ax < self.height and 0 <= ay < self.width:
            return int(self._cache[goal_key][ax, ay])
        return NOT_FOUND_PATH

    def get_distance_array(self, gx: int, gy: int) -> np.ndarray:
        """Get full distance array from goal."""
        goal_key = (gx, gy)
        if goal_key not in self._cache:
            self._cache[goal_key] = self._bfs_from_goal(gx, gy)
        return self._cache[goal_key]


class FoundationMAPFInference:
    """
    Foundation MAPF (RAILGUN) inference wrapper for pogema-benchmark.

    This is a centralized learning-based method that uses a UNet architecture
    to generate actions for all agents simultaneously based on the global map state.
    """

    def __init__(self, cfg: FoundationMAPFInferenceConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

        # Import UNet model
        from models.unet import UNet

        # Initialize model
        self.model = UNet(
            n_channels=cfg.feature_dim,
            n_classes=cfg.action_dim,
            first_layer_channels=cfg.first_layer_channels,
            bilinear=cfg.bilinear
        ).to(self.device)

        # Load model weights
        model_path = cfg.model_path
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', model_path)

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # Print model info
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        model_memory = total_params * 4 / (1024**2)
        print(f"Foundation MAPF model loaded: {total_params} parameters ({model_memory:.2f} MB)")

        # Internal state
        self.num_agents = None
        self.map_array = None  # numpy array for C++ extension
        self.map_tensor = None  # GPU tensor for model input (cached)
        self.distance_map = None  # C++ DistanceMap or LazyDistanceMap
        self.temperature = None  # On GPU
        self.use_cpp = USE_CPP

        # Pre-allocated tensors for reuse
        self._agent_positions_t = None
        self._agent_goals_t = None

        # Action mapping tensor (on GPU for fast indexing)
        self._action_mapping = torch.tensor([0, 2, 1, 3, 4], dtype=torch.long, device=self.device)

        # Store current state for centralized processing
        self.agent_locations = None
        self.goal_locations = None
        self.feature = None

        # Random seed counter for C++ extension
        self._seed_counter = 0

        # Padding offsets for small maps
        self._pad_x = 0
        self._pad_y = 0

    def _get_distance(self, agent_loc: tuple, goal_loc: tuple) -> int:
        """Get distance from agent location to goal location."""
        ax, ay = int(agent_loc[0]), int(agent_loc[1])
        gx, gy = int(goal_loc[0]), int(goal_loc[1])
        return self.distance_map.get_distance(ax, ay, gx, gy)

    def _construct_input_feature_python(
        self,
        map_data: torch.Tensor,
        agent_locations: torch.Tensor,
        goal_locations: torch.Tensor,
    ) -> torch.Tensor:
        """
        Construct input features for the UNet model (Python version).
        """
        height, width = map_data.shape
        agent_num = agent_locations.shape[0]
        device = agent_locations.device

        input_features = torch.zeros(
            (self.cfg.feature_dim, height, width), dtype=torch.float32, device=device
        )

        # Channel 0: Map obstacles (use cached tensor)
        input_features[0] = map_data

        # Channel 1 & 2: Agent positions and goals (marked with agent_id starting from 1)
        agent_indices = torch.arange(1, agent_num + 1, dtype=torch.float32, device=device)
        input_features[1, agent_locations[:, 0], agent_locations[:, 1]] = agent_indices
        input_features[2, goal_locations[:, 0], goal_locations[:, 1]] = agent_indices

        if self.cfg.feature_dim >= 4:
            # Channel 3: Distance to goal
            distances = torch.zeros(agent_num, dtype=torch.float32, device=device)
            for i in range(agent_num):
                agent_loc = (int(agent_locations[i, 0].item()), int(agent_locations[i, 1].item()))
                goal_loc = (int(goal_locations[i, 0].item()), int(goal_locations[i, 1].item()))
                distances[i] = self._get_distance(agent_loc, goal_loc)
            input_features[3, agent_locations[:, 0], agent_locations[:, 1]] = distances

        if self.cfg.feature_dim >= 6 and self.cfg.feature_type == "gradient":
            # Channels 4 & 5: Gradient directions
            dx = torch.zeros(agent_num, dtype=torch.float32, device=device)
            dy = torch.zeros(agent_num, dtype=torch.float32, device=device)
            height_int = int(height)
            width_int = int(width)

            for i in range(agent_num):
                ax, ay = int(agent_locations[i, 0].item()), int(agent_locations[i, 1].item())
                gx, gy = int(goal_locations[i, 0].item()), int(goal_locations[i, 1].item())
                current_dist = self._get_distance((ax, ay), (gx, gy))

                # Check all 4 directions
                left_dist = self._get_distance((ax - 1, ay), (gx, gy)) if ax > 0 else NOT_FOUND_PATH
                right_dist = self._get_distance((ax + 1, ay), (gx, gy)) if ax < height_int - 1 else NOT_FOUND_PATH
                up_dist = self._get_distance((ax, ay + 1), (gx, gy)) if ay < width_int - 1 else NOT_FOUND_PATH
                down_dist = self._get_distance((ax, ay - 1), (gx, gy)) if ay > 0 else NOT_FOUND_PATH

                # Check if other agents are blocking
                for j in range(agent_num):
                    if i == j:
                        continue
                    ox, oy = int(agent_locations[j, 0].item()), int(agent_locations[j, 1].item())
                    if ox == ax - 1 and oy == ay:
                        left_dist = NOT_FOUND_PATH
                    if ox == ax + 1 and oy == ay:
                        right_dist = NOT_FOUND_PATH
                    if ox == ax and oy == ay + 1:
                        up_dist = NOT_FOUND_PATH
                    if ox == ax and oy == ay - 1:
                        down_dist = NOT_FOUND_PATH

                delta_left = left_dist - current_dist
                delta_right = right_dist - current_dist
                delta_up = up_dist - current_dist
                delta_down = down_dist - current_dist

                # Compute gradient x (left/right)
                if delta_left > 0 and delta_right > 0:
                    dx[i] = 0
                elif delta_left >= 0 and delta_right < 0:
                    dx[i] = 1
                elif delta_left < 0 and delta_right >= 0:
                    dx[i] = -1
                elif delta_left < 0 and delta_right < 0:
                    dx[i] = random.choice([-1, 1])
                elif delta_left == 0 and delta_right == 0:
                    dx[i] = random.choice([-1, 0, 1])
                elif delta_left == 0 and delta_right > 0:
                    dx[i] = random.choice([0, -1])
                elif delta_left > 0 and delta_right == 0:
                    dx[i] = random.choice([0, 1])
                else:
                    dx[i] = random.choice([-1, 1])

                # Compute gradient y (down/up)
                if delta_down > 0 and delta_up > 0:
                    dy[i] = 0
                elif delta_down >= 0 and delta_up < 0:
                    dy[i] = 1
                elif delta_down < 0 and delta_up >= 0:
                    dy[i] = -1
                elif delta_down < 0 and delta_up < 0:
                    dy[i] = random.choice([-1, 1])
                elif delta_down == 0 and delta_up == 0:
                    dy[i] = random.choice([-1, 0, 1])
                elif delta_down == 0 and delta_up > 0:
                    dy[i] = random.choice([-1, 0])
                elif delta_down > 0 and delta_up == 0:
                    dy[i] = random.choice([0, 1])
                else:
                    dy[i] = random.choice([-1, 1])

            input_features[4, agent_locations[:, 0], agent_locations[:, 1]] = dx
            input_features[5, agent_locations[:, 0], agent_locations[:, 1]] = dy

        return input_features

    def _construct_input_feature_cpp(
        self,
        map_data: np.ndarray,
        agent_locations: np.ndarray,
        goal_locations: np.ndarray,
    ) -> torch.Tensor:
        """
        Construct input features using C++ extension.
        """
        # Increment seed for randomness
        self._seed_counter += 1

        # Call C++ function (map_data is already contiguous float32)
        features_np = mapf_features_cpp.construct_features(
            map_data,
            agent_locations,
            goal_locations,
            self.distance_map,
            self.cfg.feature_dim,
            self.cfg.feature_type,
            self._seed_counter
        )

        # Convert to torch tensor on GPU
        return torch.from_numpy(features_np).to(self.device)

    def _sample_action_vectorized(
        self,
        logits: torch.Tensor,
        current_locations: torch.Tensor,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample actions from model logits (vectorized version).
        Returns actions for each agent directly.
        """
        agent_num = len(current_locations)

        if self.cfg.action_choice == "sample":
            # Extract logits only at agent positions: [agent_num, action_dim]
            agent_logits = logits[0, :, current_locations[:, 0], current_locations[:, 1]].T

            # Apply temperature scaling (vectorized)
            agent_logits = agent_logits / self.temperature.unsqueeze(1)

            # Softmax and sample
            probs = torch.softmax(agent_logits, dim=-1)
            actions = torch.multinomial(probs, num_samples=1).squeeze(1)
        else:  # "max"
            # Extract logits only at agent positions and take argmax
            agent_logits = logits[0, :, current_locations[:, 0], current_locations[:, 1]].T
            actions = agent_logits.argmax(dim=-1)

        return actions

    def _pad_map_to_min_size(self, map_array: np.ndarray, min_size: int = 16) -> tuple:
        """
        Pad map with obstacles if it's smaller than min_size.
        Returns (padded_map, pad_x, pad_y) where pad_x/pad_y are the offsets.
        """
        height, width = map_array.shape
        if height >= min_size and width >= min_size:
            return map_array, 0, 0

        new_height = max(height, min_size)
        new_width = max(width, min_size)

        # Create new map filled with obstacles (1.0)
        padded_map = np.ones((new_height, new_width), dtype=np.float32)

        # Calculate padding offsets (center the original map)
        pad_x = (new_height - height) // 2
        pad_y = (new_width - width) // 2

        # Copy original map to center
        padded_map[pad_x:pad_x+height, pad_y:pad_y+width] = map_array

        return padded_map, pad_x, pad_y

    def act(self, observations, rewards=None, dones=None, info=None, skip_agents=None):
        """
        Compute actions for all agents.
        """
        num_agents = len(observations)

        # Check for reset
        if 'after_reset' in observations[0] or self.num_agents != num_agents:
            self.num_agents = num_agents

            # Get map from pogema and transpose to foundation_mapf format
            pogema_map = np.array(observations[0]['global_obstacles'])
            raw_map = np.ascontiguousarray(pogema_map.T, dtype=np.float32)

            # Pad map if too small for UNet (minimum 16x16)
            self.map_array, self._pad_x, self._pad_y = self._pad_map_to_min_size(raw_map, min_size=16)

            # Cache map tensor on GPU
            self.map_tensor = torch.from_numpy(self.map_array).to(self.device)

            # Create lazy distance map (no precomputation)
            if self.use_cpp:
                self.distance_map = mapf_features_cpp.DistanceMap()
                self.distance_map.compute(self.map_array)
            else:
                self.distance_map = LazyDistanceMap(self.map_array)

            # Pre-allocate temperature on GPU
            self.temperature = torch.ones(num_agents, device=self.device)

            # Pre-allocate position tensors
            self._agent_positions_t = torch.zeros((num_agents, 2), dtype=torch.long, device=self.device)
            self._agent_goals_t = torch.zeros((num_agents, 2), dtype=torch.long, device=self.device)

        # Convert pogema coordinates to foundation_mapf coordinates
        # Add padding offset if map was padded
        agent_positions_list = []
        agent_goals_list = []

        for obs in observations:
            row, col = obs['global_xy']
            target_row, target_col = obs['global_target_xy']
            # Apply padding offset: foundation_mapf uses (x=col, y=row)
            agent_positions_list.append([col + self._pad_x, row + self._pad_y])
            agent_goals_list.append([target_col + self._pad_x, target_row + self._pad_y])

        # Update pre-allocated tensors in-place
        agent_positions_np = np.array(agent_positions_list, dtype=np.int64)
        agent_goals_np = np.array(agent_goals_list, dtype=np.int64)

        # Copy to GPU tensors (reuse allocated memory)
        self._agent_positions_t.copy_(torch.from_numpy(agent_positions_np))
        self._agent_goals_t.copy_(torch.from_numpy(agent_goals_np))

        # Construct input features
        if self.use_cpp:
            self.feature = self._construct_input_feature_cpp(
                self.map_array, agent_positions_np, agent_goals_np
            )
        else:
            self.feature = self._construct_input_feature_python(
                self.map_tensor, self._agent_positions_t, self._agent_goals_t
            )

        # Model inference
        with torch.no_grad():
            logits, _ = self.model(self.feature.unsqueeze(0))

        # Sample actions (vectorized)
        foundation_actions = self._sample_action_vectorized(
            logits, self._agent_positions_t, self.feature
        )

        # Convert foundation_mapf actions to pogema actions using tensor indexing
        pogema_actions = self._action_mapping[foundation_actions]

        # Return as list (required by pogema)
        return pogema_actions.tolist()

    def reset_states(self):
        """Reset internal states."""
        self.num_agents = None
        self.map_array = None
        self.map_tensor = None
        self.distance_map = None
        self.temperature = None
        self.agent_locations = None
        self.goal_locations = None
        self.feature = None
        self._agent_positions_t = None
        self._agent_goals_t = None
        self._pad_x = 0
        self._pad_y = 0

    def after_step(self, dones):
        """Called after each step."""
        pass

    def after_reset(self):
        """Called after environment reset."""
        pass

    def get_additional_info(self):
        """Return additional info for logging."""
        return {}
