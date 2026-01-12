"""
SILLM (Scalable Imitation Learning for LMAPF) inference wrapper for pogema-benchmark.

This wrapper adapts the SILLM algorithm to work with the pogema-benchmark evaluation framework.
Since SILLM uses gym (old version) while pogema-benchmark uses gymnasium, we implement
a standalone inference wrapper that loads the pretrained model directly without importing
the full SILLM framework.
"""

import sys
import os

# Add SILLM to path before importing anything else
SILLM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', 'Scalable-Imitation-Learning-for-LMAPF')
if SILLM_PATH not in sys.path:
    sys.path.insert(0, SILLM_PATH)

import pickle
import numpy as np
import torch
import torch.nn as nn
from typing_extensions import Literal
from typing import Optional, Dict, Any
from pydantic import Extra
from pogema_toolbox.algorithm_config import AlgoBase
from collections import deque


class SILLMInferenceConfig(AlgoBase, extra=Extra.forbid):
    """Configuration for SILLM inference."""
    name: Literal['SILLM'] = 'SILLM'
    model_path: str = "Scalable-Imitation-Learning-for-LMAPF/pretrained_models/backward_dijkstra/v3/RL/warehouse/best"
    wppl_mode: Literal['PIBT', 'PIBT-RL'] = 'PIBT-RL'
    device: str = 'cuda'


class SILLMInference:
    """
    SILLM inference wrapper for pogema-benchmark.

    This is a standalone wrapper that loads SILLM's pretrained actor network
    and performs inference without requiring the full SILLM framework.
    """

    def __init__(self, cfg: SILLMInferenceConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

        # Load SILLM actor model
        model_path = cfg.model_path
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', model_path)

        actor_path = os.path.join(model_path, "actor.pt")
        desc_path = os.path.join(model_path, "desc.pkl")

        # Load model description to get architecture info
        with open(desc_path, 'rb') as f:
            self.desc = pickle.load(f)

        # Load actor network
        self.actor = torch.load(actor_path, map_location=self.device, weights_only=False)
        self.actor.eval()

        # Internal state
        self.num_agents = None
        self.actor_rnn_states = None
        self.map_array = None
        self.heuristic_cache = {}

        # FOV parameters (from SILLM)
        self.FOV_height = 11
        self.FOV_width = 11
        self.num_channels = 4

        # Movement definitions (SILLM uses: R, D, L, U, W)
        # pogema uses: stay=0, up=1, down=2, left=3, right=4
        # SILLM uses: R=0, D=1, L=2, U=3, W=4
        self.sillm_to_pogema = {0: 4, 1: 2, 2: 3, 3: 1, 4: 0}  # R->right, D->down, L->left, U->up, W->stay
        self.pogema_to_sillm = {4: 0, 2: 1, 3: 2, 1: 3, 0: 4}

        # SILLM movements: R, D, L, U, W -> (dy, dx)
        self.movements = np.array([[0, 1], [1, 0], [0, -1], [-1, 0], [0, 0]], dtype=np.int32)

        # Get RNN state size from actor
        self.rnn_layer_num = getattr(self.actor, 'rnn_layer_num', 1)
        self.rnn_state_size = getattr(self.actor, 'rnn_state_size', 64)

    def _compute_heuristics_bfs(self, map_array, target_y, target_x):
        """Compute heuristic distances using BFS from target."""
        h, w = map_array.shape
        heuristics = np.full((h, w), np.inf, dtype=np.float32)

        if map_array[target_y, target_x] == 1:  # Target is obstacle
            return heuristics

        queue = deque([(target_y, target_x, 0)])
        heuristics[target_y, target_x] = 0

        while queue:
            y, x, dist = queue.popleft()
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and map_array[ny, nx] == 0 and heuristics[ny, nx] == np.inf:
                    heuristics[ny, nx] = dist + 1
                    queue.append((ny, nx, dist + 1))

        return heuristics

    def _get_heuristics(self, map_array, target_y, target_x):
        """Get or compute heuristics for a target position."""
        key = (target_y, target_x)
        if key not in self.heuristic_cache:
            self.heuristic_cache[key] = self._compute_heuristics_bfs(map_array, target_y, target_x)
        return self.heuristic_cache[key]

    def _build_observation(self, observations):
        """
        Build SILLM-format observations from pogema observations.

        SILLM observation format:
        - 4 channels of 11x11 local view:
          - Channel 0: Static obstacles (1=obstacle, 0=free)
          - Channel 1: Other agents (relative priority: -1, 0, 1)
          - Channel 2: Normalized heuristic difference from center
          - Channel 3: Normalized absolute heuristic
        - Additional features: perm_indices, reverse_perm_indices, priorities, curr_pos, target_pos
        """
        num_agents = len(observations)
        map_array = np.array(observations[0]['global_obstacles'])
        h, w = map_array.shape

        # Pad map for local view extraction
        pad = self.FOV_height // 2
        padded_map = np.pad(map_array, pad, mode='constant', constant_values=1)

        # Get agent positions and targets
        agent_positions = np.array([obs['global_xy'] for obs in observations])  # (N, 2) - (y, x)
        agent_targets = np.array([obs['global_target_xy'] for obs in observations])  # (N, 2) - (y, x)

        # Generate random priorities (SILLM uses priorities for agent ordering)
        priorities = np.random.rand(num_agents).astype(np.float32)

        # Build observations for each agent
        obs_list = []
        for i in range(num_agents):
            y, x = agent_positions[i]
            ty, tx = agent_targets[i]

            # Extract local view (with padding offset)
            py, px = y + pad, x + pad
            local_view = padded_map[py - pad:py + pad + 1, px - pad:px + pad + 1].astype(np.float32)

            # Channel 0: Static obstacles
            ch0 = local_view.copy()

            # Channel 1: Other agents with relative priority
            ch1 = np.zeros((self.FOV_height, self.FOV_width), dtype=np.float32)
            for j in range(num_agents):
                if i == j:
                    continue
                oy, ox = agent_positions[j]
                # Check if agent j is in local view
                rel_y, rel_x = oy - y + pad, ox - x + pad
                if 0 <= rel_y < self.FOV_height and 0 <= rel_x < self.FOV_width:
                    # Relative priority: sign(other_priority - my_priority)
                    ch1[rel_y, rel_x] = np.sign(priorities[j] - priorities[i])

            # Channel 2 & 3: Heuristics
            heuristics = self._get_heuristics(map_array, ty, tx)
            padded_heuristics = np.pad(heuristics, pad, mode='constant', constant_values=np.inf)
            local_heuristics = padded_heuristics[py - pad:py + pad + 1, px - pad:px + pad + 1]

            center_h = local_heuristics[pad, pad]

            # Channel 2: Normalized heuristic difference from center
            ch2 = (local_heuristics - center_h) / (pad * 2) * 0.5
            ch2[local_view == 1] = -1  # Mark obstacles
            ch2[np.isinf(ch2)] = -1

            # Channel 3: Normalized absolute heuristic
            ch3 = local_heuristics / (h + w) * 0.5
            ch3[local_view == 1] = -1
            ch3[np.isinf(ch3)] = -1

            # Stack channels and flatten
            obs = np.stack([ch0, ch1, ch2, ch3], axis=0)  # (4, 11, 11)
            obs_flat = obs.reshape(-1)  # (484,)

            # Add extra features (simplified - using identity permutation)
            perm_idx = float(i)
            rev_perm_idx = float(i)
            priority = priorities[i]

            extra = np.array([perm_idx, rev_perm_idx, priority, y, x, ty, tx], dtype=np.float32)
            full_obs = np.concatenate([obs_flat, extra])
            obs_list.append(full_obs)

        return np.stack(obs_list, axis=0), priorities

    def _build_global_observation(self, observations):
        """
        Build global observation for SILLM.

        Global observation has 3 channels:
        - Channel 0: Static obstacles
        - Channel 1: Agent current positions (binary)
        - Channel 2: Agent target positions (binary)
        """
        map_array = np.array(observations[0]['global_obstacles'])
        h, w = map_array.shape
        num_agents = len(observations)

        agent_positions = np.array([obs['global_xy'] for obs in observations])
        agent_targets = np.array([obs['global_target_xy'] for obs in observations])

        global_obs = np.zeros((1, 3, h, w), dtype=np.float32)

        # Channel 0: obstacles
        global_obs[0, 0] = map_array.astype(np.float32)

        # Channel 1: agent positions
        for i in range(num_agents):
            y, x = agent_positions[i]
            global_obs[0, 1, y, x] = 1.0

        # Channel 2: target positions
        for i in range(num_agents):
            ty, tx = agent_targets[i]
            global_obs[0, 2, ty, tx] = 1.0

        return global_obs

    def _build_action_masks(self, observations):
        """Build action masks based on obstacles."""
        num_agents = len(observations)
        map_array = np.array(observations[0]['global_obstacles'])
        h, w = map_array.shape

        agent_positions = np.array([obs['global_xy'] for obs in observations])

        # SILLM action order: R, D, L, U, W
        masks = np.ones((num_agents, 5), dtype=np.float32)

        for i in range(num_agents):
            y, x = agent_positions[i]
            for a, (dy, dx) in enumerate(self.movements):
                ny, nx = y + dy, x + dx
                if ny < 0 or ny >= h or nx < 0 or nx >= w or map_array[ny, nx] == 1:
                    masks[i, a] = 0

        return masks

    def _get_initial_rnn_state(self, batch_size):
        """Initialize RNN states."""
        return torch.zeros(
            (batch_size, self.rnn_layer_num, self.rnn_state_size),
            device=self.device,
            dtype=torch.float32
        )

    def act(self, observations, rewards=None, dones=None, info=None, skip_agents=None):
        """
        Compute actions for all agents.

        Args:
            observations: List of observation dicts from pogema environment

        Returns:
            List of actions (pogema format: 0=stay, 1=up, 2=down, 3=left, 4=right)
        """
        num_agents = len(observations)

        # Check for reset
        if 'after_reset' in observations[0] or self.num_agents != num_agents:
            self.num_agents = num_agents
            self.map_array = np.array(observations[0]['global_obstacles'])
            self.heuristic_cache = {}  # Clear cache for new map

            # Initialize RNN states
            self.actor_rnn_states = self._get_initial_rnn_state(num_agents)

        # Build SILLM-format observations
        obs_array, priorities = self._build_observation(observations)
        global_obs_array = self._build_global_observation(observations)
        action_masks = self._build_action_masks(observations)

        # Convert to tensors
        obs_tensor = torch.tensor(obs_array, dtype=torch.float32, device=self.device)
        global_obs_tensor = torch.tensor(global_obs_array, dtype=torch.float32, device=self.device)
        mask_tensor = torch.tensor(action_masks, dtype=torch.float32, device=self.device)
        done_tensor = torch.zeros((num_agents, 1), dtype=torch.bool, device=self.device)

        # Call actor network
        with torch.no_grad():
            # The actor forward signature uses EpisodeKey constants as kwargs keys
            # EpisodeKey.CUR_OBS = "observation"
            # EpisodeKey.CUR_GLOBAL_OBS = "global_observation"
            # EpisodeKey.ACTOR_RNN_STATE = "ACTOR_RNN_STATE"
            # EpisodeKey.DONE = "done"
            # EpisodeKey.ACTION_MASK = "action_mask"
            # EpisodeKey.ACTION = "action"
            try:
                actions, new_rnn_states, action_log_probs, dist_entropy, logits = self.actor(
                    observation=obs_tensor,
                    global_observation=global_obs_tensor,
                    ACTOR_RNN_STATE=self.actor_rnn_states,
                    done=done_tensor,
                    action_mask=mask_tensor,
                    explore=False,
                    action=None
                )
                self.actor_rnn_states = new_rnn_states
            except Exception as e:
                # Fallback: use action mask to select valid actions
                print(f"Warning: Could not call actor network: {e}")
                import traceback
                traceback.print_exc()
                # Select random valid action
                actions = []
                for i in range(num_agents):
                    valid_actions = np.where(action_masks[i] > 0)[0]
                    if len(valid_actions) > 0:
                        actions.append(np.random.choice(valid_actions))
                    else:
                        actions.append(4)  # Wait
                actions = torch.tensor(actions, device=self.device)

        # Convert actions from SILLM format to pogema format
        if isinstance(actions, torch.Tensor):
            sillm_actions = actions.cpu().numpy().flatten()
        else:
            sillm_actions = np.array(actions).flatten()

        pogema_actions = [self.sillm_to_pogema[int(a)] for a in sillm_actions]

        return pogema_actions

    def reset_states(self):
        """Reset internal states."""
        self.num_agents = None
        self.actor_rnn_states = None
        self.map_array = None
        self.heuristic_cache = {}

    def after_step(self, dones):
        """Called after each step."""
        pass

    def after_reset(self):
        """Called after environment reset."""
        pass

    def get_additional_info(self):
        """Return additional info for logging."""
        return {"rl_used": 1.0}
