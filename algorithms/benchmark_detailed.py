#!/usr/bin/env python
"""
Detailed benchmark to identify bottlenecks.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np
import torch

def benchmark():
    from pogema import pogema_v0, GridConfig
    from foundation_mapf_wrapper.inference import FoundationMAPFInference, FoundationMAPFInferenceConfig

    print("=" * 70)
    print("Detailed Performance Breakdown")
    print("=" * 70)

    cfg = FoundationMAPFInferenceConfig(
        model_path="foundation_mapf/model_checkpoint_epoch_4.pth",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    algo = FoundationMAPFInference(cfg)

    config = {'num_agents': 8, 'size': 32, 'density': 0.2, 'seed': 42, 'steps': 10}

    grid_config = GridConfig(
        num_agents=config['num_agents'],
        size=config['size'],
        density=config['density'],
        seed=config['seed'],
        max_episode_steps=config['steps'],
        observation_type='MAPF',
        on_target='nothing',
    )

    print(f"\nConfig: agents={config['num_agents']}, size={config['size']}")

    for use_cpp in [True, False]:
        algo.use_cpp = use_cpp
        label = "C++" if use_cpp else "Python"
        print(f"\n[{label}]")

        env = pogema_v0(grid_config=grid_config)
        obs, _ = env.reset(seed=config['seed'])
        obs[0]['after_reset'] = True
        algo.reset_states()

        # First act (includes distance map init)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()

        # Step 1: Map processing
        num_agents = len(obs)
        pogema_map = np.array(obs[0]['global_obstacles'])
        algo.map_array = np.ascontiguousarray(pogema_map.T, dtype=np.float32)
        algo.map_tensor = torch.from_numpy(algo.map_array).to(algo.device)
        t1 = time.perf_counter()

        # Step 2: Distance map init
        if use_cpp:
            from foundation_mapf_wrapper import mapf_features_cpp
            algo.distance_map = mapf_features_cpp.DistanceMap()
            algo.distance_map.compute(algo.map_array)
        else:
            from foundation_mapf_wrapper.inference import LazyDistanceMap
            algo.distance_map = LazyDistanceMap(algo.map_array)
        t2 = time.perf_counter()

        # Step 3: Prepare positions
        agent_positions = []
        agent_goals = []
        for o in obs:
            row, col = o['global_xy']
            target_row, target_col = o['global_target_xy']
            agent_positions.append([col, row])
            agent_goals.append([target_col, target_row])
        agent_positions_np = np.array(agent_positions, dtype=np.int64)
        agent_goals_np = np.array(agent_goals, dtype=np.int64)
        t3 = time.perf_counter()

        # Step 4: Feature construction (this triggers BFS for lazy)
        if use_cpp:
            algo._seed_counter += 1
            features_np = mapf_features_cpp.construct_features(
                algo.map_array, agent_positions_np, agent_goals_np,
                algo.distance_map, algo.cfg.feature_dim, algo.cfg.feature_type,
                algo._seed_counter
            )
            feature = torch.from_numpy(features_np).to(algo.device)
        else:
            algo._agent_positions_t = torch.tensor(agent_positions, dtype=torch.long, device=algo.device)
            algo._agent_goals_t = torch.tensor(agent_goals, dtype=torch.long, device=algo.device)
            feature = algo._construct_input_feature_python(
                algo.map_tensor, algo._agent_positions_t, algo._agent_goals_t
            )
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t4 = time.perf_counter()

        # Step 5: Model inference
        with torch.no_grad():
            logits, _ = algo.model(feature.unsqueeze(0))
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t5 = time.perf_counter()

        print(f"  Map processing:      {(t1-t0)*1000:6.2f} ms")
        print(f"  Distance map init:   {(t2-t1)*1000:6.2f} ms")
        print(f"  Position prep:       {(t3-t2)*1000:6.2f} ms")
        print(f"  Feature construction:{(t4-t3)*1000:6.2f} ms")
        print(f"  Model inference:     {(t5-t4)*1000:6.2f} ms")
        print(f"  Total:               {(t5-t0)*1000:6.2f} ms")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    benchmark()
