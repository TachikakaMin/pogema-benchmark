#!/usr/bin/env python
"""
Fair benchmark with CUDA warmup.
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
    print("Foundation MAPF: C++ vs Python (Fair Comparison)")
    print("=" * 70)

    cfg = FoundationMAPFInferenceConfig(
        model_path="foundation_mapf/model_checkpoint_epoch_4.pth",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    algo = FoundationMAPFInference(cfg)

    config = {'num_agents': 8, 'size': 32, 'density': 0.2, 'seed': 42, 'steps': 30}

    grid_config = GridConfig(
        num_agents=config['num_agents'],
        size=config['size'],
        density=config['density'],
        seed=config['seed'],
        max_episode_steps=config['steps'],
        observation_type='MAPF',
        on_target='nothing',
    )

    print(f"\nConfig: agents={config['num_agents']}, size={config['size']}, steps={config['steps']}")

    # CUDA warmup
    print("\nWarming up CUDA...")
    env = pogema_v0(grid_config=grid_config)
    obs, _ = env.reset(seed=config['seed'])
    obs[0]['after_reset'] = True
    algo.use_cpp = True
    for _ in range(5):
        algo.reset_states()
        _ = algo.act(obs)
    print("Warmup complete.\n")

    for use_cpp in [True, False]:
        algo.use_cpp = use_cpp
        label = "C++" if use_cpp else "Python"

        # Fresh environment
        env = pogema_v0(grid_config=grid_config)
        obs, _ = env.reset(seed=config['seed'])
        obs[0]['after_reset'] = True
        algo.reset_states()

        # Measure first act (includes BFS)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        _ = algo.act(obs)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        first_act_time = (time.perf_counter() - t0) * 1000

        # Measure subsequent steps
        step_times = []
        for _ in range(config['steps'] - 1):
            obs, _, term, trunc, _ = env.step(algo.act(obs))
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.perf_counter()
            _ = algo.act(obs)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            step_times.append((time.perf_counter() - t0) * 1000)
            if all(term) or all(trunc):
                break

        print(f"[{label}]")
        print(f"  First act (incl. BFS): {first_act_time:6.2f} ms")
        print(f"  Subsequent steps:      {np.mean(step_times):6.2f} ± {np.std(step_times):.2f} ms")
        print()

    print("=" * 70)


if __name__ == '__main__':
    benchmark()
