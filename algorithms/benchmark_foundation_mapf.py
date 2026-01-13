#!/usr/bin/env python
"""
Performance benchmark for Foundation MAPF inference.
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

    print("=" * 60)
    print("Foundation MAPF Performance Benchmark")
    print("=" * 60)

    cfg = FoundationMAPFInferenceConfig(
        model_path="foundation_mapf/model_checkpoint_epoch_4.pth",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    algo = FoundationMAPFInference(cfg)
    print(f"Using C++ extension: {algo.use_cpp}")
    print(f"Device: {algo.device}")
    print()

    # Test configurations
    test_configs = [
        {'num_agents': 8, 'size': 32, 'density': 0.2, 'seed': 42, 'steps': 100},
        {'num_agents': 16, 'size': 64, 'density': 0.2, 'seed': 123, 'steps': 100},
        {'num_agents': 32, 'size': 64, 'density': 0.2, 'seed': 456, 'steps': 100},
        {'num_agents': 64, 'size': 128, 'density': 0.2, 'seed': 789, 'steps': 50},
    ]

    for config in test_configs:
        print(f"Config: agents={config['num_agents']}, size={config['size']}, density={config['density']}")

        grid_config = GridConfig(
            num_agents=config['num_agents'],
            size=config['size'],
            density=config['density'],
            seed=config['seed'],
            max_episode_steps=config['steps'],
            observation_type='MAPF',
            on_target='nothing',
        )

        env = pogema_v0(grid_config=grid_config)
        obs, _ = env.reset(seed=config['seed'])
        obs[0]['after_reset'] = True
        algo.reset_states()

        # Warmup
        _ = algo.act(obs)

        # Benchmark reset (distance map computation)
        algo.reset_states()
        obs[0]['after_reset'] = True

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        reset_start = time.perf_counter()
        _ = algo.act(obs)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        reset_time = time.perf_counter() - reset_start

        # Benchmark steps
        step_times = []
        for step in range(config['steps'] - 1):
            obs, rewards, terminated, truncated, infos = env.step(algo.act(obs))

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            step_start = time.perf_counter()
            actions = algo.act(obs)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            step_times.append(time.perf_counter() - step_start)

            if all(terminated) or all(truncated):
                break

        avg_step_time = np.mean(step_times) * 1000  # ms
        std_step_time = np.std(step_times) * 1000

        print(f"  Reset time (incl. distance map): {reset_time*1000:.2f} ms")
        print(f"  Avg step time: {avg_step_time:.2f} ± {std_step_time:.2f} ms")
        print(f"  Steps completed: {len(step_times) + 1}")
        print(f"  Throughput: {1000/avg_step_time:.1f} steps/sec")
        print()

    print("=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == '__main__':
    benchmark()
