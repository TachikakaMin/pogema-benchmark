#!/usr/bin/env python
"""
Performance benchmark comparing C++ vs Python implementation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import numpy as np
import torch

def benchmark_single(algo, env, config, label):
    obs, _ = env.reset(seed=config['seed'])
    obs[0]['after_reset'] = True
    algo.reset_states()

    # Warmup
    _ = algo.act(obs)

    # Benchmark reset (distance map computation)
    algo.reset_states()
    obs, _ = env.reset(seed=config['seed'])
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

    avg_step_time = np.mean(step_times) * 1000
    return reset_time * 1000, avg_step_time, len(step_times) + 1


def benchmark():
    from pogema import pogema_v0, GridConfig
    from foundation_mapf_wrapper.inference import FoundationMAPFInference, FoundationMAPFInferenceConfig

    print("=" * 70)
    print("Foundation MAPF: C++ vs Python Performance Comparison")
    print("=" * 70)

    # Test configurations
    test_configs = [
        {'num_agents': 8, 'size': 32, 'density': 0.2, 'seed': 42, 'steps': 50},
        {'num_agents': 16, 'size': 64, 'density': 0.2, 'seed': 123, 'steps': 50},
        {'num_agents': 32, 'size': 64, 'density': 0.2, 'seed': 456, 'steps': 50},
    ]

    # C++ version
    print("\n[C++ Extension]")
    cfg_cpp = FoundationMAPFInferenceConfig(
        model_path="foundation_mapf/model_checkpoint_epoch_4.pth",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    algo_cpp = FoundationMAPFInference(cfg_cpp)
    print(f"Using C++ extension: {algo_cpp.use_cpp}")

    cpp_results = []
    for config in test_configs:
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
        reset_t, step_t, steps = benchmark_single(algo_cpp, env, config, "C++")
        cpp_results.append((config, reset_t, step_t, steps))
        print(f"  agents={config['num_agents']:2d}, size={config['size']:3d}: reset={reset_t:7.2f}ms, step={step_t:5.2f}ms")

    # Python version (disable C++)
    print("\n[Python Only]")
    algo_cpp.use_cpp = False
    print(f"Using C++ extension: {algo_cpp.use_cpp}")

    py_results = []
    for config in test_configs:
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
        reset_t, step_t, steps = benchmark_single(algo_cpp, env, config, "Python")
        py_results.append((config, reset_t, step_t, steps))
        print(f"  agents={config['num_agents']:2d}, size={config['size']:3d}: reset={reset_t:7.2f}ms, step={step_t:5.2f}ms")

    # Comparison
    print("\n" + "=" * 70)
    print("Speedup (Python / C++)")
    print("=" * 70)
    print(f"{'Config':<25} {'Reset Speedup':>15} {'Step Speedup':>15}")
    print("-" * 70)
    for (cfg, cpp_reset, cpp_step, _), (_, py_reset, py_step, _) in zip(cpp_results, py_results):
        label = f"agents={cfg['num_agents']}, size={cfg['size']}"
        reset_speedup = py_reset / cpp_reset if cpp_reset > 0 else 0
        step_speedup = py_step / cpp_step if cpp_step > 0 else 0
        print(f"{label:<25} {reset_speedup:>14.2f}x {step_speedup:>14.2f}x")

    print("=" * 70)


if __name__ == '__main__':
    benchmark()
