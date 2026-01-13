#!/usr/bin/env python
"""
Quick performance benchmark comparing C++ vs Python implementation.
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
    print("Foundation MAPF: C++ vs Python Performance Comparison")
    print("=" * 70)

    cfg = FoundationMAPFInferenceConfig(
        model_path="foundation_mapf/model_checkpoint_epoch_4.pth",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    algo = FoundationMAPFInference(cfg)

    # Small test config
    config = {'num_agents': 8, 'size': 32, 'density': 0.2, 'seed': 42, 'steps': 20}

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

    # Test C++ version
    print("\n[C++ Extension]")
    algo.use_cpp = True
    env = pogema_v0(grid_config=grid_config)
    obs, _ = env.reset(seed=config['seed'])
    obs[0]['after_reset'] = True
    algo.reset_states()

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    _ = algo.act(obs)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    cpp_reset = (time.perf_counter() - t0) * 1000

    cpp_steps = []
    for _ in range(config['steps'] - 1):
        obs, _, term, trunc, _ = env.step(algo.act(obs))
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        _ = algo.act(obs)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        cpp_steps.append((time.perf_counter() - t0) * 1000)
        if all(term) or all(trunc):
            break

    cpp_step_avg = np.mean(cpp_steps)
    print(f"  Reset: {cpp_reset:.2f} ms")
    print(f"  Step:  {cpp_step_avg:.2f} ms (avg of {len(cpp_steps)} steps)")

    # Test Python version
    print("\n[Python Only]")
    algo.use_cpp = False
    env = pogema_v0(grid_config=grid_config)
    obs, _ = env.reset(seed=config['seed'])
    obs[0]['after_reset'] = True
    algo.reset_states()

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    _ = algo.act(obs)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    py_reset = (time.perf_counter() - t0) * 1000

    py_steps = []
    for _ in range(config['steps'] - 1):
        obs, _, term, trunc, _ = env.step(algo.act(obs))
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        _ = algo.act(obs)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        py_steps.append((time.perf_counter() - t0) * 1000)
        if all(term) or all(trunc):
            break

    py_step_avg = np.mean(py_steps)
    print(f"  Reset: {py_reset:.2f} ms")
    print(f"  Step:  {py_step_avg:.2f} ms (avg of {len(py_steps)} steps)")

    # Comparison
    print("\n" + "=" * 70)
    print("Speedup (Python / C++)")
    print("=" * 70)
    print(f"  Reset: {py_reset/cpp_reset:.2f}x faster with C++")
    print(f"  Step:  {py_step_avg/cpp_step_avg:.2f}x faster with C++")
    print("=" * 70)


if __name__ == '__main__':
    benchmark()
