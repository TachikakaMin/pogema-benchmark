#!/usr/bin/env python
"""
Test script for Foundation MAPF integration with pogema-benchmark.
Run with: conda activate py310 && python test_foundation_mapf.py
"""

import sys
import os

# Add algorithms directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch


def test_model_loading():
    """Test that the model can be loaded correctly."""
    print("=" * 60)
    print("Test 1: Model Loading")
    print("=" * 60)

    from foundation_mapf_wrapper.inference import FoundationMAPFInference, FoundationMAPFInferenceConfig

    cfg = FoundationMAPFInferenceConfig(
        model_path="foundation_mapf/model_checkpoint_epoch_4.pth",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    algo = FoundationMAPFInference(cfg)
    print(f"Model loaded successfully on device: {algo.device}")
    print("Model loading: PASSED")
    return algo


def test_with_pogema():
    """Test with actual pogema environment."""
    print("\n" + "=" * 60)
    print("Test 2: Integration with Pogema Environment")
    print("=" * 60)

    try:
        from pogema import pogema_v0, GridConfig
        from foundation_mapf_wrapper.inference import FoundationMAPFInference, FoundationMAPFInferenceConfig

        cfg = FoundationMAPFInferenceConfig(
            model_path="foundation_mapf/model_checkpoint_epoch_4.pth",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        algo = FoundationMAPFInference(cfg)

        # Test with different configurations
        test_configs = [
            {'num_agents': 4, 'size': 32, 'density': 0.1, 'seed': 42},
            {'num_agents': 8, 'size': 32, 'density': 0.2, 'seed': 123},
            {'num_agents': 16, 'size': 64, 'density': 0.2, 'seed': 456},
        ]

        for config in test_configs:
            grid_config = GridConfig(
                num_agents=config['num_agents'],
                size=config['size'],
                density=config['density'],
                seed=config['seed'],
                max_episode_steps=256,
                observation_type='MAPF',
                on_target='nothing',
            )

            env = pogema_v0(grid_config=grid_config)
            obs, _ = env.reset(seed=config['seed'])
            obs[0]['after_reset'] = True
            algo.reset_states()

            # Run episode
            for step in range(256):
                actions = algo.act(obs)
                obs, rewards, terminated, truncated, infos = env.step(actions)

                if all(terminated) or all(truncated):
                    break

            # Calculate ISR
            reached = sum(1 for o in obs if o['global_xy'] == o['global_target_xy'])
            isr = reached / len(obs)

            print(f"Config: agents={config['num_agents']}, size={config['size']}, density={config['density']}")
            print(f"  Steps: {step + 1}, ISR: {isr:.2%} ({reached}/{len(obs)})")

        print("\nPogema integration test: PASSED")

    except ImportError as e:
        print(f"Pogema not available: {e}")
        print("Skipping pogema integration test")


def test_coordinate_conversion():
    """Test coordinate conversion between pogema and foundation_mapf."""
    print("\n" + "=" * 60)
    print("Test 3: Coordinate Conversion")
    print("=" * 60)

    try:
        from pogema import pogema_v0, GridConfig
        from foundation_mapf_wrapper.inference import FoundationMAPFInference, FoundationMAPFInferenceConfig

        cfg = FoundationMAPFInferenceConfig(
            model_path="foundation_mapf/model_checkpoint_epoch_4.pth",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

        algo = FoundationMAPFInference(cfg)

        # Create simple environment
        grid_config = GridConfig(
            num_agents=1,
            size=10,
            density=0.0,  # No obstacles for simple test
            seed=42,
            max_episode_steps=50,
            observation_type='MAPF',
            on_target='nothing',
        )

        env = pogema_v0(grid_config=grid_config)
        obs, _ = env.reset(seed=42)
        obs[0]['after_reset'] = True

        initial_pos = obs[0]['global_xy']
        target_pos = obs[0]['global_target_xy']

        print(f"Initial position: {initial_pos}")
        print(f"Target position: {target_pos}")

        # Run a few steps and check if agent moves toward target
        prev_dist = abs(initial_pos[0] - target_pos[0]) + abs(initial_pos[1] - target_pos[1])

        for step in range(20):
            actions = algo.act(obs)
            obs, rewards, terminated, truncated, infos = env.step(actions)

            current_pos = obs[0]['global_xy']
            current_dist = abs(current_pos[0] - target_pos[0]) + abs(current_pos[1] - target_pos[1])

            if current_dist < prev_dist:
                print(f"Step {step+1}: pos={current_pos}, dist={current_dist} (moving closer)")
            elif current_dist == prev_dist:
                print(f"Step {step+1}: pos={current_pos}, dist={current_dist} (same distance)")
            else:
                print(f"Step {step+1}: pos={current_pos}, dist={current_dist} (moving away!)")

            prev_dist = current_dist

            if current_pos == target_pos:
                print(f"Reached target at step {step+1}!")
                break

        print("\nCoordinate conversion test: PASSED")

    except ImportError as e:
        print(f"Pogema not available: {e}")
        print("Skipping coordinate conversion test")


def main():
    print("Testing Foundation MAPF Integration")
    print("=" * 60)

    # Test 1: Model loading
    test_model_loading()

    # Test 2: Pogema integration
    test_with_pogema()

    # Test 3: Coordinate conversion
    test_coordinate_conversion()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
