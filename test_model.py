#!/usr/bin/env python3
"""
Test script for batch evaluation of trained pick-and-place model.
Runs N simulation episodes with sequential seeding for reproducibility.
"""

import numpy as np
import argparse
import json
from typing import Dict, Any
from run_sim import run_sim_with_model, resolve_checkpoint_path


def run_single_test_rollout(
    checkpoint_path: str,
    seed: int,
    headless: bool = True,
    max_steps: int = 10000,
    placement_threshold: float = 0.05,
    release_threshold: float = 0.1,
    verbose: bool = False
):

    return run_sim_with_model(
        checkpoint_path=checkpoint_path,
        headless=headless,
        max_steps=max_steps,
        actions_per_query=1,
        placement_threshold=placement_threshold,
        release_threshold=release_threshold,
        verbose=verbose
    )


def run_batch_testing(
    checkpoint_path: str,
    num_runs: int,
    headless: bool,
    max_steps: int,
    placement_threshold: float,
    release_threshold: float
) -> Dict[str, Any]:
    print(f"\nTesting model: {checkpoint_path}")
    print(f"Running {num_runs} simulation episodes (seeds 0-{num_runs-1})")
    print(f"Mode: {'Viewer ENABLED' if not headless else 'HEADLESS'} | Max steps: {max_steps}")
    print(f"Success criteria: obj_dist < {placement_threshold}m AND ee_dist > {release_threshold}m\n")

    results = []
    successes = 0
    failures = 0

    success_steps = []
    failure_steps = []
    failure_obj_dists = []
    failure_ee_dists = []

    for i in range(num_runs):
        np.random.seed(i)
        success, steps, obj_dist, ee_dist = run_sim_with_model(
            checkpoint_path=checkpoint_path,
            headless=headless,
            max_steps=max_steps,
            actions_per_query=1,
            placement_threshold=placement_threshold,
            release_threshold=release_threshold,
            verbose=False
        )

        if success:
            status = "SUCCESS"
            successes += 1
            success_steps.append(steps)
        else:
            status = "FAILURE"
            failures += 1
            failure_steps.append(steps)
            failure_obj_dists.append(obj_dist)
            failure_ee_dists.append(ee_dist)

        print(f"Run {i+1:3d}/{num_runs} (seed={i:3d}): {status:7s} "
                f"(steps: {steps:5d}, obj_dist: {obj_dist:.3f}m, ee_dist: {ee_dist:.3f}m)")

        results.append({
            'run_id': i,
            'seed': i,
            'success': success,
            'steps': steps,
            'obj_to_target_dist': obj_dist,
            'ee_to_obj_dist': ee_dist
        })
    return {
        'checkpoint': checkpoint_path,
        'num_runs': len(results),
        'num_successes': successes,
        'num_failures': failures,
        'success_rate': successes / len(results) if results else 0.0,
        'success_steps': success_steps,
        'failure_steps': failure_steps,
        'failure_obj_dists': failure_obj_dists,
        'failure_ee_dists': failure_ee_dists,
        'per_run_results': results
    }


def print_summary(results: Dict[str, Any]):
    """Print summary statistics from batch testing."""
    print("\n" + "="*70)
    print("RESULTS".center(70))
    print("="*70)

    total = results['num_runs']
    successes = results['num_successes']
    failures = results['num_failures']
    success_rate = results['success_rate']

    print(f"Successful runs:      {successes:3d} / {total:3d}  ({success_rate*100:.1f}%)")
    print(f"Failed runs:          {failures:3d} / {total:3d}  ({(1-success_rate)*100:.1f}%)")

    if results['success_steps']:
        success_steps = np.array(results['success_steps'])
        print(f"\nSuccess statistics:")
        print(f"  - Average steps:    {success_steps.mean():.1f} ± {success_steps.std():.1f}")
        print(f"  - Median steps:     {np.median(success_steps):.0f}")
        print(f"  - Min/Max steps:    {success_steps.min()} / {success_steps.max()}")

    if results['failure_obj_dists']:
        failure_obj_dists = np.array(results['failure_obj_dists'])
        failure_ee_dists = np.array(results['failure_ee_dists'])
        print(f"\nFailure statistics:")
        print(f"  - Average final obj distance:  {failure_obj_dists.mean():.3f}m ± {failure_obj_dists.std():.3f}m")
        print(f"  - Average final ee distance:   {failure_ee_dists.mean():.3f}m ± {failure_ee_dists.std():.3f}m")

    print(f"\nSeed range: 0-{total-1}")
    print("="*70 + "\n")


def save_results_json(results: Dict[str, Any], output_path: str):
    """Save results to JSON file."""
    # Convert numpy arrays to lists for JSON serialization
    results_copy = results.copy()
    results_copy['checkpoint'] = str(results['checkpoint'])  # Convert Path to string
    results_copy['success_steps'] = [int(x) for x in results['success_steps']]
    results_copy['failure_steps'] = [int(x) for x in results['failure_steps']]
    results_copy['failure_obj_dists'] = [float(x) for x in results['failure_obj_dists']]
    results_copy['failure_ee_dists'] = [float(x) for x in results['failure_ee_dists']]

    with open(output_path, 'w') as f:
        json.dump(results_copy, f, indent=2)

    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Batch test trained pick-and-place model over N episodes'
    )

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (or "latest", or run name)')
    parser.add_argument('--num-runs', type=int, default=100,
                       help='Number of test runs (default: 100)')
    parser.add_argument('--viewer', dest='headless', action='store_false',
                       help='Enable viewer (default)')
    parser.add_argument('--no-viewer', dest='headless', action='store_true',
                       help='Run headless without visualization')
    parser.add_argument('--max-steps', type=int, default=20000,
                       help='Maximum simulation steps (default: 20000)')
    parser.add_argument('--placement-threshold', type=float, default=0.1,
                       help='Object placement threshold in meters (default: 0.1)')
    parser.add_argument('--release-threshold', type=float, default=0.1,
                       help='EE release threshold in meters (default: 0.1)')
    parser.add_argument('--save-results', type=str,
                       help='Path to save detailed results JSON (optional)')

    parser.set_defaults(headless=False)

    args = parser.parse_args()

    # Resolve checkpoint path
    try:
        checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    except ValueError as e:
        print(f"Error: {e}")
        return 1

    # Run batch testing
    results = run_batch_testing(
        checkpoint_path=checkpoint_path,
        num_runs=args.num_runs,
        headless=args.headless,
        max_steps=args.max_steps,
        placement_threshold=args.placement_threshold,
        release_threshold=args.release_threshold
    )

    # Print summary
    print_summary(results)

    # Save results if requested
    if args.save_results:
        save_results_json(results, args.save_results)

    return 0


if __name__ == "__main__":
    exit(main())
