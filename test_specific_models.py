#!/usr/bin/env python3
"""
Test specific models and automatically save results to the correct directory.

Usage examples:
    # List all available checkpoints
    python test_specific_models.py --list

    # Test a specific checkpoint by name
    python test_specific_models.py --checkpoint 20260106_132631

    # Test multiple specific checkpoints
    python test_specific_models.py --checkpoint 20260105_170114 20260106_132631

    # Test the latest checkpoint
    python test_specific_models.py --checkpoint latest

    # Test all checkpoints
    python test_specific_models.py --all

    # Test with custom parameters
    python test_specific_models.py --checkpoint 20260106_132631 --num-runs 50 --headless
"""

import numpy as np
import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from run_sim import run_sim_with_model, resolve_checkpoint_path, get_latest_checkpoint_dir


CHECKPOINTS_DIR = Path("checkpoints")
TEST_RESULTS_DIR = Path("test_results")


def get_all_checkpoints() -> List[Path]:
    """Get all checkpoint directories sorted by timestamp."""
    if not CHECKPOINTS_DIR.exists():
        return []

    checkpoint_dirs = [
        d for d in CHECKPOINTS_DIR.iterdir()
        if d.is_dir() and d.name.replace('_', '').isdigit()
    ]
    return sorted(checkpoint_dirs, key=lambda x: x.name)


def list_checkpoints():
    """List all available checkpoints with their status."""
    checkpoints = get_all_checkpoints()

    if not checkpoints:
        print("No checkpoints found in 'checkpoints/' directory.")
        return

    print("\n" + "="*80)
    print("AVAILABLE CHECKPOINTS".center(80))
    print("="*80)
    print(f"{'Run Name':<20} {'Best Model':<12} {'Latest Model':<14} {'Tested':<10}")
    print("-"*80)

    for cp_dir in checkpoints:
        run_name = cp_dir.name
        has_best = (cp_dir / "best_model.pth").exists()
        has_latest = (cp_dir / "latest_model.pth").exists()
        has_results = (TEST_RESULTS_DIR / f"{run_name}_results.json").exists()

        best_status = "Yes" if has_best else "No"
        latest_status = "Yes" if has_latest else "No"
        tested_status = "Yes" if has_results else "No"

        print(f"{run_name:<20} {best_status:<12} {latest_status:<14} {tested_status:<10}")

    print("-"*80)
    print(f"Total: {len(checkpoints)} checkpoints")
    print("="*80 + "\n")


def run_batch_testing(
    checkpoint_path: str,
    num_runs: int,
    headless: bool,
    max_steps: int,
    placement_threshold: float,
    release_threshold: float,
    verbose: bool = True
) -> Dict[str, Any]:
    """Run batch testing on a single checkpoint."""
    if verbose:
        print(f"\nTesting model: {checkpoint_path}")
        print(f"Running {num_runs} simulation episodes (seeds 0-{num_runs-1})")
        print(f"Mode: {'HEADLESS' if headless else 'Viewer ENABLED'} | Max steps: {max_steps}")
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

        if verbose:
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
        'checkpoint': str(checkpoint_path),
        'num_runs': len(results),
        'num_successes': successes,
        'num_failures': failures,
        'success_rate': successes / len(results) if results else 0.0,
        'success_steps': success_steps,
        'failure_steps': failure_steps,
        'failure_obj_dists': failure_obj_dists,
        'failure_ee_dists': failure_ee_dists,
        'per_run_results': results,
        'test_timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }


def print_summary(results: Dict[str, Any], run_name: str):
    """Print summary statistics from batch testing."""
    print("\n" + "="*70)
    print(f"RESULTS FOR {run_name}".center(70))
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
        print(f"  - Average steps:    {success_steps.mean():.1f} +/- {success_steps.std():.1f}")
        print(f"  - Median steps:     {np.median(success_steps):.0f}")
        print(f"  - Min/Max steps:    {success_steps.min()} / {success_steps.max()}")

    if results['failure_obj_dists']:
        failure_obj_dists = np.array(results['failure_obj_dists'])
        failure_ee_dists = np.array(results['failure_ee_dists'])
        print(f"\nFailure statistics:")
        print(f"  - Avg final obj distance:  {failure_obj_dists.mean():.3f}m +/- {failure_obj_dists.std():.3f}m")
        print(f"  - Avg final ee distance:   {failure_ee_dists.mean():.3f}m +/- {failure_ee_dists.std():.3f}m")

    print("="*70 + "\n")


def save_results(results: Dict[str, Any], run_name: str):
    """Save results to JSON and text files in the test_results directory."""
    TEST_RESULTS_DIR.mkdir(exist_ok=True)

    json_path = TEST_RESULTS_DIR / f"{run_name}_results.json"
    txt_path = TEST_RESULTS_DIR / f"{run_name}_results.txt"

    # Prepare JSON-serializable results
    results_copy = results.copy()
    results_copy['success_steps'] = [int(x) for x in results['success_steps']]
    results_copy['failure_steps'] = [int(x) for x in results['failure_steps']]
    results_copy['failure_obj_dists'] = [float(x) for x in results['failure_obj_dists']]
    results_copy['failure_ee_dists'] = [float(x) for x in results['failure_ee_dists']]

    # Save JSON
    with open(json_path, 'w') as f:
        json.dump(results_copy, f, indent=2)

    # Save human-readable text summary
    with open(txt_path, 'w') as f:
        f.write(f"Test Results for {run_name}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Checkpoint: {results['checkpoint']}\n")
        f.write(f"Test timestamp: {results.get('test_timestamp', 'N/A')}\n\n")
        f.write(f"Total runs: {results['num_runs']}\n")
        f.write(f"Successes: {results['num_successes']} ({results['success_rate']*100:.1f}%)\n")
        f.write(f"Failures: {results['num_failures']} ({(1-results['success_rate'])*100:.1f}%)\n\n")

        if results['success_steps']:
            success_steps = np.array(results['success_steps'])
            f.write("Success Statistics:\n")
            f.write(f"  Average steps: {success_steps.mean():.1f} +/- {success_steps.std():.1f}\n")
            f.write(f"  Median steps: {np.median(success_steps):.0f}\n")
            f.write(f"  Min/Max steps: {success_steps.min()} / {success_steps.max()}\n\n")

        if results['failure_obj_dists']:
            failure_obj_dists = np.array(results['failure_obj_dists'])
            failure_ee_dists = np.array(results['failure_ee_dists'])
            f.write("Failure Statistics:\n")
            f.write(f"  Avg obj distance: {failure_obj_dists.mean():.3f}m +/- {failure_obj_dists.std():.3f}m\n")
            f.write(f"  Avg ee distance: {failure_ee_dists.mean():.3f}m +/- {failure_ee_dists.std():.3f}m\n\n")

        f.write(f"\n{'='*60}\n")
        f.write("Per-run results:\n")
        f.write(f"{'='*60}\n")
        for run in results['per_run_results']:
            status = "SUCCESS" if run['success'] else "FAILURE"
            f.write(f"Run {run['run_id']:3d} (seed={run['seed']:3d}): {status:7s} "
                   f"steps={run['steps']:5d} obj_dist={run['obj_to_target_dist']:.3f}m "
                   f"ee_dist={run['ee_to_obj_dist']:.3f}m\n")

    print(f"Results saved to:")
    print(f"  - {json_path}")
    print(f"  - {txt_path}")


def print_comparison_summary(all_results: Dict[str, Dict[str, Any]]):
    """Print a comparison summary of all tested models."""
    if len(all_results) <= 1:
        return

    print("\n" + "="*80)
    print("COMPARISON SUMMARY".center(80))
    print("="*80)
    print(f"{'Run Name':<20} {'Success Rate':<15} {'Avg Steps':<15} {'Successes':<12}")
    print("-"*80)

    # Sort by success rate (descending)
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['success_rate'], reverse=True)

    for run_name, results in sorted_results:
        success_rate = f"{results['success_rate']*100:.1f}%"

        if results['success_steps']:
            avg_steps = f"{np.mean(results['success_steps']):.0f}"
        else:
            avg_steps = "N/A"

        successes = f"{results['num_successes']}/{results['num_runs']}"

        print(f"{run_name:<20} {success_rate:<15} {avg_steps:<15} {successes:<12}")

    print("-"*80)

    # Find best model
    best_run = sorted_results[0][0]
    best_rate = sorted_results[0][1]['success_rate'] * 100
    print(f"Best model: {best_run} ({best_rate:.1f}% success rate)")
    print("="*80 + "\n")


def extract_run_name(checkpoint_path: str) -> str:
    """Extract run name from checkpoint path."""
    path = Path(checkpoint_path)
    # Handle paths like checkpoints/20260106_132631/best_model.pth
    if path.parent.name != "checkpoints":
        return path.parent.name
    return path.stem


def test_checkpoints(
    checkpoint_names: List[str],
    num_runs: int,
    headless: bool,
    max_steps: int,
    placement_threshold: float,
    release_threshold: float,
    skip_existing: bool = False
) -> Dict[str, Dict[str, Any]]:
    """Test multiple checkpoints and return all results."""
    all_results = {}

    for cp_name in checkpoint_names:
        # Resolve checkpoint path
        try:
            checkpoint_path = resolve_checkpoint_path(cp_name)
        except Exception as e:
            print(f"Error resolving checkpoint '{cp_name}': {e}")
            continue

        if checkpoint_path is None:
            print(f"Checkpoint not found: {cp_name}")
            continue

        run_name = extract_run_name(str(checkpoint_path))

        # Check if results already exist
        if skip_existing and (TEST_RESULTS_DIR / f"{run_name}_results.json").exists():
            print(f"Skipping {run_name} (results already exist, use --force to override)")
            continue

        print(f"\n{'#'*80}")
        print(f"# Testing: {run_name}")
        print(f"{'#'*80}")

        # Run tests
        results = run_batch_testing(
            checkpoint_path=str(checkpoint_path),
            num_runs=num_runs,
            headless=headless,
            max_steps=max_steps,
            placement_threshold=placement_threshold,
            release_threshold=release_threshold
        )

        # Print summary
        print_summary(results, run_name)

        # Save results
        save_results(results, run_name)

        all_results[run_name] = results

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Test specific models and save results to the correct directory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_specific_models.py --list                           # List all checkpoints
  python test_specific_models.py --checkpoint 20260106_132631     # Test specific model
  python test_specific_models.py --checkpoint 20260105_170114 20260106_132631  # Test multiple
  python test_specific_models.py --checkpoint latest              # Test latest model
  python test_specific_models.py --all                            # Test all models
  python test_specific_models.py --all --skip-existing            # Test only untested models
        """
    )

    # Checkpoint selection
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--list', action='store_true',
                       help='List all available checkpoints')
    group.add_argument('--checkpoint', '-c', nargs='+', type=str,
                       help='Checkpoint name(s) to test (e.g., 20260106_132631, or "latest")')
    group.add_argument('--all', action='store_true',
                       help='Test all available checkpoints')

    # Testing parameters
    parser.add_argument('--num-runs', '-n', type=int, default=100,
                       help='Number of test runs per checkpoint (default: 100)')
    parser.add_argument('--headless', action='store_true', default=True,
                       help='Run in headless mode without visualization (default: True)')
    parser.add_argument('--viewer', dest='headless', action='store_false',
                       help='Enable viewer for visualization')
    parser.add_argument('--max-steps', type=int, default=20000,
                       help='Maximum simulation steps (default: 20000)')
    parser.add_argument('--placement-threshold', type=float, default=0.1,
                       help='Object placement threshold in meters (default: 0.1)')
    parser.add_argument('--release-threshold', type=float, default=0.1,
                       help='EE release threshold in meters (default: 0.1)')

    # Additional options
    parser.add_argument('--skip-existing', action='store_true',
                       help='Skip checkpoints that already have results')
    parser.add_argument('--force', action='store_true',
                       help='Force re-testing even if results exist (opposite of --skip-existing)')

    args = parser.parse_args()

    # Handle --list
    if args.list:
        list_checkpoints()
        return 0

    # Determine which checkpoints to test
    if args.all:
        checkpoints = get_all_checkpoints()
        if not checkpoints:
            print("No checkpoints found in 'checkpoints/' directory.")
            return 1
        checkpoint_names = [cp.name for cp in checkpoints]
    elif args.checkpoint:
        checkpoint_names = args.checkpoint
    else:
        # Default: show help
        parser.print_help()
        return 0

    # Determine skip behavior
    skip_existing = args.skip_existing and not args.force

    # Run tests
    all_results = test_checkpoints(
        checkpoint_names=checkpoint_names,
        num_runs=args.num_runs,
        headless=args.headless,
        max_steps=args.max_steps,
        placement_threshold=args.placement_threshold,
        release_threshold=args.release_threshold,
        skip_existing=skip_existing
    )

    # Print comparison if testing multiple models
    if len(all_results) > 1:
        print_comparison_summary(all_results)

    if not all_results:
        print("No models were tested.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
