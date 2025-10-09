#!/usr/bin/env python3
"""
Run Coordinate Accuracy Evaluation on Experiment Results

This script evaluates VLM detection accuracy against OCAtari ground truth
for experiments run with the advance_game_runner.py pipeline.

Usage:
    # Evaluate a single experiment
    python run_coordinate_evaluation.py \
        --experiment_dir experiments/Pong_bedrock_seed42 \
        --game pong

    # Evaluate multiple experiments and aggregate
    python run_coordinate_evaluation.py \
        --experiment_dir experiments/ \
        --game pong \
        --aggregate

    # Evaluate specific frames
    python run_coordinate_evaluation.py \
        --experiment_dir experiments/Pong_bedrock_seed42 \
        --game pong \
        --frames 0,10,20,50,100
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from coordinate_accuracy_evaluator import CoordinateAccuracyEvaluator, FrameEvaluationResult
from ocatari_ground_truth import OCAtariGroundTruth


def find_experiment_dirs(base_dir: Path, game: str) -> List[Path]:
    """Find all experiment directories for a given game."""
    experiment_dirs = []

    if not base_dir.exists():
        print(f"Error: Directory {base_dir} does not exist")
        return []

    # Look for directories matching the game name
    game_lower = game.lower()
    for item in base_dir.iterdir():
        if item.is_dir() and game_lower in item.name.lower():
            # Check if it has Results subdirectory
            results_dir = item / "Results"
            if results_dir.exists():
                experiment_dirs.append(item)

    return sorted(experiment_dirs)


def load_vlm_detections(experiment_dir: Path, frame_number: int) -> Dict[str, Any]:
    """Load VLM detections from experiment results.

    Args:
        experiment_dir: Path to experiment directory
        frame_number: Frame number to load

    Returns:
        Dictionary with VLM detections
    """
    # Look for detection results in various possible locations
    possible_paths = [
        experiment_dir / "Results" / "detections" / f"frame_{frame_number:06d}_detections.json",
        experiment_dir / "Results" / f"frame_{frame_number:06d}" / "analysis.json",
        experiment_dir / f"frame_{frame_number:06d}" / "analysis.json",
    ]

    for path in possible_paths:
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)

                # Extract symbolic_state if it's wrapped
                if 'symbolic_state' in data:
                    return {'objects': data['symbolic_state'].get('objects', [])}
                elif 'objects' in data:
                    return data
                else:
                    continue

    # If no detections found, return empty
    return {'objects': []}


def generate_ocatari_ground_truth(game: str, seed: int, num_frames: int, actions: List[int]) -> List[Dict]:
    """Generate OCAtari ground truth for comparison.

    Args:
        game: Game name
        seed: Random seed
        num_frames: Number of frames to generate
        actions: List of actions taken at each frame

    Returns:
        List of ground truth data for each frame
    """
    print(f"Generating OCAtari ground truth for {game} (seed={seed}, frames={num_frames})")

    # Map game names to OCAtari format
    game_map = {
        'pong': 'Pong',
        'breakout': 'Breakout',
        'spaceinvaders': 'SpaceInvaders',
        'space_invaders': 'SpaceInvaders'
    }

    ocatari_game_name = game_map.get(game.lower(), game)

    # Initialize OCAtari
    extractor = OCAtariGroundTruth(ocatari_game_name)
    extractor.reset(seed=seed)

    ground_truth_data = []

    # Generate ground truth for each frame
    for frame_idx in range(num_frames):
        # Get current state
        frame, objects = extractor.get_frame_and_objects()

        # Store ground truth
        gt_data = {
            'frame': frame_idx,
            'objects': [obj.to_dict() for obj in objects]
        }
        ground_truth_data.append(gt_data)

        # Take action for next frame
        if frame_idx < len(actions):
            action = actions[frame_idx]
        else:
            action = 0  # NOOP

        extractor.step(action)

    extractor.close()

    return ground_truth_data


def load_actions_from_experiment(experiment_dir: Path) -> List[int]:
    """Load action sequence from experiment results.

    Args:
        experiment_dir: Path to experiment directory

    Returns:
        List of actions taken
    """
    actions_file = experiment_dir / "Results" / "actions_rewards.csv"

    if not actions_file.exists():
        print(f"Warning: Actions file not found: {actions_file}")
        return []

    actions = []
    with open(actions_file, 'r') as f:
        # Skip header
        next(f)
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 3:
                action = int(parts[2])  # Action is 3rd column
                actions.append(action)

    return actions


def evaluate_experiment(experiment_dir: Path, game: str, frames_to_eval: List[int] = None) -> Dict[str, Any]:
    """Evaluate a single experiment.

    Args:
        experiment_dir: Path to experiment directory
        game: Game name
        frames_to_eval: List of frame numbers to evaluate (None = all)

    Returns:
        Evaluation results dictionary
    """
    print(f"\n{'='*80}")
    print(f"Evaluating: {experiment_dir.name}")
    print(f"{'='*80}")

    # Extract seed from directory name
    dir_name = experiment_dir.name
    seed = None
    for part in dir_name.split('_'):
        if part.startswith('seed') and part[4:].isdigit():
            seed = int(part[4:])
            break

    if seed is None:
        print("Warning: Could not extract seed from directory name, using seed=42")
        seed = 42

    # Load actions
    actions = load_actions_from_experiment(experiment_dir)

    if not actions:
        print("Error: No actions found, cannot generate ground truth")
        return None

    # Determine frames to evaluate
    if frames_to_eval is None:
        # Evaluate every 10th frame by default
        frames_to_eval = list(range(0, len(actions), 10))
    else:
        frames_to_eval = [f for f in frames_to_eval if f < len(actions)]

    print(f"Evaluating {len(frames_to_eval)} frames: {frames_to_eval[:10]}{'...' if len(frames_to_eval) > 10 else ''}")

    # Generate ground truth
    max_frame = max(frames_to_eval) + 1 if frames_to_eval else len(actions)
    ground_truth_data = generate_ocatari_ground_truth(game, seed, max_frame, actions)

    # Initialize evaluator
    evaluator = CoordinateAccuracyEvaluator(game)

    # Evaluate each frame
    results = []
    for frame_num in frames_to_eval:
        if frame_num >= len(ground_truth_data):
            continue

        # Load VLM detections
        vlm_detections = load_vlm_detections(experiment_dir, frame_num)

        # Get ground truth
        gt_data = ground_truth_data[frame_num]

        # Evaluate
        result = evaluator.evaluate_frame(gt_data, vlm_detections, frame_id=frame_num)
        results.append(result)

        print(f"  Frame {frame_num}: "
              f"Precision={result.precision:.3f}, "
              f"Recall={result.recall:.3f}, "
              f"F1={result.f1_score:.3f}, "
              f"Important F1={result.important_f1:.3f}")

    # Aggregate results
    if results:
        aggregate = {
            'experiment': experiment_dir.name,
            'game': game,
            'seed': seed,
            'frames_evaluated': len(results),
            'avg_precision': np.mean([r.precision for r in results]),
            'avg_recall': np.mean([r.recall for r in results]),
            'avg_f1': np.mean([r.f1_score for r in results]),
            'avg_important_precision': np.mean([r.important_precision for r in results]),
            'avg_important_recall': np.mean([r.important_recall for r in results]),
            'avg_important_f1': np.mean([r.important_f1 for r in results]),
            'avg_iou': np.mean([r.avg_iou for r in results]),
            'avg_center_distance': np.mean([r.avg_center_distance for r in results]),
            'frame_results': [
                {
                    'frame': r.frame_id,
                    'precision': r.precision,
                    'recall': r.recall,
                    'f1': r.f1_score,
                    'important_f1': r.important_f1,
                    'iou': r.avg_iou,
                    'center_distance': r.avg_center_distance
                }
                for r in results
            ]
        }

        # Save results
        output_dir = experiment_dir / "Results" / "coordinate_evaluation"
        output_dir.mkdir(exist_ok=True)

        output_file = output_dir / "evaluation_summary.json"
        with open(output_file, 'w') as f:
            json.dump(aggregate, f, indent=2)

        print(f"\n✅ Evaluation complete! Results saved to: {output_file}")
        print(f"\nSUMMARY:")
        print(f"  Average Precision: {aggregate['avg_precision']:.3f}")
        print(f"  Average Recall: {aggregate['avg_recall']:.3f}")
        print(f"  Average F1: {aggregate['avg_f1']:.3f}")
        print(f"  Important Objects F1: {aggregate['avg_important_f1']:.3f}")

        return aggregate

    return None


def generate_aggregate_report(all_results: List[Dict], output_path: Path):
    """Generate aggregate report across multiple experiments."""
    if not all_results:
        print("No results to aggregate")
        return

    report = f"""
{'='*80}
AGGREGATE COORDINATE ACCURACY REPORT
{'='*80}

Total Experiments Evaluated: {len(all_results)}

OVERALL METRICS (Mean ± Std):
"""

    # Calculate aggregate statistics
    metrics = {
        'precision': [r['avg_precision'] for r in all_results],
        'recall': [r['avg_recall'] for r in all_results],
        'f1': [r['avg_f1'] for r in all_results],
        'important_precision': [r['avg_important_precision'] for r in all_results],
        'important_recall': [r['avg_important_recall'] for r in all_results],
        'important_f1': [r['avg_important_f1'] for r in all_results],
        'iou': [r['avg_iou'] for r in all_results],
        'center_distance': [r['avg_center_distance'] for r in all_results]
    }

    for metric_name, values in metrics.items():
        mean = np.mean(values)
        std = np.std(values)
        report += f"  {metric_name.replace('_', ' ').title()}: {mean:.3f} ± {std:.3f}\n"

    report += f"\nPER-EXPERIMENT BREAKDOWN:\n"
    for result in all_results:
        report += f"\n  {result['experiment']}:\n"
        report += f"    Precision: {result['avg_precision']:.3f}\n"
        report += f"    Recall: {result['avg_recall']:.3f}\n"
        report += f"    F1: {result['avg_f1']:.3f}\n"
        report += f"    Important F1: {result['avg_important_f1']:.3f}\n"

    report += f"\n{'='*80}\n"

    # Save report
    with open(output_path, 'w') as f:
        f.write(report)

    print(report)
    print(f"\n✅ Aggregate report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM detection accuracy vs OCAtari ground truth")
    parser.add_argument('--experiment_dir', type=str, required=True,
                       help='Path to experiment directory or parent directory containing experiments')
    parser.add_argument('--game', type=str, required=True,
                       choices=['pong', 'breakout', 'spaceinvaders', 'space_invaders'],
                       help='Game name')
    parser.add_argument('--frames', type=str, default=None,
                       help='Comma-separated list of frame numbers to evaluate (default: every 10th frame)')
    parser.add_argument('--aggregate', action='store_true',
                       help='Aggregate results across multiple experiments')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for aggregate report (default: experiment_dir)')

    args = parser.parse_args()

    # Parse frames
    frames_to_eval = None
    if args.frames:
        try:
            frames_to_eval = [int(f.strip()) for f in args.frames.split(',')]
        except ValueError:
            print(f"Error: Invalid frame numbers: {args.frames}")
            sys.exit(1)

    # Find experiments
    base_dir = Path(args.experiment_dir)

    if args.aggregate:
        # Find all experiment directories
        experiment_dirs = find_experiment_dirs(base_dir, args.game)

        if not experiment_dirs:
            print(f"Error: No experiment directories found for game '{args.game}' in {base_dir}")
            sys.exit(1)

        print(f"Found {len(experiment_dirs)} experiment directories")

        # Evaluate each experiment
        all_results = []
        for exp_dir in experiment_dirs:
            result = evaluate_experiment(exp_dir, args.game, frames_to_eval)
            if result:
                all_results.append(result)

        # Generate aggregate report
        output_dir = Path(args.output_dir) if args.output_dir else base_dir
        output_path = output_dir / f"{args.game}_coordinate_evaluation_aggregate.txt"
        generate_aggregate_report(all_results, output_path)

    else:
        # Evaluate single experiment
        if not base_dir.exists():
            print(f"Error: Directory {base_dir} does not exist")
            sys.exit(1)

        evaluate_experiment(base_dir, args.game, frames_to_eval)


if __name__ == "__main__":
    main()
