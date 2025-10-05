#!/usr/bin/env python3
"""
Run spatial control benchmark on both pipelines - CORRECTED VERSION.

CRITICAL FIX:
- Vision-Only: Gets question WITHOUT coordinates (tests visual estimation)
- Vision+Symbol: Gets question WITH coordinates (tests math ability)

This tests CONTROL capability, not just descriptive ability.
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from spatial_control_benchmark import SpatialControlEvaluator
from automatic_benchmark.pipeline_adapters import DirectFrameAdapter, AdvancedGameAdapter
from automatic_benchmark import BenchmarkDatasetLoader


def run_spatial_control_benchmark(
    pipeline,
    pipeline_name: str,
    dataset_path: str,
    output_dir: str,
    game_filter: str = None,
    limit: int = None
):
    """Run spatial control tests on a pipeline with appropriate questions."""

    loader = BenchmarkDatasetLoader(dataset_path)
    evaluator = SpatialControlEvaluator()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create detailed logs directory
    logs_dir = output_path / "detailed_logs" / pipeline_name.replace(' ', '_').replace('+', '')
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Spatial Control Benchmark: {pipeline_name}")
    print(f"{'='*70}\n")

    # Determine if this pipeline should get coordinates in the question
    include_coordinates = 'symbol' in pipeline_name.lower()

    print(f"Question type: {'WITH coordinates' if include_coordinates else 'WITHOUT coordinates (vision only)'}")
    print(f"Detailed logs: {logs_dir}\n")

    # Get frames
    all_frames = loader.get_frames()
    if game_filter:
        frames = [f for f in all_frames if f['game'].lower() == game_filter.lower()]
    else:
        frames = all_frames

    # Apply limit if specified
    if limit is not None:
        frames = frames[:limit]
        print(f"⚠️  Limited to {limit} frames for testing\n")

    print(f"Testing on {len(frames)} frames\n")

    results = {
        'pipeline_name': pipeline_name,
        'timestamp': datetime.now().isoformat(),
        'include_coordinates_in_question': include_coordinates,
        'tests': []
    }

    for frame_data in tqdm(frames, desc=f"Running {pipeline_name}"):
        frame_id = frame_data['frame_id']

        # Load ground truth
        ground_truth = loader.load_ground_truth(frame_id)
        if not ground_truth:
            continue

        # Create movement distance test WITH/WITHOUT coordinates based on pipeline
        test = evaluator.create_movement_distance_test(
            ground_truth,
            include_coordinates=include_coordinates
        )
        if not test:
            continue

        # Load frame image
        frame = loader.load_frame(frame_id)
        if frame is None:
            continue

        try:
            # Get VLM response
            response = pipeline.process(frame, test.question)

            # Evaluate objectively
            eval_result = evaluator.evaluate_movement_distance(response, test.ground_truth)

            # Store result
            test_result = {
                'frame_id': frame_id,
                'test_type': test.test_type,
                'question': test.question,
                'response': response,
                'score': eval_result['score'],
                'error_pixels': eval_result['error'],
                'predicted': eval_result['predicted'],
                'optimal': eval_result['optimal'],
                'ground_truth': test.ground_truth,
                'metadata': test.metadata
            }

            results['tests'].append(test_result)

            # Save detailed log for this test
            log_file = logs_dir / f"{frame_id}_log.json"
            with open(log_file, 'w') as f:
                json.dump({
                    'frame_id': frame_id,
                    'pipeline': pipeline_name,
                    'include_coordinates': include_coordinates,
                    'question': test.question,
                    'response': response,
                    'evaluation': eval_result,
                    'ground_truth': test.ground_truth
                }, f, indent=2)

            print(f"  {frame_id}: {eval_result['score']:.2f} (error: {eval_result['error']}px)")

        except Exception as e:
            print(f"  ERROR on {frame_id}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Compute statistics
    if results['tests']:
        scores = [t['score'] for t in results['tests']]
        errors = [t['error_pixels'] for t in results['tests']]

        results['statistics'] = {
            'mean_score': sum(scores) / len(scores),
            'mean_error_pixels': sum(errors) / len(errors),
            'num_tests': len(results['tests']),
            'perfect_predictions': sum(1 for s in scores if s >= 0.95)
        }
    else:
        results['statistics'] = {
            'mean_score': 0.0,
            'mean_error_pixels': 0.0,
            'num_tests': 0,
            'perfect_predictions': 0
        }

    # Save results (consistent naming: vision_only, vision_symbol)
    clean_name = pipeline_name.lower().replace(' ', '_').replace('+', '').replace('-', '_')
    output_file = output_path / f"{clean_name}_spatial_control.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Results for {pipeline_name}:")
    print(f"  Mean Score: {results['statistics']['mean_score']:.3f}")
    print(f"  Mean Error: {results['statistics']['mean_error_pixels']:.1f} pixels")
    print(f"  Perfect Predictions: {results['statistics']['perfect_predictions']}/{results['statistics']['num_tests']}")
    print(f"  Saved to: {output_file}")
    print(f"  Logs saved to: {logs_dir}")
    print(f"{'='*70}\n")

    return results


def compare_results(baseline_file: Path, comparison_file: Path):
    """Compare spatial control results."""
    with open(baseline_file) as f:
        baseline = json.load(f)
    with open(comparison_file) as f:
        comparison = json.load(f)

    baseline_score = baseline['statistics']['mean_score']
    comparison_score = comparison['statistics']['mean_score']

    improvement = comparison_score - baseline_score
    improvement_pct = (improvement / baseline_score * 100) if baseline_score > 0 else float('inf')

    print(f"\n{'='*70}")
    print("SPATIAL CONTROL COMPARISON")
    print(f"{'='*70}")
    print(f"\n{baseline['pipeline_name']:20} → Score: {baseline_score:.3f}")
    print(f"  Question: {'WITHOUT' if not baseline['include_coordinates_in_question'] else 'WITH'} coordinates")
    print(f"\n{comparison['pipeline_name']:20} → Score: {comparison_score:.3f}")
    print(f"  Question: {'WITHOUT' if not comparison['include_coordinates_in_question'] else 'WITH'} coordinates")
    print(f"\nImprovement: {improvement:+.3f} ({improvement_pct:+.1f}%)")
    print(f"\nThis is the REAL spatial reasoning advantage!")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Run spatial control benchmark (FIXED VERSION)')
    parser.add_argument('--pipeline', type=str, required=True,
                       choices=['vision_only', 'vision_symbol', 'both'])
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output', type=str, default='./spatial_control_results')
    parser.add_argument('--provider', type=str, default='bedrock')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--aws_region', type=str, default='us-east-1')
    parser.add_argument('--game', type=str, default=None)
    parser.add_argument('--game_type', type=str, default='breakout')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of frames to test (for quick testing)')

    args = parser.parse_args()

    # Run benchmarks
    if args.pipeline in ['vision_only', 'both']:
        print("\n" + "="*70)
        print("VISION-ONLY PIPELINE")
        print("="*70)

        vision_only = DirectFrameAdapter(
            provider=args.provider,
            model_id=args.model,
            aws_region=args.aws_region
        )

        baseline_results = run_spatial_control_benchmark(
            vision_only,
            'Vision-Only',
            args.dataset,
            args.output,
            game_filter=args.game,
            limit=args.limit
        )

    if args.pipeline in ['vision_symbol', 'both']:
        print("\n" + "="*70)
        print("VISION+SYMBOL PIPELINE")
        print("="*70)

        vision_symbol = AdvancedGameAdapter(
            provider=args.provider,
            model_id=args.model,
            aws_region=args.aws_region,
            game_type=args.game_type
        )

        comparison_results = run_spatial_control_benchmark(
            vision_symbol,
            'Vision+Symbol',
            args.dataset,
            args.output,
            game_filter=args.game,
            limit=args.limit
        )

    # Compare if both were run
    if args.pipeline == 'both':
        compare_results(
            Path(args.output) / 'vision_only_spatial_control.json',
            Path(args.output) / 'visionsymbol_spatial_control.json'
        )


if __name__ == "__main__":
    main()
