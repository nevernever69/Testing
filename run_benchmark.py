#!/usr/bin/env python3
"""
Run the automatic benchmark on your pipelines.

Usage:
    # Test vision-only pipeline
    python run_benchmark.py --pipeline vision_only --dataset ./benchmark_v1.0

    # Test vision+symbol pipeline
    python run_benchmark.py --pipeline vision_symbol --dataset ./benchmark_v1.0

    # Compare both
    python run_benchmark.py --pipeline both --dataset ./benchmark_v1.0
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import numpy as np

from automatic_benchmark import AutomaticEvaluator, BenchmarkDatasetLoader
from automatic_benchmark.pipeline_adapters import DirectFrameAdapter, AdvancedGameAdapter
from automatic_benchmark.utils import get_all_prompts
from automatic_benchmark.utils.detailed_logger import DetailedBenchmarkLogger


class BenchmarkRunner:
    """Runs benchmark evaluation on pipelines."""

    def __init__(self, dataset_path: str, output_dir: str, evaluator: AutomaticEvaluator, enable_detailed_logging: bool = True):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.loader = BenchmarkDatasetLoader(dataset_path)
        self.evaluator = evaluator

        # Initialize detailed logger
        self.detailed_logging = enable_detailed_logging
        if self.detailed_logging:
            self.logger = DetailedBenchmarkLogger(str(self.output_dir / "detailed_logs"))
        else:
            self.logger = None

        print(f"\n{'='*70}")
        print(f"Benchmark Runner Initialized")
        print(f"{'='*70}")
        print(f"Dataset: {dataset_path}")
        print(f"Total frames: {len(self.loader)}")
        print(f"Output: {output_dir}")
        if self.detailed_logging:
            print(f"Detailed logging: {self.logger.get_session_dir()}")
        print(f"{'='*70}\n")

    def run_single_pipeline(self, pipeline, pipeline_name: str, limit: int = None, game_filter: str = None) -> dict:
        """
        Run benchmark on a single pipeline.

        Args:
            pipeline: Pipeline adapter instance
            pipeline_name: Name for results

        Returns:
            Results dictionary
        """
        print(f"\n{'='*70}")
        print(f"Evaluating: {pipeline_name}")
        print(f"{'='*70}\n")

        # Get all task prompts
        prompts = get_all_prompts()

        # Results storage
        results = {
            'pipeline_name': pipeline_name,
            'dataset_path': str(self.dataset_path),
            'timestamp': datetime.now().isoformat(),
            'evaluations': []
        }

        # Get all frames
        if game_filter:
            # Try case-insensitive match
            all_frames = self.loader.get_frames()
            frames = [f for f in all_frames if f['game'].lower() == game_filter.lower()]
            print(f"🎮 Filtering to game: {game_filter} ({len(frames)} frames)\n")
        else:
            frames = self.loader.get_frames()

        # Limit frames if specified
        if limit is not None:
            frames = frames[:limit]
            print(f"⚠️  Limited to {limit} frames for testing\n")

        # Evaluate each frame
        for frame_data in tqdm(frames, desc=f"Evaluating {pipeline_name}"):
            frame_id = frame_data['frame_id']
            game = frame_data['game']

            # Load frame and ground truth
            frame = self.loader.load_frame(frame_id)
            ground_truth = self.loader.load_ground_truth(frame_id)

            if frame is None or ground_truth is None:
                print(f"Warning: Could not load frame {frame_id}, skipping")
                continue

            # Evaluate all 4 tasks
            frame_results = {
                'frame_id': frame_id,
                'game': game,
                'complexity': frame_data.get('complexity', {}),
                'tasks': {}
            }

            for task_type, prompt in prompts.items():
                try:
                    # Get VLM response
                    response = pipeline.process(frame, prompt)

                    # Evaluate response
                    eval_result = self.evaluator.evaluate(
                        response=response,
                        task_type=task_type,
                        ground_truth=ground_truth,
                        game_name=game
                    )

                    # Store results
                    frame_results['tasks'][task_type] = {
                        'prompt': prompt,
                        'response': response,
                        'score': eval_result.final_score,
                        'confidence': eval_result.confidence,
                        'reasoning': eval_result.reasoning,
                        'rule_based_score': eval_result.rule_based_result['score'] if eval_result.rule_based_result else None,
                        'semantic_score': eval_result.semantic_result['score'] if eval_result.semantic_result else None,
                        'llm_judge_score': eval_result.llm_judge_result['score'] if eval_result.llm_judge_result else None,
                        'combination_method': eval_result.combination_method
                    }

                    # Detailed logging
                    if self.detailed_logging and self.logger:
                        # Get LLM judge details if available
                        llm_judge_prompt = None
                        llm_judge_response = None
                        if eval_result.llm_judge_result and 'details' in eval_result.llm_judge_result:
                            llm_judge_prompt = eval_result.llm_judge_result['details'].get('judge_prompt')
                            judge_raw = eval_result.llm_judge_result['details'].get('judge_raw_response')
                            if judge_raw:
                                llm_judge_response = json.dumps(judge_raw, indent=2)

                        # Get actual prompt and detection results (for Vision+Symbol)
                        actual_prompt = prompt
                        detection_results = None
                        if hasattr(pipeline, 'get_actual_prompt'):
                            actual_prompt = pipeline.get_actual_prompt()
                        if hasattr(pipeline, 'get_detection_results'):
                            detection_results = pipeline.get_detection_results()

                        # Log complete evaluation
                        self.logger.log_evaluation(
                            frame_id=frame_id,
                            pipeline=pipeline_name,
                            task_type=task_type,
                            task_prompt=actual_prompt,
                            vlm_response=response,
                            ground_truth=ground_truth,
                            eval_result=eval_result,
                            llm_judge_prompt=llm_judge_prompt,
                            llm_judge_response=llm_judge_response,
                            frame=frame,
                            detection_results=detection_results
                        )

                    print(f"  {frame_id} - {task_type}: {eval_result.final_score:.3f}")

                except Exception as e:
                    print(f"  ERROR evaluating {frame_id} - {task_type}: {e}")
                    frame_results['tasks'][task_type] = {
                        'prompt': prompt,
                        'response': f"ERROR: {str(e)}",
                        'score': 0.0,
                        'confidence': 0.0,
                        'reasoning': f"Evaluation error: {str(e)}"
                    }

            results['evaluations'].append(frame_results)

        # Compute aggregate statistics
        results['statistics'] = self._compute_statistics(results['evaluations'])

        # Get evaluator statistics
        results['evaluator_stats'] = self.evaluator.get_statistics()

        # Save results
        results_file = self.output_dir / f"{pipeline_name.lower().replace(' ', '_').replace('+', '_').replace('-', '_')}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*70}")
        print(f"Results saved: {results_file}")
        print(f"{'='*70}\n")

        # Save detailed logs summary
        if self.detailed_logging and self.logger:
            self.logger.save_summary()

        # Print summary
        self._print_summary(results)

        return results

    def _compute_statistics(self, evaluations: list) -> dict:
        """Compute aggregate statistics from evaluations."""
        stats = {
            'overall': {},
            'per_task': {},
            'per_game': {},
            'per_complexity': {}
        }

        # Collect scores
        all_scores = []
        task_scores = {'visual': [], 'spatial': [], 'strategy': [], 'identification': []}
        game_scores = {}
        complexity_scores = {'easy': [], 'medium': [], 'hard': []}

        for eval in evaluations:
            game = eval['game']
            complexity = eval.get('complexity', {}).get('complexity_category', 'unknown')

            if game not in game_scores:
                game_scores[game] = []

            for task, result in eval['tasks'].items():
                score = result['score']
                all_scores.append(score)
                task_scores[task].append(score)
                game_scores[game].append(score)

                if complexity in complexity_scores:
                    complexity_scores[complexity].append(score)

        # Overall statistics
        if all_scores:
            stats['overall'] = {
                'mean': float(np.mean(all_scores)),
                'std': float(np.std(all_scores)),
                'median': float(np.median(all_scores)),
                'min': float(np.min(all_scores)),
                'max': float(np.max(all_scores)),
                'n': len(all_scores)
            }

        # Per-task statistics
        for task, scores in task_scores.items():
            if scores:
                stats['per_task'][task] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'n': len(scores)
                }

        # Per-game statistics
        for game, scores in game_scores.items():
            if scores:
                stats['per_game'][game] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'n': len(scores)
                }

        # Per-complexity statistics
        for complexity, scores in complexity_scores.items():
            if scores:
                stats['per_complexity'][complexity] = {
                    'mean': float(np.mean(scores)),
                    'std': float(np.std(scores)),
                    'n': len(scores)
                }

        return stats

    def _print_summary(self, results: dict):
        """Print summary of results."""
        stats = results['statistics']

        print(f"\n{'='*70}")
        print(f"SUMMARY: {results['pipeline_name']}")
        print(f"{'='*70}\n")

        # Check if any evaluations were run
        if not results['evaluations']:
            print("⚠️  No evaluations were run (0 frames matched filters)")
            return

        # Overall
        if 'overall' in stats and stats['overall']:
            overall = stats['overall']
            print(f"Overall Score: {overall['mean']:.3f} ± {overall['std']:.3f} (n={overall['n']})")

        # Per-task
        if 'per_task' in stats:
            print(f"\nPer-Task Scores:")
            for task, task_stats in stats['per_task'].items():
                print(f"  {task.capitalize():15}: {task_stats['mean']:.3f} ± {task_stats['std']:.3f} (n={task_stats['n']})")

        # Per-game
        if 'per_game' in stats:
            print(f"\nPer-Game Scores:")
            for game, game_stats in stats['per_game'].items():
                print(f"  {game:20}: {game_stats['mean']:.3f} ± {game_stats['std']:.3f} (n={game_stats['n']})")

        # Evaluator stats
        if 'evaluator_stats' in results:
            eval_stats = results['evaluator_stats']
            print(f"\nEvaluator Statistics:")
            print(f"  Total evaluations: {eval_stats.get('total_evaluations', 0)}")
            print(f"  LLM judge usage: {eval_stats.get('llm_judge_usage_rate', 0):.1%}")
            print(f"  Estimated cost: ${eval_stats.get('llm_judge_cost_estimate', 0):.2f}")

        print(f"\n{'='*70}\n")

    def compare_results(self, baseline_file: Path, comparison_file: Path) -> dict:
        """Compare two result files."""
        print(f"\n{'='*70}")
        print(f"Comparing Results")
        print(f"{'='*70}\n")

        # Load results
        with open(baseline_file, 'r') as f:
            baseline = json.load(f)

        with open(comparison_file, 'r') as f:
            comparison = json.load(f)

        # Compare
        comparison_results = {
            'baseline_name': baseline['pipeline_name'],
            'comparison_name': comparison['pipeline_name'],
            'overall_comparison': {},
            'task_comparison': {},
            'game_comparison': {}
        }

        # Overall comparison (normalize to 0-100 scale)
        baseline_overall = baseline['statistics']['overall']['mean']
        comparison_overall = comparison['statistics']['overall']['mean']
        improvement = comparison_overall - baseline_overall

        comparison_results['overall_comparison'] = {
            'baseline_score': baseline_overall,
            'comparison_score': comparison_overall,
            'absolute_improvement': improvement,
            'baseline_score_100': baseline_overall * 100,
            'comparison_score_100': comparison_overall * 100,
            'improvement_points': improvement * 100
        }

        # Per-task comparison (normalize to 0-100 scale)
        for task in ['visual', 'spatial', 'strategy', 'identification']:
            if task in baseline['statistics']['per_task'] and task in comparison['statistics']['per_task']:
                baseline_score = baseline['statistics']['per_task'][task]['mean']
                comparison_score = comparison['statistics']['per_task'][task]['mean']
                improvement = comparison_score - baseline_score

                comparison_results['task_comparison'][task] = {
                    'baseline_score': baseline_score,
                    'comparison_score': comparison_score,
                    'absolute_improvement': improvement,
                    'baseline_score_100': baseline_score * 100,
                    'comparison_score_100': comparison_score * 100,
                    'improvement_points': improvement * 100
                }

        # Per-game comparison (normalize to 0-100 scale)
        for game in baseline['statistics'].get('per_game', {}).keys():
            if game in comparison['statistics'].get('per_game', {}):
                baseline_score = baseline['statistics']['per_game'][game]['mean']
                comparison_score = comparison['statistics']['per_game'][game]['mean']
                improvement = comparison_score - baseline_score

                comparison_results['game_comparison'][game] = {
                    'baseline_score': baseline_score,
                    'comparison_score': comparison_score,
                    'absolute_improvement': improvement,
                    'baseline_score_100': baseline_score * 100,
                    'comparison_score_100': comparison_score * 100,
                    'improvement_points': improvement * 100
                }

        # Save comparison
        comparison_output = self.output_dir / "comparison_results.json"
        with open(comparison_output, 'w') as f:
            json.dump(comparison_results, f, indent=2)

        # Print comparison
        self._print_comparison(comparison_results)

        return comparison_results

    def _print_comparison(self, comparison: dict):
        """Print comparison results."""
        print(f"\n{'='*70}")
        print(f"COMPARISON: {comparison['baseline_name']} vs {comparison['comparison_name']}")
        print(f"{'='*70}\n")

        # Overall
        overall = comparison['overall_comparison']
        print(f"Overall: {overall['baseline_score_100']:.1f} → {overall['comparison_score_100']:.1f} points ({overall['improvement_points']:+.1f})")

        # Per-task
        print(f"\nPer-Task Improvements:")
        for task, stats in comparison['task_comparison'].items():
            print(f"  {task.capitalize():15}: {stats['baseline_score_100']:.1f} → {stats['comparison_score_100']:.1f} points ({stats['improvement_points']:+.1f})")

        # Per-game
        if comparison['game_comparison']:
            print(f"\nPer-Game Improvements:")
            for game, stats in comparison['game_comparison'].items():
                print(f"  {game:20}: {stats['baseline_score_100']:.1f} → {stats['comparison_score_100']:.1f} points ({stats['improvement_points']:+.1f})")

        print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Run automatic benchmark on pipelines')
    parser.add_argument('--pipeline', type=str, required=True,
                       choices=['vision_only', 'vision_symbol', 'both'],
                       help='Which pipeline to evaluate')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Path to benchmark dataset')
    parser.add_argument('--output', type=str, default='./benchmark_results',
                       help='Output directory for results')
    parser.add_argument('--provider', type=str, default='bedrock',
                       help='LLM provider (bedrock, openai, anthropic)')
    parser.add_argument('--model', type=str, default=None,
                       help='Model ID')
    parser.add_argument('--aws_region', type=str, default='us-east-1',
                       help='AWS region for Bedrock')
    parser.add_argument('--openrouter_key', type=str, default=None,
                       help='OpenRouter API key for detection model')
    parser.add_argument('--detection_model', type=str, default=None,
                       help='Detection model for vision+symbol pipeline (auto-selects based on provider)')
    parser.add_argument('--game_type', type=str, default='pong',
                       choices=['pong', 'breakout', 'space_invaders'],
                       help='Game type for detector')
    parser.add_argument('--game', type=str, default=None,
                       help='Filter to only run on specific game (e.g., "pong", "breakout")')
    parser.add_argument('--use_llm_judge', action='store_true',
                       help='Enable LLM-as-judge scoring')
    parser.add_argument('--llm_judge_only', action='store_true',
                       help='Use ONLY LLM judge for scoring (disables rule-based and semantic scoring)')
    parser.add_argument('--llm_judge_provider', type=str, default=None,
                       choices=['openai', 'anthropic', 'bedrock'],
                       help='Provider for LLM judge (defaults to same as --provider)')
    parser.add_argument('--disable_detailed_logs', action='store_true',
                       help='Disable detailed logging (saves disk space)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of frames to evaluate (for testing)')

    args = parser.parse_args()

    # If llm_judge_provider not specified, use same provider as main pipeline
    if args.llm_judge_provider is None:
        args.llm_judge_provider = args.provider

    # Initialize evaluator
    evaluator = AutomaticEvaluator(
        use_semantic=True,
        use_llm_judge=args.use_llm_judge or args.llm_judge_only,
        llm_provider=args.llm_judge_provider if (args.use_llm_judge or args.llm_judge_only) else None,
        aws_region=args.aws_region,
        llm_judge_only=args.llm_judge_only
    )

    # Initialize runner
    runner = BenchmarkRunner(
        dataset_path=args.dataset,
        output_dir=args.output,
        evaluator=evaluator,
        enable_detailed_logging=not args.disable_detailed_logs
    )

    # Run benchmark(s)
    if args.pipeline in ['vision_only', 'both']:
        print("\n" + "="*70)
        print("VISION-ONLY PIPELINE")
        print("="*70)

        vision_only = DirectFrameAdapter(
            provider=args.provider,
            model_id=args.model,
            aws_region=args.aws_region
        )

        vision_only_results = runner.run_single_pipeline(vision_only, 'Vision-Only', limit=args.limit, game_filter=args.game)

    if args.pipeline in ['vision_symbol', 'both']:
        print("\n" + "="*70)
        print("VISION+SYMBOL PIPELINE")
        print("="*70)

        vision_symbol = AdvancedGameAdapter(
            provider=args.provider,
            model_id=args.model,
            openrouter_api_key=args.openrouter_key,
            detection_model=args.detection_model,
            aws_region=args.aws_region,
            game_type=args.game_type
        )

        vision_symbol_results = runner.run_single_pipeline(vision_symbol, 'Vision+Symbol', limit=args.limit, game_filter=args.game)

    # Compare if both were run
    if args.pipeline == 'both':
        runner.compare_results(
            Path(args.output) / 'vision_only_results.json',
            Path(args.output) / 'vision_symbol_results.json'
        )

    print(f"\n{'='*70}")
    print(f"Benchmark Complete!")
    print(f"Results saved to: {args.output}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
