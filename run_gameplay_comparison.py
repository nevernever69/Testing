#!/usr/bin/env python3
"""
Run gameplay comparison between Vision-Only and Vision+Symbol pipelines.

Runs multiple seeds for each pipeline and performs statistical comparison.
This is the REAL test of spatial reasoning advantage.
"""

import argparse
import subprocess
import json
from pathlib import Path
from gameplay_metrics import aggregate_seeds, compare_pipelines, print_comparison_report


def run_single_seed(pipeline: str, game: str, seed: int, output_dir: str,
                   provider: str, model: str, aws_region: str,
                   num_frames: int = 600, detection_model: str = None):
    """Run a single seed of a pipeline."""
    
    print(f"\n{'='*70}")
    print(f"Running {pipeline} - Seed {seed}")
    print(f"{'='*70}\n")
    
    if pipeline == 'vision_only':
        cmd = [
            'python', 'direct_frame_runner.py',
            '--env', f'ALE/{game}-v5',
            '--provider', provider,
            '--model', model,
            '--aws_region', aws_region,
            '--num_frames', str(num_frames),
            '--seed', str(seed),
            '--output', output_dir,
            '--track_metrics'  # New flag we'll add
        ]
    else:  # vision_symbol
        cmd = [
            'python', 'advance_game_runner.py',
            '--env', f'ALE/{game}-v5',
            '--provider', provider,
            '--model', model,
            '--aws_region', aws_region,
            '--num_frames', str(num_frames),
            '--seed', str(seed),
            '--output', output_dir,
            '--track_metrics'  # New flag we'll add
        ]
        
        if detection_model:
            cmd.extend(['--detection_model', detection_model])
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR running seed {seed}:")
        print(e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description='Run gameplay comparison with multiple seeds')
    parser.add_argument('--game', type=str, required=True,
                       choices=['Pong', 'Breakout', 'SpaceInvaders'],
                       help='Game to test')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456],
                       help='Seeds to run (default: 42 123 456)')
    parser.add_argument('--provider', type=str, default='bedrock')
    parser.add_argument('--model', type=str, default='claude-4.5-sonnet')
    parser.add_argument('--aws_region', type=str, default='us-east-1')
    parser.add_argument('--num_frames', type=int, default=600,
                       help='Frames to run per episode')
    parser.add_argument('--output', type=str, default='./gameplay_comparison',
                       help='Output directory')
    parser.add_argument('--detection_model', type=str, default='claude-4-sonnet',
                       help='Model for Vision+Symbol detection')
    parser.add_argument('--skip_vision_only', action='store_true',
                       help='Skip Vision-Only runs')
    parser.add_argument('--skip_vision_symbol', action='store_true',
                       help='Skip Vision+Symbol runs')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)
    
    print("="*70)
    print(f"GAMEPLAY COMPARISON: {args.game}")
    print("="*70)
    print(f"Seeds: {args.seeds}")
    print(f"Frames per run: {args.num_frames}")
    print(f"Output: {output_dir}")
    print("="*70)
    
    # Run Vision-Only seeds
    vision_only_files = []
    if not args.skip_vision_only:
        print("\n\n" + "="*70)
        print("PHASE 1: VISION-ONLY PIPELINE")
        print("="*70)
        
        for seed in args.seeds:
            success = run_single_seed(
                pipeline='vision_only',
                game=args.game,
                seed=seed,
                output_dir=str(output_dir / 'vision_only'),
                provider=args.provider,
                model=args.model,
                aws_region=args.aws_region,
                num_frames=args.num_frames
            )
            
            if success:
                # Metrics file will be saved by the runner
                metrics_file = metrics_dir / f"gameplay_metrics_visiononly_{seed}.json"
                if metrics_file.exists():
                    vision_only_files.append(metrics_file)
    
    # Run Vision+Symbol seeds
    vision_symbol_files = []
    if not args.skip_vision_symbol:
        print("\n\n" + "="*70)
        print("PHASE 2: VISION+SYMBOL PIPELINE")
        print("="*70)
        
        for seed in args.seeds:
            success = run_single_seed(
                pipeline='vision_symbol',
                game=args.game,
                seed=seed,
                output_dir=str(output_dir / 'vision_symbol'),
                provider=args.provider,
                model=args.model,
                aws_region=args.aws_region,
                num_frames=args.num_frames,
                detection_model=args.detection_model
            )
            
            if success:
                metrics_file = metrics_dir / f"gameplay_metrics_visionsymbol_{seed}.json"
                if metrics_file.exists():
                    vision_symbol_files.append(metrics_file)
    
    # Aggregate and compare
    if vision_only_files and vision_symbol_files:
        print("\n\n" + "="*70)
        print("PHASE 3: AGGREGATION & COMPARISON")
        print("="*70)
        
        # Aggregate each pipeline
        vision_only_agg = aggregate_seeds(vision_only_files)
        vision_symbol_agg = aggregate_seeds(vision_symbol_files)
        
        # Save aggregated results
        vo_agg_file = metrics_dir / "aggregated_vision_only.json"
        vs_agg_file = metrics_dir / "aggregated_vision_symbol.json"
        
        with open(vo_agg_file, 'w') as f:
            json.dump(vision_only_agg, f, indent=2)
        
        with open(vs_agg_file, 'w') as f:
            json.dump(vision_symbol_agg, f, indent=2)
        
        print(f"\n✓ Aggregated Vision-Only: {vo_agg_file}")
        print(f"✓ Aggregated Vision+Symbol: {vs_agg_file}")
        
        # Statistical comparison
        comparison = compare_pipelines(vo_agg_file, vs_agg_file)
        
        # Save comparison
        comparison_file = output_dir / "gameplay_comparison_results.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        print(f"✓ Comparison results: {comparison_file}")
        
        # Print report
        print_comparison_report(comparison)
        
        # Summary for paper
        print("\n" + "="*70)
        print("PAPER-READY SUMMARY")
        print("="*70)
        print(f"\nGame: {args.game}")
        print(f"Number of seeds: {len(args.seeds)}")
        print(f"Frames per episode: {args.num_frames}")
        print(f"\nVision-Only:  {vision_only_agg['final_score']['mean']:.2f} ± {vision_only_agg['final_score']['std']:.2f}")
        print(f"Vision+Symbol: {vision_symbol_agg['final_score']['mean']:.2f} ± {vision_symbol_agg['final_score']['std']:.2f}")
        print(f"\nImprovement: {comparison['improvement']['absolute']:+.2f} ({comparison['improvement']['percentage']:+.1f}%)")
        print(f"p-value: {comparison['statistical_tests']['p_value']:.4f}")
        print(f"Effect size: {comparison['statistical_tests']['cohens_d']:.3f} ({comparison['statistical_tests']['effect_size_interpretation']})")
        print("\n" + "="*70)
    
    else:
        print("\n⚠️  Missing results - cannot perform comparison")
        print(f"Vision-Only files: {len(vision_only_files)}")
        print(f"Vision+Symbol files: {len(vision_symbol_files)}")


if __name__ == "__main__":
    main()
