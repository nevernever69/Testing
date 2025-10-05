#!/usr/bin/env python3
"""
Analyze comparison results between Vision-Only and Vision+Symbol.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import json


def load_results(base_dir, pattern):
    """Load all results matching pattern."""
    results = []
    base_path = Path(base_dir)

    # Try both possible locations: direct and in Results subdirectory
    search_patterns = [
        f'{pattern}/*/actions_rewards.csv',  # Direct location
        f'{pattern}/*/Results/actions_rewards.csv'  # Results subdirectory
    ]

    for search_pattern in search_patterns:
        for csv_path in base_path.glob(search_pattern):
            try:
                df = pd.read_csv(csv_path)
                final_score = df['cumulative_reward'].iloc[-1]
                frames = len(df)

                # Extract seed from path
                seed_str = str(csv_path).split('seed')[1].split('/')[0] if 'seed' in str(csv_path) else 'unknown'

                # Avoid duplicates (in case both locations exist)
                if not any(r['seed'] == seed_str and r['path'] in str(csv_path) for r in results):
                    results.append({
                        'seed': seed_str,
                        'final_score': final_score,
                        'frames': frames,
                        'mean_reward': final_score / frames,
                        'path': str(csv_path.parent)
                    })
            except Exception as e:
                print(f"Warning: Could not load {csv_path}: {e}")

    return results


def print_summary(results, pipeline_name):
    """Print summary statistics."""
    if not results:
        print(f"  No results found for {pipeline_name}")
        return

    scores = [r['final_score'] for r in results]
    frames = [r['frames'] for r in results]

    print(f"\n{pipeline_name} (n={len(results)}):")
    print(f"  Scores: {[f'{s:.2f}' for s in scores]}")
    print(f"  Mean Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"  Min/Max: {np.min(scores):.2f} / {np.max(scores):.2f}")
    print(f"  Frames: {frames[0]} (all runs)" if len(set(frames)) == 1 else f"  Frames: {frames}")


def compare_pipelines(vision_only, vision_symbol):
    """Compare two pipelines statistically."""
    if not vision_only or not vision_symbol:
        print("\nCannot compare - missing results for one or both pipelines")
        return

    vo_scores = [r['final_score'] for r in vision_only]
    vs_scores = [r['final_score'] for r in vision_symbol]

    print("\n" + "="*70)
    print("STATISTICAL COMPARISON")
    print("="*70)

    # Descriptive stats
    print(f"\nVision-Only:")
    print(f"  Mean: {np.mean(vo_scores):.2f}")
    print(f"  Std:  {np.std(vo_scores):.2f}")

    print(f"\nVision+Symbol:")
    print(f"  Mean: {np.mean(vs_scores):.2f}")
    print(f"  Std:  {np.std(vs_scores):.2f}")

    # Improvement
    improvement = np.mean(vs_scores) - np.mean(vo_scores)
    pct_improvement = (improvement / abs(np.mean(vo_scores))) * 100 if np.mean(vo_scores) != 0 else 0

    print(f"\nImprovement:")
    print(f"  Absolute: {improvement:+.2f}")
    print(f"  Relative: {pct_improvement:+.1f}%")

    # Statistical test
    if len(vo_scores) >= 2 and len(vs_scores) >= 2:
        t_stat, p_value = stats.ttest_ind(vs_scores, vo_scores)

        print(f"\nStatistical Test (Independent t-test):")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Significant at α=0.05: {'YES ✅' if p_value < 0.05 else 'NO ❌'}")
        print(f"  Significant at α=0.10: {'YES ✅' if p_value < 0.10 else 'NO ❌'}")

        if p_value < 0.05:
            if improvement > 0:
                print(f"\n🎉 Vision+Symbol is SIGNIFICANTLY BETTER than Vision-Only!")
            else:
                print(f"\n⚠️  Vision-Only is significantly better than Vision+Symbol")
        elif p_value < 0.10:
            print(f"\n📊 Marginally significant improvement (p < 0.10)")
        else:
            print(f"\n❌ No significant difference between pipelines")

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(vo_scores)**2 + np.std(vs_scores)**2) / 2)
        cohen_d = improvement / pooled_std if pooled_std > 0 else 0
        print(f"\nEffect Size (Cohen's d): {cohen_d:.3f}")
        if abs(cohen_d) < 0.2:
            print("  Interpretation: Small effect")
        elif abs(cohen_d) < 0.5:
            print("  Interpretation: Medium effect")
        else:
            print("  Interpretation: Large effect")
    else:
        print("\n⚠️  Need at least 2 samples per group for statistical test")


def save_summary(vision_only, vision_symbol, output_file):
    """Save summary to JSON."""
    summary = {
        'vision_only': {
            'n_runs': len(vision_only),
            'scores': [r['final_score'] for r in vision_only],
            'mean': float(np.mean([r['final_score'] for r in vision_only])) if vision_only else 0,
            'std': float(np.std([r['final_score'] for r in vision_only])) if vision_only else 0,
        },
        'vision_symbol': {
            'n_runs': len(vision_symbol),
            'scores': [r['final_score'] for r in vision_symbol],
            'mean': float(np.mean([r['final_score'] for r in vision_symbol])) if vision_symbol else 0,
            'std': float(np.std([r['final_score'] for r in vision_symbol])) if vision_symbol else 0,
        }
    }

    if vision_only and vision_symbol and len(vision_only) >= 2 and len(vision_symbol) >= 2:
        vo_scores = [r['final_score'] for r in vision_only]
        vs_scores = [r['final_score'] for r in vision_symbol]

        t_stat, p_value = stats.ttest_ind(vs_scores, vo_scores)
        improvement = np.mean(vs_scores) - np.mean(vo_scores)

        summary['comparison'] = {
            'improvement_absolute': float(improvement),
            'improvement_percent': float((improvement / abs(np.mean(vo_scores))) * 100) if np.mean(vo_scores) != 0 else 0,
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant_at_0.05': p_value < 0.05,
            'significant_at_0.10': p_value < 0.10
        }

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Summary saved to: {output_file}")


def main():
    import argparse
    import glob

    parser = argparse.ArgumentParser(description='Analyze gameplay comparison results')
    parser.add_argument('--results_dir', type=str, default=None,
                       help='Directory containing results (auto-detects if not specified)')
    parser.add_argument('--output', type=str, default='comparison_summary.json',
                       help='Output JSON file')

    args = parser.parse_args()

    print("="*70)
    print("GAMEPLAY COMPARISON ANALYSIS")
    print("="*70)

    # Auto-detect results directory if not specified
    if args.results_dir is None:
        # Look for directories like pong_openrouter_results/ or comparison_results/
        possible_dirs = glob.glob('./comparison_results/*_*_results')
        if possible_dirs:
            args.results_dir = possible_dirs[0]
            print(f"\n📁 Auto-detected results directory: {args.results_dir}")
        else:
            args.results_dir = './comparison_results'
            print(f"\n📁 Using default directory: {args.results_dir}")

    print()

    # Load results
    vision_only = load_results(args.results_dir, 'vision_only_*')
    vision_symbol = load_results(args.results_dir, 'vision_symbol_*')

    # Print summaries
    print_summary(vision_only, "Vision-Only")
    print_summary(vision_symbol, "Vision+Symbol")

    # Compare
    compare_pipelines(vision_only, vision_symbol)

    # Save summary
    save_summary(vision_only, vision_symbol, args.output)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
