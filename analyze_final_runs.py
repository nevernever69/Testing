#!/usr/bin/env python3
"""
Analyze single-run comparison from final/ directory.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt


def load_single_run(csv_path):
    """Load a single run's CSV file."""
    df = pd.read_csv(csv_path)
    return {
        'final_score': df['cumulative_reward'].iloc[-1],
        'frames': len(df),
        'cumulative_rewards': df['cumulative_reward'].values,
        'actions': df['action'].values
    }


def plot_comparison(vision_only, vision_symbol, output_file='comparison_plot.png'):
    """Plot cumulative rewards over time."""
    plt.figure(figsize=(12, 6))

    frames_vo = np.arange(len(vision_only['cumulative_rewards']))
    frames_vs = np.arange(len(vision_symbol['cumulative_rewards']))

    plt.plot(frames_vo, vision_only['cumulative_rewards'],
             label='Vision-Only', linewidth=2, alpha=0.8)
    plt.plot(frames_vs, vision_symbol['cumulative_rewards'],
             label='Vision+Symbol', linewidth=2, alpha=0.8)

    plt.xlabel('Frame', fontsize=12)
    plt.ylabel('Cumulative Reward', fontsize=12)
    plt.title('Pong: Vision-Only vs Vision+Symbol (Bedrock)', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {output_file}")


def print_analysis(vision_only, vision_symbol):
    """Print detailed analysis."""
    print("="*70)
    print("PONG COMPARISON ANALYSIS - FINAL RUNS")
    print("="*70)

    print("\n📊 Vision-Only (Direct Frame):")
    print(f"  Final Score: {vision_only['final_score']:.2f}")
    print(f"  Total Frames: {vision_only['frames']}")
    print(f"  Avg Reward per Frame: {vision_only['final_score'] / vision_only['frames']:.4f}")

    print("\n📊 Vision+Symbol (Symbolic Detection):")
    print(f"  Final Score: {vision_symbol['final_score']:.2f}")
    print(f"  Total Frames: {vision_symbol['frames']}")
    print(f"  Avg Reward per Frame: {vision_symbol['final_score'] / vision_symbol['frames']:.4f}")

    # Comparison
    improvement = vision_symbol['final_score'] - vision_only['final_score']
    pct_improvement = (improvement / abs(vision_only['final_score'])) * 100 if vision_only['final_score'] != 0 else float('inf')

    print("\n" + "="*70)
    print("COMPARISON")
    print("="*70)
    print(f"\nAbsolute Improvement: {improvement:+.2f}")
    print(f"Relative Improvement: {pct_improvement:+.1f}%")

    if improvement > 0:
        print(f"\n🎉 Vision+Symbol is BETTER by {abs(improvement):.2f} points!")
        print(f"   ({abs(pct_improvement):.1f}% improvement)")
    elif improvement < 0:
        print(f"\n⚠️  Vision-Only is better by {abs(improvement):.2f} points")
    else:
        print(f"\n🤝 Both pipelines performed equally")

    # Action analysis
    print("\n" + "="*70)
    print("ACTION DISTRIBUTION")
    print("="*70)

    vo_actions = vision_only['actions']
    vs_actions = vision_symbol['actions']

    print("\nVision-Only:")
    for action in range(6):
        count = np.sum(vo_actions == action)
        pct = (count / len(vo_actions)) * 100
        print(f"  Action {action}: {count:4d} times ({pct:5.1f}%)")

    print("\nVision+Symbol:")
    for action in range(6):
        count = np.sum(vs_actions == action)
        pct = (count / len(vs_actions)) * 100
        print(f"  Action {action}: {count:4d} times ({pct:5.1f}%)")

    # Reward trajectory
    print("\n" + "="*70)
    print("REWARD TRAJECTORY")
    print("="*70)

    vo_rewards = vision_only['cumulative_rewards']
    vs_rewards = vision_symbol['cumulative_rewards']

    # Check key milestones
    milestones = [100, 200, 300, 400, 500]
    print("\nScore at Key Frames:")
    print(f"{'Frame':<10} {'Vision-Only':<15} {'Vision+Symbol':<15} {'Difference':<15}")
    print("-" * 60)
    for frame in milestones:
        if frame < len(vo_rewards) and frame < len(vs_rewards):
            vo_score = vo_rewards[frame]
            vs_score = vs_rewards[frame]
            diff = vs_score - vo_score
            print(f"{frame:<10} {vo_score:<15.2f} {vs_score:<15.2f} {diff:+.2f}")

    print(f"{'Final':<10} {vo_rewards[-1]:<15.2f} {vs_rewards[-1]:<15.2f} {vs_rewards[-1] - vo_rewards[-1]:+.2f}")


def save_summary(vision_only, vision_symbol, output_file='final_comparison_summary.json'):
    """Save summary to JSON."""
    improvement = vision_symbol['final_score'] - vision_only['final_score']
    pct_improvement = (improvement / abs(vision_only['final_score'])) * 100 if vision_only['final_score'] != 0 else 0

    summary = {
        'vision_only': {
            'final_score': float(vision_only['final_score']),
            'frames': int(vision_only['frames']),
            'avg_reward_per_frame': float(vision_only['final_score'] / vision_only['frames'])
        },
        'vision_symbol': {
            'final_score': float(vision_symbol['final_score']),
            'frames': int(vision_symbol['frames']),
            'avg_reward_per_frame': float(vision_symbol['final_score'] / vision_symbol['frames'])
        },
        'comparison': {
            'improvement_absolute': float(improvement),
            'improvement_percent': float(pct_improvement),
            'winner': 'vision_symbol' if improvement > 0 else 'vision_only' if improvement < 0 else 'tie'
        }
    }

    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Summary saved to: {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Analyze final Pong runs')
    parser.add_argument('--dir', type=str, default='./final',
                       help='Directory containing final runs')
    parser.add_argument('--output', type=str, default='final_comparison_summary.json',
                       help='Output JSON file')
    parser.add_argument('--plot', type=str, default='final_comparison_plot.png',
                       help='Output plot file')

    args = parser.parse_args()

    base_dir = Path(args.dir)

    # Find CSV files
    vo_csv = base_dir / 'Pong_bedrock_direct_frame' / 'actions_rewards.csv'
    vs_csv = base_dir / 'Pong_bedrock_symbolic_only' / 'actions_rewards.csv'

    if not vo_csv.exists():
        print(f"❌ Error: Vision-Only CSV not found at {vo_csv}")
        return

    if not vs_csv.exists():
        print(f"❌ Error: Vision+Symbol CSV not found at {vs_csv}")
        return

    # Load data
    print(f"Loading Vision-Only from: {vo_csv}")
    vision_only = load_single_run(vo_csv)

    print(f"Loading Vision+Symbol from: {vs_csv}")
    vision_symbol = load_single_run(vs_csv)

    print()

    # Analyze
    print_analysis(vision_only, vision_symbol)

    # Save summary
    save_summary(vision_only, vision_symbol, args.output)

    # Plot
    plot_comparison(vision_only, vision_symbol, args.plot)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
