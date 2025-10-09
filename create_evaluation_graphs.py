#!/usr/bin/env python3
"""
Create comprehensive evaluation graphs and tables for paper.

Reads from evaluation_results/ directory and generates:
- Per-game comparison graphs (all models on each game)
- Per-model comparison graphs (all games for each model)
- Aggregate performance heatmaps
- Frame-by-frame metric plots
- LaTeX tables for paper

Usage:
    python create_evaluation_graphs.py
    python create_evaluation_graphs.py --games pong breakout space_invaders
    python create_evaluation_graphs.py --output-dir paper_figures/
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import defaultdict

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.figsize'] = (8, 6)


def load_all_results(results_base: Path = Path("evaluation_results")) -> List[Dict[str, Any]]:
    """Load all evaluation results from centralized index."""
    index_file = results_base / "all_results_index.json"
    if not index_file.exists():
        print(f"❌ No results index found at {index_file}")
        print(f"   Run evaluations first using eval_detection_only.sh")
        return []

    with open(index_file, 'r') as f:
        data = json.load(f)

    print(f"✅ Loaded {len(data)} evaluation results")
    return data


def load_detailed_frame_results(output_dir: Path) -> List[Dict[str, Any]]:
    """Load detailed per-frame results for trend analysis."""
    detailed_file = output_dir / "detailed_frame_results.json"
    if detailed_file.exists():
        with open(detailed_file, 'r') as f:
            return json.load(f)
    return []


def create_per_game_comparison(results: List[Dict], games: List[str], output_dir: Path):
    """Create comparison graphs for each game showing all models."""
    print("\n📊 Creating per-game comparison graphs...")

    for game in games:
        game_results = [r for r in results if r['game'] == game]
        if not game_results:
            print(f"  ⚠️  No results for {game}, skipping")
            continue

        # Group by model
        models = sorted(set(f"{r['provider']}/{r['model']}" for r in game_results))

        if len(models) == 0:
            continue

        # Create figure with 2 metrics only (F1 and IoU)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Detection Performance: {game.replace("_", " ").title()}', fontsize=14, fontweight='bold')

        metrics = [
            ('f1', 'F1 Score (Detection Accuracy)', ax1),
            ('iou', 'IoU (Coordinate Accuracy)', ax2)
        ]

        for metric_key, metric_name, ax in metrics:
            model_names = []
            values = []

            for model in models:
                model_results = [r for r in game_results if f"{r['provider']}/{r['model']}" == model]
                if model_results:
                    avg_value = np.mean([r[metric_key] for r in model_results])
                    model_names.append(model.split('/')[-1][:20])  # Truncate long names
                    values.append(avg_value)

            bars = ax.bar(range(len(model_names)), values, color=sns.color_palette("husl", len(model_names)))
            ax.set_ylabel(metric_name)
            ax.set_ylim(0, 1.0)
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels(model_names, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)

            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.3f}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        output_file = output_dir / f"per_game_{game}_comparison.png"
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: {output_file}")


def create_per_model_comparison(results: List[Dict], output_dir: Path):
    """Create comparison graphs for each model showing all games."""
    print("\n📊 Creating per-model comparison graphs...")

    # Group by provider/model
    model_groups = defaultdict(list)
    for r in results:
        key = f"{r['provider']}_{r['model'].replace('/', '_')}"
        model_groups[key].append(r)

    for model_key, model_results in model_groups.items():
        games = sorted(set(r['game'] for r in model_results))

        if len(games) < 2:
            continue

        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        model_name = model_results[0]['model']
        provider = model_results[0]['provider']
        fig.suptitle(f'Detection Performance: {provider}/{model_name}', fontsize=14, fontweight='bold')

        metrics = [
            ('f1', 'F1 Score', axes[0, 0]),
            ('important_f1', 'Important F1 Score', axes[0, 1]),
            ('iou', 'IoU (Coordinate Accuracy)', axes[1, 0]),
            ('center_distance', 'Center Distance (pixels)', axes[1, 1])
        ]

        for metric_key, metric_name, ax in metrics:
            game_names = []
            values = []

            for game in games:
                game_results = [r for r in model_results if r['game'] == game]
                if game_results:
                    avg_value = np.mean([r[metric_key] for r in game_results])
                    game_names.append(game.replace('_', ' ').title())
                    values.append(avg_value)

            bars = ax.bar(range(len(game_names)), values, color=sns.color_palette("husl", len(game_names)))
            ax.set_ylabel(metric_name)

            if metric_key != 'center_distance':
                ax.set_ylim(0, 1.0)

            ax.set_xticks(range(len(game_names)))
            ax.set_xticklabels(game_names, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)

            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                if metric_key == 'center_distance':
                    label = f'{val:.1f}px'
                else:
                    label = f'{val:.3f}'
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       label, ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        output_file = output_dir / f"per_model_{model_key}_comparison.png"
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: {output_file}")


def create_aggregate_heatmap(results: List[Dict], games: List[str], output_dir: Path):
    """Create heatmap showing F1 scores across all models and games."""
    print("\n📊 Creating aggregate performance heatmap...")

    # Get unique models and games
    models = sorted(set(f"{r['provider']}/{r['model']}" for r in results))
    games = sorted(set(r['game'] for r in results if r['game'] in games))

    if len(models) == 0 or len(games) == 0:
        print("  ⚠️  Not enough data for heatmap")
        return

    # Create matrices for different metrics
    for metric_key, metric_name in [('f1', 'F1 Score'), ('important_f1', 'Important F1')]:
        matrix = np.zeros((len(models), len(games)))
        matrix[:] = np.nan

        for i, model in enumerate(models):
            for j, game in enumerate(games):
                matching = [r for r in results
                           if f"{r['provider']}/{r['model']}" == model and r['game'] == game]
                if matching:
                    matrix[i, j] = np.mean([r[metric_key] for r in matching])

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, max(6, len(models) * 0.5)))

        sns.heatmap(matrix, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0, vmax=1,
                   xticklabels=[g.replace('_', ' ').title() for g in games],
                   yticklabels=[m.split('/')[-1][:30] for m in models],
                   cbar_kws={'label': metric_name},
                   ax=ax)

        ax.set_title(f'Model Performance Heatmap: {metric_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Game', fontsize=12)
        ax.set_ylabel('Model', fontsize=12)

        plt.tight_layout()
        output_file = output_dir / f"heatmap_{metric_key.replace('_', '-')}.png"
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: {output_file}")


def create_frame_by_frame_plots(results: List[Dict], output_dir: Path, results_base: Path):
    """Create plots showing metric trends over frames."""
    print("\n📊 Creating frame-by-frame trend plots...")

    for result in results:
        output_path = Path(result['output_dir'])
        detailed_frames = load_detailed_frame_results(output_path)

        if not detailed_frames or len(detailed_frames) < 5:
            continue

        frames = [f['frame'] for f in detailed_frames]
        precision = [f['precision'] for f in detailed_frames]
        recall = [f['recall'] for f in detailed_frames]
        f1 = [f['f1'] for f in detailed_frames]
        important_f1 = [f['important_f1'] for f in detailed_frames]
        iou = [f['iou'] for f in detailed_frames]

        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        model_name = f"{result['provider']}/{result['model']}"
        game_name = result['game'].replace('_', ' ').title()
        fig.suptitle(f'Frame-by-Frame Performance: {game_name}\n{model_name}',
                    fontsize=12, fontweight='bold')

        # Plot 1: P/R/F1
        ax1.plot(frames, precision, 'o-', label='Precision', linewidth=2, markersize=4)
        ax1.plot(frames, recall, 's-', label='Recall', linewidth=2, markersize=4)
        ax1.plot(frames, f1, '^-', label='F1 Score', linewidth=2, markersize=4)
        ax1.plot(frames, important_f1, 'd-', label='Important F1', linewidth=2, markersize=4)
        ax1.set_ylabel('Score', fontsize=11)
        ax1.set_ylim(0, 1.0)
        ax1.grid(alpha=0.3)
        ax1.legend(loc='best')
        ax1.set_title('Detection Metrics Over Time')

        # Plot 2: IoU
        ax2.plot(frames, iou, 'o-', color='purple', linewidth=2, markersize=4)
        ax2.set_xlabel('Frame', fontsize=11)
        ax2.set_ylabel('IoU', fontsize=11)
        ax2.set_ylim(0, 1.0)
        ax2.grid(alpha=0.3)
        ax2.set_title('Coordinate Accuracy (IoU) Over Time')

        plt.tight_layout()

        model_safe = result['model'].replace('/', '_').replace(':', '_')
        output_file = output_dir / f"frame_trends_{result['game']}_{result['provider']}_{model_safe}.png"
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        print(f"  ✅ Saved: {output_file}")


def create_latex_tables(results: List[Dict], games: List[str], output_dir: Path):
    """Generate LaTeX tables for paper."""
    print("\n📝 Creating LaTeX tables...")

    # Table 1: Main results table
    latex_output = []
    latex_output.append("% Main Results Table - Detection Performance")
    latex_output.append("\\begin{table}[h]")
    latex_output.append("\\centering")
    latex_output.append("\\caption{VLM Object Detection Performance on Atari Games}")
    latex_output.append("\\label{tab:detection-results}")
    latex_output.append("\\begin{tabular}{llcccc}")
    latex_output.append("\\toprule")
    latex_output.append("\\textbf{Model} & \\textbf{Game} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1} & \\textbf{IoU} \\\\")
    latex_output.append("\\midrule")

    # Group by model
    models = sorted(set(f"{r['provider']}/{r['model']}" for r in results))

    for model in models:
        model_results = [r for r in results if f"{r['provider']}/{r['model']}" == model]
        model_name = model.split('/')[-1].replace('_', '\\_')

        for game in games:
            game_results = [r for r in model_results if r['game'] == game]
            if game_results:
                r = game_results[0]
                game_name = game.replace('_', ' ').title()
                latex_output.append(
                    f"{model_name} & {game_name} & "
                    f"{r['precision']:.3f} & {r['recall']:.3f} & "
                    f"{r['f1']:.3f} & {r['iou']:.3f} \\\\"
                )

        latex_output.append("\\midrule")

    latex_output.append("\\bottomrule")
    latex_output.append("\\end{tabular}")
    latex_output.append("\\end{table}")

    # Save LaTeX table
    output_file = output_dir / "results_table.tex"
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex_output))
    print(f"  ✅ Saved: {output_file}")

    # Table 2: Summary statistics
    latex_output2 = []
    latex_output2.append("% Summary Statistics Table")
    latex_output2.append("\\begin{table}[h]")
    latex_output2.append("\\centering")
    latex_output2.append("\\caption{Summary Statistics Across All Games}")
    latex_output2.append("\\label{tab:summary-stats}")
    latex_output2.append("\\begin{tabular}{lcccc}")
    latex_output2.append("\\toprule")
    latex_output2.append("\\textbf{Model} & \\textbf{Avg F1} & \\textbf{Avg Imp. F1} & \\textbf{Avg IoU} & \\textbf{Frames} \\\\")
    latex_output2.append("\\midrule")

    for model in models:
        model_results = [r for r in results if f"{r['provider']}/{r['model']}" == model]
        model_name = model.split('/')[-1].replace('_', '\\_')

        avg_f1 = np.mean([r['f1'] for r in model_results])
        avg_imp_f1 = np.mean([r['important_f1'] for r in model_results])
        avg_iou = np.mean([r['iou'] for r in model_results])
        total_frames = sum([r['frames_evaluated'] for r in model_results])

        latex_output2.append(
            f"{model_name} & {avg_f1:.3f} & {avg_imp_f1:.3f} & "
            f"{avg_iou:.3f} & {total_frames} \\\\"
        )

    latex_output2.append("\\bottomrule")
    latex_output2.append("\\end{tabular}")
    latex_output2.append("\\end{table}")

    output_file2 = output_dir / "summary_stats_table.tex"
    with open(output_file2, 'w') as f:
        f.write('\n'.join(latex_output2))
    print(f"  ✅ Saved: {output_file2}")


def create_summary_report(results: List[Dict], output_dir: Path):
    """Create a markdown summary report."""
    print("\n📄 Creating summary report...")

    report = ["# VLM Detection Evaluation Summary\n"]
    report.append(f"**Total Evaluations:** {len(results)}\n")

    # Overall statistics
    report.append("## Overall Statistics\n")
    report.append(f"- **Average F1:** {np.mean([r['f1'] for r in results]):.3f}")
    report.append(f"- **Average Important F1:** {np.mean([r['important_f1'] for r in results]):.3f}")
    report.append(f"- **Average Precision:** {np.mean([r['precision'] for r in results]):.3f}")
    report.append(f"- **Average Recall:** {np.mean([r['recall'] for r in results]):.3f}")
    report.append(f"- **Average IoU:** {np.mean([r['iou'] for r in results]):.3f}")
    report.append(f"- **Average Center Distance:** {np.mean([r['center_distance'] for r in results]):.1f}px\n")

    # Per-game breakdown
    report.append("## Per-Game Performance\n")
    games = sorted(set(r['game'] for r in results))
    for game in games:
        game_results = [r for r in results if r['game'] == game]
        report.append(f"### {game.replace('_', ' ').title()}")
        report.append(f"- Models evaluated: {len(set(r['model'] for r in game_results))}")
        report.append(f"- Average F1: {np.mean([r['f1'] for r in game_results]):.3f}")
        report.append(f"- Average Important F1: {np.mean([r['important_f1'] for r in game_results]):.3f}")
        report.append(f"- Average IoU: {np.mean([r['iou'] for r in game_results]):.3f}\n")

    # Per-model breakdown
    report.append("## Per-Model Performance\n")
    models = sorted(set(f"{r['provider']}/{r['model']}" for r in results))
    for model in models:
        model_results = [r for r in results if f"{r['provider']}/{r['model']}" == model]
        report.append(f"### {model}")
        report.append(f"- Games evaluated: {len(set(r['game'] for r in model_results))}")
        report.append(f"- Average F1: {np.mean([r['f1'] for r in model_results]):.3f}")
        report.append(f"- Average Important F1: {np.mean([r['important_f1'] for r in model_results]):.3f}")
        report.append(f"- Average IoU: {np.mean([r['iou'] for r in model_results]):.3f}")
        report.append(f"- Total frames evaluated: {sum(r['frames_evaluated'] for r in model_results)}\n")

    # Best performances
    report.append("## Best Performances\n")
    best_f1 = max(results, key=lambda x: x['f1'])
    report.append(f"### Highest F1 Score: {best_f1['f1']:.3f}")
    report.append(f"- Model: {best_f1['provider']}/{best_f1['model']}")
    report.append(f"- Game: {best_f1['game']}\n")

    best_iou = max(results, key=lambda x: x['iou'])
    report.append(f"### Best Coordinate Accuracy (IoU): {best_iou['iou']:.3f}")
    report.append(f"- Model: {best_iou['provider']}/{best_iou['model']}")
    report.append(f"- Game: {best_iou['game']}\n")

    output_file = output_dir / "EVALUATION_SUMMARY.md"
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    print(f"  ✅ Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Create comprehensive evaluation graphs and tables"
    )
    parser.add_argument('--results-dir', type=str, default='evaluation_results',
                       help='Base directory containing evaluation results')
    parser.add_argument('--output-dir', type=str, default='evaluation_results/analysis',
                       help='Output directory for graphs and tables')
    parser.add_argument('--games', nargs='+',
                       default=['pong', 'breakout', 'space_invaders'],
                       help='Games to include in analysis')

    args = parser.parse_args()

    results_base = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("VLM Detection Evaluation - Graph & Table Generator")
    print("=" * 80)

    # Load all results
    results = load_all_results(results_base)
    if not results:
        print("\n❌ No results found. Run evaluations first!")
        return

    # Filter to requested games
    results = [r for r in results if r['game'] in args.games]

    print(f"\n📊 Analyzing {len(results)} evaluation results")
    print(f"   Games: {', '.join(args.games)}")
    unique_models = set(f"{r['provider']}/{r['model']}" for r in results)
    print(f"   Models: {len(unique_models)}")

    # Create all visualizations
    create_per_game_comparison(results, args.games, output_dir)
    create_per_model_comparison(results, output_dir)
    create_aggregate_heatmap(results, args.games, output_dir)
    create_frame_by_frame_plots(results, output_dir, results_base)
    create_latex_tables(results, args.games, output_dir)
    create_summary_report(results, output_dir)

    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"All outputs saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - Per-game comparison graphs")
    print("  - Per-model comparison graphs")
    print("  - Performance heatmaps")
    print("  - Frame-by-frame trend plots")
    print("  - LaTeX tables (results_table.tex, summary_stats_table.tex)")
    print("  - Summary report (EVALUATION_SUMMARY.md)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
