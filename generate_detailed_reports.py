#!/usr/bin/env python3
"""
Generate detailed per-game reports and comprehensive HTML report.

Usage:
    python generate_detailed_reports.py --results ./benchmark_results
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime

# Set publication-quality defaults
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.3)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300


class DetailedReportGenerator:
    """Generate detailed per-game reports and comprehensive HTML report."""

    def __init__(self, results_dir: str, output_dir: str = "./detailed_reports"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / "per_game").mkdir(exist_ok=True)
        (self.output_dir / "per_evaluation").mkdir(exist_ok=True)

        # Load results
        self.vision_only_results = self._load_results("vision_only_results.json")
        self.vision_symbol_results = self._load_results("vision_symbol_results.json")
        self.comparison_results = self._load_results("comparison_results.json")

    def _load_results(self, filename: str) -> Dict:
        """Load results JSON file."""
        filepath = self.results_dir / filename
        if not filepath.exists():
            print(f"Warning: {filepath} not found")
            return {}

        with open(filepath, 'r') as f:
            return json.load(f)

    def generate_all_reports(self):
        """Generate all detailed reports."""
        print("=" * 80)
        print("GENERATING DETAILED REPORTS")
        print("=" * 80)
        print()

        # Per-game reports
        print("1. Generating per-game reports...")
        self.generate_per_game_reports()

        # Per-evaluation detailed reports
        print("2. Generating per-evaluation reports...")
        self.generate_per_evaluation_reports()

        # Comprehensive HTML report
        print("3. Generating comprehensive HTML report...")
        self.generate_html_report()

        # Generate CSV exports
        print("4. Exporting data to CSV...")
        self.export_to_csv()

        print()
        print("=" * 80)
        print(f"All reports generated in: {self.output_dir}")
        print("=" * 80)

    def generate_per_game_reports(self):
        """Generate detailed report for each game."""
        games = set()

        # Get all games
        for eval in self.vision_only_results.get('evaluations', []):
            games.add(eval['game'])

        for game in sorted(games):
            print(f"   - Processing {game}...")
            self._generate_single_game_report(game)

    def _generate_single_game_report(self, game: str):
        """Generate comprehensive report for a single game."""
        # Filter evaluations for this game
        vo_evals = [e for e in self.vision_only_results.get('evaluations', []) if e['game'] == game]
        vs_evals = [e for e in self.vision_symbol_results.get('evaluations', []) if e['game'] == game]

        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{game} - Detailed Performance Analysis', fontsize=20, fontweight='bold')

        # 1. Per-task comparison for this game
        ax = axes[0, 0]
        tasks = ['visual', 'spatial', 'strategy', 'identification']
        vo_scores = [np.mean([e['tasks'][t]['score'] for e in vo_evals]) for t in tasks]
        vs_scores = [np.mean([e['tasks'][t]['score'] for e in vs_evals]) for t in tasks]

        x = np.arange(len(tasks))
        width = 0.35

        ax.bar(x - width/2, vo_scores, width, label='Vision-Only', color='#FF6B6B')
        ax.bar(x + width/2, vs_scores, width, label='Vision+Symbol', color='#4ECDC4')

        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title(f'{game}: Task Performance', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([t.capitalize() for t in tasks], rotation=45)
        ax.legend()
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for i, (vo, vs) in enumerate(zip(vo_scores, vs_scores)):
            ax.text(i - width/2, vo, f'{vo:.2f}', ha='center', va='bottom', fontsize=9)
            ax.text(i + width/2, vs, f'{vs:.2f}', ha='center', va='bottom', fontsize=9)

        # 2. Score distribution for this game
        ax = axes[0, 1]
        all_vo_scores = [e['tasks'][t]['score'] for e in vo_evals for t in tasks]
        all_vs_scores = [e['tasks'][t]['score'] for e in vs_evals for t in tasks]

        ax.hist(all_vo_scores, bins=20, alpha=0.5, label='Vision-Only', color='#FF6B6B')
        ax.hist(all_vs_scores, bins=20, alpha=0.5, label='Vision+Symbol', color='#4ECDC4')
        ax.set_xlabel('Score', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title(f'{game}: Score Distribution', fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # 3. Frame-by-frame comparison
        ax = axes[1, 0]
        frame_ids = [e['frame_id'].split('_')[-1] for e in vo_evals[:15]]  # Show first 15
        vo_frame_scores = [np.mean([e['tasks'][t]['score'] for t in tasks]) for e in vo_evals[:15]]
        vs_frame_scores = [np.mean([e['tasks'][t]['score'] for t in tasks]) for e in vs_evals[:15]]

        x = np.arange(len(frame_ids))
        ax.plot(x, vo_frame_scores, 'o-', label='Vision-Only', color='#FF6B6B', linewidth=2)
        ax.plot(x, vs_frame_scores, 's-', label='Vision+Symbol', color='#4ECDC4', linewidth=2)
        ax.set_xlabel('Frame', fontweight='bold')
        ax.set_ylabel('Average Score', fontweight='bold')
        ax.set_title(f'{game}: Frame-by-Frame Performance', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(frame_ids, rotation=90, fontsize=8)
        ax.legend()
        ax.set_ylim([0, 1.0])
        ax.grid(alpha=0.3)

        # 4. Improvement breakdown
        ax = axes[1, 1]
        improvements = [vs_scores[i] - vo_scores[i] for i in range(len(tasks))]
        colors = ['green' if imp > 0 else 'red' for imp in improvements]

        ax.barh(tasks, improvements, color=colors, alpha=0.7)
        ax.set_xlabel('Improvement (points)', fontweight='bold')
        ax.set_title(f'{game}: Per-Task Improvement', fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (task, imp) in enumerate(zip(tasks, improvements)):
            ax.text(imp, i, f'{imp:+.2f}', ha='left' if imp > 0 else 'right', va='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(self.output_dir / "per_game" / f'{game.lower()}_detailed_report.png', bbox_inches='tight')
        plt.savefig(self.output_dir / "per_game" / f'{game.lower()}_detailed_report.pdf', bbox_inches='tight')
        plt.close()

        # Generate text summary
        self._generate_game_text_summary(game, vo_evals, vs_evals)

    def _generate_game_text_summary(self, game: str, vo_evals: List, vs_evals: List):
        """Generate text summary for a game."""
        summary = []
        summary.append(f"{'=' * 80}")
        summary.append(f"{game} - Detailed Summary")
        summary.append(f"{'=' * 80}\n")

        summary.append(f"Total Frames: {len(vo_evals)}\n")

        summary.append("Per-Task Performance:")
        summary.append("-" * 80)
        summary.append(f"{'Task':<15} {'Vision-Only':<15} {'Vision+Symbol':<15} {'Improvement':<15}")
        summary.append("-" * 80)

        tasks = ['visual', 'spatial', 'strategy', 'identification']
        for task in tasks:
            vo_score = np.mean([e['tasks'][task]['score'] for e in vo_evals])
            vs_score = np.mean([e['tasks'][task]['score'] for e in vs_evals])
            improvement = vs_score - vo_score

            summary.append(f"{task.capitalize():<15} {vo_score:<15.3f} {vs_score:<15.3f} {improvement:+.3f}")

        summary.append("-" * 80)
        vo_overall = np.mean([e['tasks'][t]['score'] for e in vo_evals for t in tasks])
        vs_overall = np.mean([e['tasks'][t]['score'] for e in vs_evals for t in tasks])
        overall_improvement = vs_overall - vo_overall
        summary.append(f"{'Overall':<15} {vo_overall:<15.3f} {vs_overall:<15.3f} {overall_improvement:+.3f}")

        summary.append("\n" + "=" * 80 + "\n")

        # Save to file
        with open(self.output_dir / "per_game" / f'{game.lower()}_summary.txt', 'w') as f:
            f.write('\n'.join(summary))

    def generate_per_evaluation_reports(self):
        """Generate detailed CSV report for every evaluation."""
        all_evaluations = []

        # Process Vision-Only
        for eval in self.vision_only_results.get('evaluations', []):
            for task, result in eval['tasks'].items():
                all_evaluations.append({
                    'pipeline': 'Vision-Only',
                    'game': eval['game'],
                    'frame_id': eval['frame_id'],
                    'task': task,
                    'score': result['score'],
                    'confidence': result['confidence'],
                    'reasoning': result['reasoning'],
                    'rule_based_score': result.get('rule_based_score', 'N/A'),
                    'semantic_score': result.get('semantic_score', 'N/A'),
                    'llm_judge_score': result.get('llm_judge_score', 'N/A'),
                    'combination_method': result.get('combination_method', 'N/A')
                })

        # Process Vision+Symbol
        for eval in self.vision_symbol_results.get('evaluations', []):
            for task, result in eval['tasks'].items():
                all_evaluations.append({
                    'pipeline': 'Vision+Symbol',
                    'game': eval['game'],
                    'frame_id': eval['frame_id'],
                    'task': task,
                    'score': result['score'],
                    'confidence': result['confidence'],
                    'reasoning': result['reasoning'],
                    'rule_based_score': result.get('rule_based_score', 'N/A'),
                    'semantic_score': result.get('semantic_score', 'N/A'),
                    'llm_judge_score': result.get('llm_judge_score', 'N/A'),
                    'combination_method': result.get('combination_method', 'N/A')
                })

        # Save to CSV
        df = pd.DataFrame(all_evaluations)
        df.to_csv(self.output_dir / 'all_evaluations_detailed.csv', index=False)
        print(f"   - Saved {len(all_evaluations)} evaluations to CSV")

    def export_to_csv(self):
        """Export all data to CSV files."""
        # Export comparison results
        if self.comparison_results:
            # Overall comparison
            overall_data = []
            overall = self.comparison_results['overall_comparison']
            overall_data.append({
                'metric': 'Overall',
                'vision_only': overall['baseline_score'],
                'vision_symbol': overall['comparison_score'],
                'improvement_points': overall['improvement_points']
            })

            # Per-task
            for task, data in self.comparison_results['task_comparison'].items():
                overall_data.append({
                    'metric': task.capitalize(),
                    'vision_only': data['baseline_score'],
                    'vision_symbol': data['comparison_score'],
                    'improvement_points': data['improvement_points']
                })

            df = pd.DataFrame(overall_data)
            df.to_csv(self.output_dir / 'comparison_summary.csv', index=False)

            # Per-game comparison
            game_data = []
            for game, data in self.comparison_results['game_comparison'].items():
                game_data.append({
                    'game': game,
                    'vision_only': data['baseline_score'],
                    'vision_symbol': data['comparison_score'],
                    'improvement_points': data['improvement_points']
                })

            df = pd.DataFrame(game_data)
            df.to_csv(self.output_dir / 'per_game_comparison.csv', index=False)

    def generate_html_report(self):
        """Generate comprehensive HTML report."""
        html = []
        html.append("<!DOCTYPE html>")
        html.append("<html>")
        html.append("<head>")
        html.append("<title>Atari VLM Benchmark - Detailed Report</title>")
        html.append("<style>")
        html.append("""
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
            h2 { color: #34495e; margin-top: 30px; border-bottom: 2px solid #95a5a6; padding-bottom: 5px; }
            h3 { color: #7f8c8d; margin-top: 20px; }
            table { border-collapse: collapse; width: 100%; margin: 20px 0; background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            th { background-color: #3498db; color: white; padding: 12px; text-align: left; }
            td { padding: 10px; border-bottom: 1px solid #ddd; }
            tr:hover { background-color: #f5f5f5; }
            .positive { color: green; font-weight: bold; }
            .negative { color: red; font-weight: bold; }
            .game-section { background-color: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
            .metric { display: inline-block; margin: 10px 20px; padding: 15px; background-color: #ecf0f1; border-radius: 5px; }
            .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
            .metric-label { font-size: 14px; color: #7f8c8d; }
            img { max-width: 100%; height: auto; margin: 20px 0; border: 1px solid #ddd; border-radius: 4px; }
        """)
        html.append("</style>")
        html.append("</head>")
        html.append("<body>")

        # Header
        html.append("<h1>Atari VLM Benchmark - Comprehensive Report</h1>")
        html.append(f"<p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>")

        # Overall summary
        html.append("<h2>Overall Performance</h2>")
        if self.comparison_results:
            overall = self.comparison_results['overall_comparison']
            html.append('<div class="game-section">')
            html.append('<div class="metric">')
            html.append(f'<div class="metric-label">Vision-Only</div>')
            html.append(f'<div class="metric-value">{overall["baseline_score"]:.3f}</div>')
            html.append('</div>')
            html.append('<div class="metric">')
            html.append(f'<div class="metric-label">Vision+Symbol</div>')
            html.append(f'<div class="metric-value">{overall["comparison_score"]:.3f}</div>')
            html.append('</div>')
            html.append('<div class="metric">')
            html.append(f'<div class="metric-label">Improvement</div>')
            imp_class = 'positive' if overall["improvement_points"] > 0 else 'negative'
            html.append(f'<div class="metric-value {imp_class}">{overall["improvement_points"]:+.1f} pts</div>')
            html.append('</div>')
            html.append('</div>')

        # Per-task breakdown
        html.append("<h2>Per-Task Performance</h2>")
        html.append('<table>')
        html.append('<tr><th>Task</th><th>Vision-Only</th><th>Vision+Symbol</th><th>Improvement</th></tr>')

        if self.comparison_results:
            for task, data in self.comparison_results['task_comparison'].items():
                imp_class = 'positive' if data['improvement_points'] > 0 else 'negative'
                html.append(f"<tr>")
                html.append(f"<td><strong>{task.capitalize()}</strong></td>")
                html.append(f"<td>{data['baseline_score']:.3f}</td>")
                html.append(f"<td>{data['comparison_score']:.3f}</td>")
                html.append(f"<td class='{imp_class}'>{data['improvement_points']:+.1f} pts</td>")
                html.append(f"</tr>")

        html.append('</table>')

        # Per-game breakdown
        html.append("<h2>Per-Game Performance</h2>")

        if self.comparison_results:
            for game, data in sorted(self.comparison_results['game_comparison'].items()):
                html.append(f'<div class="game-section">')
                html.append(f'<h3>{game}</h3>')

                # Metrics
                html.append('<div class="metric">')
                html.append(f'<div class="metric-label">Vision-Only</div>')
                html.append(f'<div class="metric-value">{data["baseline_score"]:.3f}</div>')
                html.append('</div>')
                html.append('<div class="metric">')
                html.append(f'<div class="metric-label">Vision+Symbol</div>')
                html.append(f'<div class="metric-value">{data["comparison_score"]:.3f}</div>')
                html.append('</div>')
                html.append('<div class="metric">')
                html.append(f'<div class="metric-label">Improvement</div>')
                imp_class = 'positive' if data["improvement_points"] > 0 else 'negative'
                html.append(f'<div class="metric-value {imp_class}">{data["improvement_points"]:+.1f} pts</div>')
                html.append('</div>')

                # Image
                img_path = f"per_game/{game.lower()}_detailed_report.png"
                if (self.output_dir / img_path).exists():
                    html.append(f'<img src="{img_path}" alt="{game} Report">')

                html.append('</div>')

        # Detailed evaluations table
        html.append("<h2>All Evaluations</h2>")
        html.append('<p><a href="all_evaluations_detailed.csv">Download CSV with all evaluations</a></p>')

        html.append("</body>")
        html.append("</html>")

        # Save HTML
        with open(self.output_dir / 'comprehensive_report.html', 'w') as f:
            f.write('\n'.join(html))

        print(f"   - HTML report saved to: {self.output_dir / 'comprehensive_report.html'}")


def main():
    parser = argparse.ArgumentParser(description='Generate detailed per-game reports')
    parser.add_argument('--results', type=str, default='./benchmark_results',
                       help='Directory containing benchmark results')
    parser.add_argument('--output', type=str, default='./detailed_reports',
                       help='Output directory for reports')

    args = parser.parse_args()

    generator = DetailedReportGenerator(args.results, args.output)
    generator.generate_all_reports()

    print()
    print("✅ All reports generated successfully!")
    print(f"📁 Check: {args.output}/")
    print(f"   - comprehensive_report.html (open in browser)")
    print(f"   - per_game/ (individual game reports)")
    print(f"   - all_evaluations_detailed.csv (all evaluations)")
    print()


if __name__ == "__main__":
    main()
