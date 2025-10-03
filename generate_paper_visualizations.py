#!/usr/bin/env python3
"""
Generate comprehensive visualizations for research paper.
Creates publication-quality figures for all benchmark results.

Usage:
    python generate_paper_visualizations.py --results ./benchmark_results
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
from scipy import stats

# Set publication-quality defaults
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']


class PaperVisualizationGenerator:
    """Generate all visualizations needed for research paper."""

    def __init__(self, results_dir: str, output_dir: str = "./paper_figures"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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

    def generate_all_visualizations(self):
        """Generate all visualizations for paper."""
        print("=" * 80)
        print("GENERATING PAPER VISUALIZATIONS")
        print("=" * 80)
        print()

        # Main results
        print("1. Overall performance comparison...")
        self.plot_overall_comparison()

        print("2. Per-task performance breakdown...")
        self.plot_per_task_breakdown()

        print("3. Per-game performance breakdown...")
        self.plot_per_game_breakdown()

        print("4. Improvement heatmap...")
        self.plot_improvement_heatmap()

        print("5. Score distributions...")
        self.plot_score_distributions()

        print("6. Task difficulty analysis...")
        self.plot_task_difficulty()

        print("7. Statistical significance test...")
        self.generate_statistical_tests()

        print("8. Confusion matrix (identification task)...")
        self.plot_identification_confusion_matrix()

        print("9. Error analysis...")
        self.plot_error_analysis()

        print("10. Radar chart comparison...")
        self.plot_radar_comparison()

        print("11. Box plots by task...")
        self.plot_task_boxplots()

        print("12. Performance vs complexity...")
        self.plot_complexity_analysis()

        # Generate summary table
        print("13. Generating LaTeX table...")
        self.generate_latex_table()

        print()
        print("=" * 80)
        print(f"✅ All visualizations saved to: {self.output_dir}")
        print("=" * 80)

    def plot_overall_comparison(self):
        """Figure 1: Overall performance comparison (bar chart)."""
        if not self.comparison_results:
            return

        fig, ax = plt.subplots(figsize=(8, 6))

        pipelines = ['Vision-Only', 'Vision+Symbol']
        scores = [
            self.comparison_results['overall_comparison']['baseline_score'],
            self.comparison_results['overall_comparison']['comparison_score']
        ]

        colors = ['#3498db', '#2ecc71']
        bars = ax.bar(pipelines, scores, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=12)

        # Add improvement annotation
        improvement_points = self.comparison_results['overall_comparison']['improvement_points']
        ax.annotate(f'+{improvement_points:.1f} pts',
                   xy=(1, scores[1]), xytext=(0.5, scores[1] + 0.05),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2),
                   fontsize=14, color='red', fontweight='bold')

        ax.set_ylabel('Average Score', fontsize=14, fontweight='bold')
        ax.set_title('Overall Performance Comparison', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure1_overall_comparison.png', bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure1_overall_comparison.pdf', bbox_inches='tight')
        plt.close()

    def plot_per_task_breakdown(self):
        """Figure 2: Per-task performance breakdown (grouped bar chart)."""
        if not self.comparison_results:
            return

        fig, ax = plt.subplots(figsize=(12, 6))

        tasks = list(self.comparison_results['task_comparison'].keys())
        vision_only_scores = [self.comparison_results['task_comparison'][task]['baseline_score'] for task in tasks]
        vision_symbol_scores = [self.comparison_results['task_comparison'][task]['comparison_score'] for task in tasks]

        x = np.arange(len(tasks))
        width = 0.35

        bars1 = ax.bar(x - width/2, vision_only_scores, width, label='Vision-Only',
                      color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, vision_symbol_scores, width, label='Vision+Symbol',
                      color='#2ecc71', alpha=0.8, edgecolor='black', linewidth=1.5)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=10)

        ax.set_xlabel('Task Type', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=14, fontweight='bold')
        ax.set_title('Performance by Task Type', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([t.capitalize() for t in tasks], fontsize=12)
        ax.set_ylim([0, 1.0])
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure2_per_task_breakdown.png', bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure2_per_task_breakdown.pdf', bbox_inches='tight')
        plt.close()

    def plot_per_game_breakdown(self):
        """Figure 3: Per-game performance breakdown."""
        if not self.comparison_results:
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        games = list(self.comparison_results['game_comparison'].keys())
        vision_only_scores = [self.comparison_results['game_comparison'][game]['baseline_score'] for game in games]
        vision_symbol_scores = [self.comparison_results['game_comparison'][game]['comparison_score'] for game in games]

        x = np.arange(len(games))
        width = 0.35

        bars1 = ax.bar(x - width/2, vision_only_scores, width, label='Vision-Only',
                      color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, vision_symbol_scores, width, label='Vision+Symbol',
                      color='#27ae60', alpha=0.8, edgecolor='black', linewidth=1.5)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=10)

        ax.set_xlabel('Game', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=14, fontweight='bold')
        ax.set_title('Performance by Game', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(games, fontsize=12)
        ax.set_ylim([0, 1.0])
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure3_per_game_breakdown.png', bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure3_per_game_breakdown.pdf', bbox_inches='tight')
        plt.close()

    def plot_improvement_heatmap(self):
        """Figure 4: Improvement heatmap (task × game)."""
        if not self.vision_only_results or not self.vision_symbol_results:
            return

        # Extract improvements for each task-game combination
        games = list(set([e['game'] for e in self.vision_only_results['evaluations']]))
        tasks = ['visual', 'spatial', 'strategy', 'identification']

        improvement_matrix = np.zeros((len(tasks), len(games)))

        for i, task in enumerate(tasks):
            for j, game in enumerate(games):
                vo_scores = []
                vs_scores = []

                for eval in self.vision_only_results['evaluations']:
                    if eval['game'] == game and task in eval['tasks']:
                        vo_scores.append(eval['tasks'][task]['score'])

                for eval in self.vision_symbol_results['evaluations']:
                    if eval['game'] == game and task in eval['tasks']:
                        vs_scores.append(eval['tasks'][task]['score'])

                if vo_scores and vs_scores:
                    vo_mean = np.mean(vo_scores)
                    vs_mean = np.mean(vs_scores)
                    improvement = ((vs_mean - vo_mean) / vo_mean * 100) if vo_mean > 0 else 0
                    improvement_matrix[i, j] = improvement

        fig, ax = plt.subplots(figsize=(10, 8))

        sns.heatmap(improvement_matrix, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   xticklabels=games, yticklabels=[t.capitalize() for t in tasks],
                   cbar_kws={'label': 'Improvement (%)'}, ax=ax,
                   linewidths=1, linecolor='black')

        ax.set_title('Performance Improvement: Vision+Symbol vs Vision-Only',
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Game', fontsize=14, fontweight='bold')
        ax.set_ylabel('Task', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure4_improvement_heatmap.png', bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure4_improvement_heatmap.pdf', bbox_inches='tight')
        plt.close()

    def plot_score_distributions(self):
        """Figure 5: Score distributions (violin plots)."""
        if not self.vision_only_results or not self.vision_symbol_results:
            return

        tasks = ['visual', 'spatial', 'strategy', 'identification']

        # Collect scores
        data = []
        for task in tasks:
            for eval in self.vision_only_results['evaluations']:
                if task in eval['tasks']:
                    data.append({
                        'Task': task.capitalize(),
                        'Pipeline': 'Vision-Only',
                        'Score': eval['tasks'][task]['score']
                    })

            for eval in self.vision_symbol_results['evaluations']:
                if task in eval['tasks']:
                    data.append({
                        'Task': task.capitalize(),
                        'Pipeline': 'Vision+Symbol',
                        'Score': eval['tasks'][task]['score']
                    })

        df = pd.DataFrame(data)

        fig, ax = plt.subplots(figsize=(14, 6))

        sns.violinplot(data=df, x='Task', y='Score', hue='Pipeline',
                      palette=['#3498db', '#2ecc71'], split=False, ax=ax)

        ax.set_xlabel('Task Type', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=14, fontweight='bold')
        ax.set_title('Score Distribution by Task and Pipeline', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylim([0, 1.0])
        ax.legend(fontsize=12, loc='lower right')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure5_score_distributions.png', bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure5_score_distributions.pdf', bbox_inches='tight')
        plt.close()

    def plot_task_difficulty(self):
        """Figure 6: Task difficulty analysis (show which tasks are hardest)."""
        if not self.vision_only_results:
            return

        tasks = ['visual', 'spatial', 'strategy', 'identification']
        task_scores = []

        for task in tasks:
            scores = []
            for eval in self.vision_only_results['evaluations']:
                if task in eval['tasks']:
                    scores.append(eval['tasks'][task]['score'])

            if scores:
                task_scores.append({
                    'task': task.capitalize(),
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores)
                })

        # Sort by difficulty (ascending mean score = harder)
        task_scores = sorted(task_scores, key=lambda x: x['mean'])

        fig, ax = plt.subplots(figsize=(10, 6))

        tasks_sorted = [t['task'] for t in task_scores]
        means = [t['mean'] for t in task_scores]
        stds = [t['std'] for t in task_scores]

        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(tasks_sorted)))

        bars = ax.barh(tasks_sorted, means, xerr=stds, color=colors,
                      alpha=0.8, edgecolor='black', linewidth=1.5,
                      error_kw={'linewidth': 2, 'ecolor': 'black'})

        # Add value labels
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(mean + std + 0.02, i, f'{mean:.3f}±{std:.3f}',
                   va='center', fontsize=11, fontweight='bold')

        ax.set_xlabel('Score', fontsize=14, fontweight='bold')
        ax.set_ylabel('Task Type', fontsize=14, fontweight='bold')
        ax.set_title('Task Difficulty Analysis (Vision-Only)', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlim([0, 1.0])
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure6_task_difficulty.png', bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure6_task_difficulty.pdf', bbox_inches='tight')
        plt.close()

    def generate_statistical_tests(self):
        """Figure 7: Statistical significance tests (paired t-tests)."""
        if not self.vision_only_results or not self.vision_symbol_results:
            return

        tasks = ['visual', 'spatial', 'strategy', 'identification']
        results = []

        for task in tasks:
            vo_scores = []
            vs_scores = []

            # Match frames
            for vo_eval in self.vision_only_results['evaluations']:
                frame_id = vo_eval['frame_id']
                vs_eval = next((e for e in self.vision_symbol_results['evaluations']
                              if e['frame_id'] == frame_id), None)

                if vs_eval and task in vo_eval['tasks'] and task in vs_eval['tasks']:
                    vo_scores.append(vo_eval['tasks'][task]['score'])
                    vs_scores.append(vs_eval['tasks'][task]['score'])

            if len(vo_scores) > 1:
                # Paired t-test
                t_stat, p_value = stats.ttest_rel(vs_scores, vo_scores)

                # Effect size (Cohen's d)
                diff = np.array(vs_scores) - np.array(vo_scores)
                cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0

                results.append({
                    'task': task.capitalize(),
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'n': len(vo_scores)
                })

        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 6))

        tasks_list = [r['task'] for r in results]
        p_values = [r['p_value'] for r in results]
        cohens_d = [r['cohens_d'] for r in results]

        x = np.arange(len(tasks_list))
        width = 0.35

        # Plot p-values
        ax2 = ax.twinx()
        bars1 = ax.bar(x - width/2, cohens_d, width, label="Cohen's d (Effect Size)",
                      color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)
        line = ax2.plot(x, p_values, 'ro-', linewidth=2, markersize=10,
                       label='p-value', markeredgecolor='black', markeredgewidth=1.5)

        # Add significance markers
        for i, p in enumerate(p_values):
            if p < 0.001:
                marker = '***'
            elif p < 0.01:
                marker = '**'
            elif p < 0.05:
                marker = '*'
            else:
                marker = 'n.s.'

            ax2.text(i, p + 0.05, marker, ha='center', fontsize=16, fontweight='bold')

        # Add significance line at p=0.05
        ax2.axhline(y=0.05, color='r', linestyle='--', linewidth=2, alpha=0.5, label='p=0.05')

        ax.set_xlabel('Task Type', fontsize=14, fontweight='bold')
        ax.set_ylabel("Cohen's d (Effect Size)", fontsize=14, fontweight='bold')
        ax2.set_ylabel('p-value', fontsize=14, fontweight='bold')
        ax.set_title('Statistical Significance of Improvements', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(tasks_list, fontsize=12)
        ax.grid(axis='y', alpha=0.3)

        # Combine legends
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, fontsize=11, loc='upper left')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure7_statistical_tests.png', bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure7_statistical_tests.pdf', bbox_inches='tight')
        plt.close()

        # Save detailed statistics to text file
        with open(self.output_dir / 'statistical_tests.txt', 'w') as f:
            f.write("STATISTICAL SIGNIFICANCE TESTS\n")
            f.write("=" * 70 + "\n\n")
            f.write("Paired t-tests comparing Vision+Symbol vs Vision-Only\n\n")

            for r in results:
                f.write(f"Task: {r['task']}\n")
                f.write(f"  n = {r['n']}\n")
                f.write(f"  t-statistic = {r['t_statistic']:.4f}\n")
                f.write(f"  p-value = {r['p_value']:.6f}")

                if r['p_value'] < 0.001:
                    f.write(" ***\n")
                elif r['p_value'] < 0.01:
                    f.write(" **\n")
                elif r['p_value'] < 0.05:
                    f.write(" *\n")
                else:
                    f.write(" (not significant)\n")

                f.write(f"  Cohen's d = {r['cohens_d']:.4f}")

                if abs(r['cohens_d']) < 0.2:
                    f.write(" (small effect)\n")
                elif abs(r['cohens_d']) < 0.5:
                    f.write(" (medium effect)\n")
                else:
                    f.write(" (large effect)\n")

                f.write("\n")

            f.write("\nSignificance levels: * p<0.05, ** p<0.01, *** p<0.001\n")

    def plot_identification_confusion_matrix(self):
        """Figure 8: Confusion matrix for identification task."""
        if not self.vision_only_results:
            return

        # Extract identification responses
        games = ['Pong', 'Breakout', 'SpaceInvaders']
        n_games = len(games)

        # Create confusion matrices for both pipelines
        for pipeline_name, results in [
            ('Vision-Only', self.vision_only_results),
            ('Vision+Symbol', self.vision_symbol_results)
        ]:
            if not results:
                continue

            confusion = np.zeros((n_games, n_games))

            for eval in results['evaluations']:
                true_game = eval['game']
                if 'identification' in eval['tasks']:
                    response = eval['tasks']['identification']['response'].lower()

                    # Simple game detection
                    predicted_game = None
                    if 'pong' in response:
                        predicted_game = 'Pong'
                    elif 'breakout' in response:
                        predicted_game = 'Breakout'
                    elif 'space' in response or 'invader' in response:
                        predicted_game = 'SpaceInvaders'

                    if true_game in games and predicted_game in games:
                        true_idx = games.index(true_game)
                        pred_idx = games.index(predicted_game)
                        confusion[true_idx, pred_idx] += 1

            # Normalize by row
            row_sums = confusion.sum(axis=1, keepdims=True)
            confusion_norm = np.divide(confusion, row_sums, where=row_sums!=0)

            # Plot
            fig, ax = plt.subplots(figsize=(8, 8))

            sns.heatmap(confusion_norm, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=games, yticklabels=games,
                       vmin=0, vmax=1, ax=ax, cbar_kws={'label': 'Proportion'},
                       linewidths=1, linecolor='black')

            ax.set_xlabel('Predicted Game', fontsize=14, fontweight='bold')
            ax.set_ylabel('True Game', fontsize=14, fontweight='bold')
            ax.set_title(f'Game Identification Accuracy - {pipeline_name}',
                        fontsize=16, fontweight='bold', pad=20)

            plt.tight_layout()
            filename = f"figure8_confusion_matrix_{pipeline_name.lower().replace('+', '_').replace(' ', '_')}.png"
            plt.savefig(self.output_dir / filename, bbox_inches='tight')
            plt.savefig(self.output_dir / filename.replace('.png', '.pdf'), bbox_inches='tight')
            plt.close()

    def plot_error_analysis(self):
        """Figure 9: Error analysis (show failure modes)."""
        if not self.vision_only_results:
            return

        # Categorize scores into error bins
        bins = [0, 0.3, 0.6, 0.9, 1.0]
        bin_labels = ['Failure\n(0.0-0.3)', 'Poor\n(0.3-0.6)', 'Good\n(0.6-0.9)', 'Excellent\n(0.9-1.0)']

        # Count errors for each pipeline and task
        tasks = ['visual', 'spatial', 'strategy', 'identification']
        pipelines = ['Vision-Only', 'Vision+Symbol']

        data = {task: {pipeline: [0]*len(bin_labels) for pipeline in pipelines} for task in tasks}

        for task in tasks:
            # Vision-Only
            for eval in self.vision_only_results['evaluations']:
                if task in eval['tasks']:
                    score = eval['tasks'][task]['score']
                    bin_idx = np.digitize([score], bins)[0] - 1
                    bin_idx = min(bin_idx, len(bin_labels) - 1)
                    data[task]['Vision-Only'][bin_idx] += 1

            # Vision+Symbol
            if self.vision_symbol_results:
                for eval in self.vision_symbol_results['evaluations']:
                    if task in eval['tasks']:
                        score = eval['tasks'][task]['score']
                        bin_idx = np.digitize([score], bins)[0] - 1
                        bin_idx = min(bin_idx, len(bin_labels) - 1)
                        data[task]['Vision+Symbol'][bin_idx] += 1

        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        colors = ['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71']

        for idx, task in enumerate(tasks):
            ax = axes[idx]

            x = np.arange(len(bin_labels))
            width = 0.35

            vo_counts = data[task]['Vision-Only']
            vs_counts = data[task].get('Vision+Symbol', [0]*len(bin_labels))

            bars1 = ax.bar(x - width/2, vo_counts, width, label='Vision-Only',
                          color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            bars2 = ax.bar(x + width/2, vs_counts, width, label='Vision+Symbol',
                          color=colors, alpha=1.0, edgecolor='black', linewidth=1.5)

            ax.set_xlabel('Score Range', fontsize=12, fontweight='bold')
            ax.set_ylabel('Count', fontsize=12, fontweight='bold')
            ax.set_title(f'{task.capitalize()} Task', fontsize=14, fontweight='bold', pad=15)
            ax.set_xticks(x)
            ax.set_xticklabels(bin_labels, fontsize=10)
            ax.legend(fontsize=10)
            ax.grid(axis='y', alpha=0.3)

        plt.suptitle('Error Analysis by Score Range', fontsize=18, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure9_error_analysis.png', bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure9_error_analysis.pdf', bbox_inches='tight')
        plt.close()

    def plot_radar_comparison(self):
        """Figure 10: Radar chart comparing pipelines across tasks."""
        if not self.comparison_results:
            return

        tasks = list(self.comparison_results['task_comparison'].keys())
        vo_scores = [self.comparison_results['task_comparison'][task]['baseline_score'] for task in tasks]
        vs_scores = [self.comparison_results['task_comparison'][task]['comparison_score'] for task in tasks]

        # Setup radar chart
        angles = np.linspace(0, 2 * np.pi, len(tasks), endpoint=False).tolist()
        vo_scores += vo_scores[:1]
        vs_scores += vs_scores[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        ax.plot(angles, vo_scores, 'o-', linewidth=3, label='Vision-Only',
               color='#3498db', markersize=10, markeredgecolor='black', markeredgewidth=2)
        ax.fill(angles, vo_scores, alpha=0.25, color='#3498db')

        ax.plot(angles, vs_scores, 'o-', linewidth=3, label='Vision+Symbol',
               color='#2ecc71', markersize=10, markeredgecolor='black', markeredgewidth=2)
        ax.fill(angles, vs_scores, alpha=0.25, color='#2ecc71')

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([t.capitalize() for t in tasks], fontsize=13, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.7)

        ax.set_title('Pipeline Comparison Across Tasks', fontsize=18, fontweight='bold', pad=30)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=13)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure10_radar_comparison.png', bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure10_radar_comparison.pdf', bbox_inches='tight')
        plt.close()

    def plot_task_boxplots(self):
        """Figure 11: Box plots showing score variance by task."""
        if not self.vision_only_results or not self.vision_symbol_results:
            return

        tasks = ['visual', 'spatial', 'strategy', 'identification']
        data = []

        for task in tasks:
            for eval in self.vision_only_results['evaluations']:
                if task in eval['tasks']:
                    data.append({
                        'Task': task.capitalize(),
                        'Pipeline': 'Vision-Only',
                        'Score': eval['tasks'][task]['score']
                    })

            for eval in self.vision_symbol_results['evaluations']:
                if task in eval['tasks']:
                    data.append({
                        'Task': task.capitalize(),
                        'Pipeline': 'Vision+Symbol',
                        'Score': eval['tasks'][task]['score']
                    })

        df = pd.DataFrame(data)

        fig, ax = plt.subplots(figsize=(14, 8))

        sns.boxplot(data=df, x='Task', y='Score', hue='Pipeline',
                   palette=['#3498db', '#2ecc71'], ax=ax,
                   linewidth=2, fliersize=6)

        ax.set_xlabel('Task Type', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score', fontsize=14, fontweight='bold')
        ax.set_title('Score Variance by Task Type', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylim([0, 1.0])
        ax.legend(fontsize=12, loc='lower right')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure11_task_boxplots.png', bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure11_task_boxplots.pdf', bbox_inches='tight')
        plt.close()

    def plot_complexity_analysis(self):
        """Figure 12: Performance vs complexity."""
        if not self.vision_only_results:
            return

        # Extract complexity data
        complexity_levels = ['easy', 'medium', 'hard']
        pipelines_data = {
            'Vision-Only': self.vision_only_results,
            'Vision+Symbol': self.vision_symbol_results
        }

        fig, ax = plt.subplots(figsize=(10, 6))

        for pipeline_name, results in pipelines_data.items():
            if not results:
                continue

            complexity_scores = {level: [] for level in complexity_levels}

            for eval in results['evaluations']:
                complexity = eval.get('complexity', {}).get('complexity_category', 'unknown')
                if complexity in complexity_levels:
                    for task_result in eval['tasks'].values():
                        complexity_scores[complexity].append(task_result['score'])

            means = [np.mean(complexity_scores[level]) if complexity_scores[level] else 0
                    for level in complexity_levels]
            stds = [np.std(complexity_scores[level]) if complexity_scores[level] else 0
                   for level in complexity_levels]

            color = '#3498db' if 'Only' in pipeline_name else '#2ecc71'
            ax.plot(complexity_levels, means, 'o-', linewidth=3, markersize=10,
                   label=pipeline_name, color=color,
                   markeredgecolor='black', markeredgewidth=2)
            ax.fill_between(complexity_levels,
                           [m-s for m, s in zip(means, stds)],
                           [m+s for m, s in zip(means, stds)],
                           alpha=0.2, color=color)

        ax.set_xlabel('Complexity Level', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average Score', fontsize=14, fontweight='bold')
        ax.set_title('Performance vs Complexity', fontsize=16, fontweight='bold', pad=20)
        ax.set_ylim([0, 1.0])
        ax.legend(fontsize=12, loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figure12_complexity_analysis.png', bbox_inches='tight')
        plt.savefig(self.output_dir / 'figure12_complexity_analysis.pdf', bbox_inches='tight')
        plt.close()

    def generate_latex_table(self):
        """Generate LaTeX table for paper."""
        if not self.comparison_results:
            return

        latex = []
        latex.append("% Table: Benchmark Results")
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append("\\caption{Performance comparison across tasks and pipelines}")
        latex.append("\\label{tab:results}")
        latex.append("\\begin{tabular}{lccc}")
        latex.append("\\toprule")
        latex.append("\\textbf{Task} & \\textbf{Vision-Only} & \\textbf{Vision+Symbol} & \\textbf{Improvement} \\\\")
        latex.append("\\midrule")

        for task, data in self.comparison_results['task_comparison'].items():
            vo = data['baseline_score']
            vs = data['comparison_score']
            imp = data['improvement_points']

            latex.append(f"{task.capitalize()} & {vo:.3f} & {vs:.3f} & +{imp:.1f} pts \\\\")

        latex.append("\\midrule")

        overall = self.comparison_results['overall_comparison']
        vo_overall = overall['baseline_score']
        vs_overall = overall['comparison_score']
        imp_overall = overall['improvement_points']

        latex.append(f"\\textbf{{Overall}} & \\textbf{{{vo_overall:.3f}}} & \\textbf{{{vs_overall:.3f}}} & \\textbf{{+{imp_overall:.1f} pts}} \\\\")

        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}")

        with open(self.output_dir / 'table_results.tex', 'w') as f:
            f.write('\n'.join(latex))

        print(f"LaTeX table saved to: {self.output_dir / 'table_results.tex'}")


def main():
    parser = argparse.ArgumentParser(description='Generate paper visualizations')
    parser.add_argument('--results', type=str, default='./benchmark_results',
                       help='Path to benchmark results directory')
    parser.add_argument('--output', type=str, default='./paper_figures',
                       help='Output directory for figures')

    args = parser.parse_args()

    generator = PaperVisualizationGenerator(args.results, args.output)
    generator.generate_all_visualizations()


if __name__ == '__main__':
    main()
