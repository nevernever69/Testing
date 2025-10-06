#!/usr/bin/env python3
"""
Spatial Reasoning Benchmark Analyzer

Analyzes VLM spatial reasoning capabilities by examining:
1. How often the model correctly identifies object positions
2. Movement prediction accuracy
3. Action appropriateness given spatial context
4. Dies/Misses/Hits statistics (for Space Invaders, Pong, etc.)
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
import re

class SpatialReasoningAnalyzer:
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.game_type = self._detect_game_type()

    def _detect_game_type(self):
        """Detect game type from directory name"""
        dir_str = str(self.results_dir).lower()
        if 'pong' in dir_str:
            return 'pong'
        elif 'breakout' in dir_str:
            return 'breakout'
        elif 'space' in dir_str or 'invaders' in dir_str:
            return 'space_invaders'
        return 'unknown'

    def load_responses(self):
        """Load all response JSON files"""
        response_dir = self.results_dir / 'Results' / 'responses'
        responses = []

        if not response_dir.exists():
            print(f"Warning: Response directory not found: {response_dir}")
            return responses

        for response_file in sorted(response_dir.glob('response_*.json')):
            with open(response_file, 'r') as f:
                responses.append(json.load(f))

        return responses

    def load_actions_rewards(self):
        """Load actions and rewards CSV"""
        csv_path = self.results_dir / 'actions_rewards.csv'
        if csv_path.exists():
            return pd.read_csv(csv_path)

        # Try Results directory
        csv_path = self.results_dir / 'Results' / 'checkpoint_final' / 'actions_rewards.csv'
        if csv_path.exists():
            return pd.read_csv(csv_path)

        print(f"Warning: Could not find actions_rewards.csv")
        return None

    def analyze_spatial_keywords(self, responses):
        """Analyze spatial reasoning keywords in responses"""
        spatial_keywords = {
            'position': ['position', 'located', 'placement', 'coordinates', 'at x=', 'at y='],
            'direction': ['moving', 'trajectory', 'heading', 'direction', 'towards', 'away'],
            'distance': ['close', 'far', 'near', 'distance', 'approaching', 'distant'],
            'relative': ['above', 'below', 'left', 'right', 'top', 'bottom', 'side', 'center'],
        }

        stats = defaultdict(int)
        total_responses = len(responses)

        for response in responses:
            reasoning = response.get('reasoning', '').lower()

            for category, keywords in spatial_keywords.items():
                if any(kw in reasoning for kw in keywords):
                    stats[f'{category}_mentions'] += 1

        # Calculate percentages
        percentages = {k: (v / total_responses * 100) for k, v in stats.items()}

        return stats, percentages

    def analyze_pong_spatial_reasoning(self, responses, df):
        """Specific analysis for Pong spatial reasoning"""
        analysis = {
            'total_frames': len(responses),
            'paddle_mentions': 0,
            'ball_mentions': 0,
            'correct_side_awareness': 0,  # Knows it controls right/green paddle
            'trajectory_predictions': 0,
            'defensive_actions': 0,
            'offensive_actions': 0,
            'noop_actions': 0,
            'spatial_errors': [],
        }

        for resp in responses:
            reasoning = resp.get('reasoning', '').lower()
            action = resp.get('action', 0)

            # Count mentions
            if 'paddle' in reasoning:
                analysis['paddle_mentions'] += 1
            if 'ball' in reasoning:
                analysis['ball_mentions'] += 1

            # Check side awareness (green paddle = right side)
            if any(word in reasoning for word in ['green', 'right', 'my side']):
                analysis['correct_side_awareness'] += 1

            # Check trajectory prediction
            if any(word in reasoning for word in ['trajectory', 'path', 'heading', 'moving towards']):
                analysis['trajectory_predictions'] += 1

            # Categorize actions
            if action == 0:
                analysis['noop_actions'] += 1
            elif action in [2, 4]:  # UP
                analysis['defensive_actions'] += 1
            elif action in [3, 5]:  # DOWN
                analysis['offensive_actions'] += 1

            # Detect spatial errors (e.g., mentions wrong side)
            if any(word in reasoning for word in ['left paddle', 'opponent paddle']) and 'control' in reasoning:
                analysis['spatial_errors'].append(f"Step {resp['step']}: Confused about which paddle to control")

        return analysis

    def analyze_reward_correlation(self, responses, df):
        """Analyze correlation between spatial reasoning quality and rewards"""
        if df is None:
            return None

        # Merge response data with reward data
        spatial_quality_scores = []
        rewards = []

        for resp in responses:
            step = resp.get('step', 0)
            reasoning = resp.get('reasoning', '').lower()

            # Calculate spatial quality score (0-10)
            score = 0

            # Position awareness (+2)
            if any(kw in reasoning for kw in ['position', 'located', 'coordinates']):
                score += 2

            # Direction awareness (+2)
            if any(kw in reasoning for kw in ['moving', 'trajectory', 'direction']):
                score += 2

            # Distance awareness (+2)
            if any(kw in reasoning for kw in ['close', 'far', 'approaching']):
                score += 2

            # Relative positioning (+2)
            if any(kw in reasoning for kw in ['above', 'below', 'left', 'right']):
                score += 2

            # Strategic thinking (+2)
            if any(kw in reasoning for kw in ['predict', 'anticipate', 'intercept']):
                score += 2

            spatial_quality_scores.append(score)

            # Get reward for this step
            if step < len(df):
                rewards.append(df.iloc[step]['cumulative_reward'])
            else:
                rewards.append(0)

        correlation = np.corrcoef(spatial_quality_scores, rewards)[0, 1]

        return {
            'spatial_quality_scores': spatial_quality_scores,
            'rewards': rewards,
            'correlation': correlation,
            'avg_quality': np.mean(spatial_quality_scores),
            'avg_reward': np.mean(rewards),
        }

    def detect_death_events(self, df):
        """Detect when the agent dies (reward suddenly drops)"""
        if df is None:
            return []

        deaths = []
        for i in range(1, len(df)):
            reward_drop = df.iloc[i]['cumulative_reward'] - df.iloc[i-1]['cumulative_reward']
            if reward_drop < -0.5:  # Significant negative reward
                deaths.append({
                    'step': i,
                    'reward_drop': reward_drop,
                })

        return deaths

    def detect_hits_and_misses(self, df, responses):
        """Detect hits and misses based on reward changes"""
        if df is None:
            return None

        hits = []
        misses = []

        for i in range(1, len(df)):
            reward_change = df.iloc[i]['cumulative_reward'] - df.iloc[i-1]['cumulative_reward']

            if reward_change > 0.1:  # Positive reward = hit/score
                hits.append({
                    'step': i,
                    'reward': reward_change,
                    'reasoning': responses[i].get('reasoning', '') if i < len(responses) else ''
                })
            elif reward_change < -0.1:  # Negative reward = miss/death
                misses.append({
                    'step': i,
                    'reward': reward_change,
                    'reasoning': responses[i].get('reasoning', '') if i < len(responses) else ''
                })

        return {
            'hits': hits,
            'misses': misses,
            'hit_rate': len(hits) / (len(hits) + len(misses)) if (len(hits) + len(misses)) > 0 else 0,
        }

    def generate_report(self, output_file='spatial_reasoning_report.txt'):
        """Generate comprehensive spatial reasoning report"""
        print("Loading data...")
        responses = self.load_responses()
        df = self.load_actions_rewards()

        if not responses:
            print("No responses found!")
            return

        print(f"Analyzing {len(responses)} responses...")

        report = []
        report.append("=" * 80)
        report.append("SPATIAL REASONING BENCHMARK REPORT")
        report.append("=" * 80)
        report.append(f"Game: {self.game_type.upper()}")
        report.append(f"Results Directory: {self.results_dir}")
        report.append(f"Total Frames Analyzed: {len(responses)}")
        report.append("")

        # 1. Spatial Keywords Analysis
        report.append("-" * 80)
        report.append("1. SPATIAL REASONING KEYWORD ANALYSIS")
        report.append("-" * 80)
        stats, percentages = self.analyze_spatial_keywords(responses)
        for key, pct in percentages.items():
            report.append(f"  {key.replace('_', ' ').title()}: {pct:.1f}% ({stats[key]} frames)")
        report.append("")

        # 2. Game-Specific Analysis
        if self.game_type == 'pong':
            report.append("-" * 80)
            report.append("2. PONG-SPECIFIC SPATIAL ANALYSIS")
            report.append("-" * 80)
            pong_analysis = self.analyze_pong_spatial_reasoning(responses, df)
            report.append(f"  Paddle Awareness: {pong_analysis['paddle_mentions']}/{pong_analysis['total_frames']} frames ({pong_analysis['paddle_mentions']/pong_analysis['total_frames']*100:.1f}%)")
            report.append(f"  Ball Tracking: {pong_analysis['ball_mentions']}/{pong_analysis['total_frames']} frames ({pong_analysis['ball_mentions']/pong_analysis['total_frames']*100:.1f}%)")
            report.append(f"  Correct Side Awareness: {pong_analysis['correct_side_awareness']}/{pong_analysis['total_frames']} frames ({pong_analysis['correct_side_awareness']/pong_analysis['total_frames']*100:.1f}%)")
            report.append(f"  Trajectory Predictions: {pong_analysis['trajectory_predictions']}/{pong_analysis['total_frames']} frames ({pong_analysis['trajectory_predictions']/pong_analysis['total_frames']*100:.1f}%)")
            report.append("")
            report.append("  Action Distribution:")
            report.append(f"    NOOP (wait): {pong_analysis['noop_actions']} ({pong_analysis['noop_actions']/pong_analysis['total_frames']*100:.1f}%)")
            report.append(f"    UP moves: {pong_analysis['defensive_actions']} ({pong_analysis['defensive_actions']/pong_analysis['total_frames']*100:.1f}%)")
            report.append(f"    DOWN moves: {pong_analysis['offensive_actions']} ({pong_analysis['offensive_actions']/pong_analysis['total_frames']*100:.1f}%)")

            if pong_analysis['spatial_errors']:
                report.append("")
                report.append("  Spatial Errors Detected:")
                for error in pong_analysis['spatial_errors']:
                    report.append(f"    - {error}")
            report.append("")

        # 3. Reward Correlation
        if df is not None:
            report.append("-" * 80)
            report.append("3. SPATIAL REASONING QUALITY vs PERFORMANCE")
            report.append("-" * 80)
            correlation_analysis = self.analyze_reward_correlation(responses, df)
            if correlation_analysis:
                report.append(f"  Average Spatial Quality Score: {correlation_analysis['avg_quality']:.2f}/10")
                report.append(f"  Average Cumulative Reward: {correlation_analysis['avg_reward']:.2f}")
                report.append(f"  Correlation (Quality vs Reward): {correlation_analysis['correlation']:.3f}")

                if correlation_analysis['correlation'] > 0.5:
                    report.append("  → Strong positive correlation: Better spatial reasoning = Better performance")
                elif correlation_analysis['correlation'] > 0.2:
                    report.append("  → Moderate positive correlation: Spatial reasoning helps performance")
                else:
                    report.append("  → Weak correlation: Spatial reasoning may not strongly impact performance")
            report.append("")

        # 4. Hits and Misses
        if df is not None:
            report.append("-" * 80)
            report.append("4. HITS, MISSES, AND DEATHS")
            report.append("-" * 80)

            hits_misses = self.detect_hits_and_misses(df, responses)
            if hits_misses:
                report.append(f"  Total Hits (positive rewards): {len(hits_misses['hits'])}")
                report.append(f"  Total Misses (negative rewards): {len(hits_misses['misses'])}")
                report.append(f"  Hit Rate: {hits_misses['hit_rate']*100:.1f}%")
                report.append("")

                # Show some examples
                if hits_misses['hits']:
                    report.append("  Sample Successful Hits:")
                    for hit in hits_misses['hits'][:3]:
                        report.append(f"    Step {hit['step']}: +{hit['reward']:.2f} reward")
                        report.append(f"      Reasoning: {hit['reasoning'][:100]}...")
                    report.append("")

                if hits_misses['misses']:
                    report.append("  Sample Misses/Deaths:")
                    for miss in hits_misses['misses'][:3]:
                        report.append(f"    Step {miss['step']}: {miss['reward']:.2f} reward")
                        report.append(f"      Reasoning: {miss['reasoning'][:100]}...")

            deaths = self.detect_death_events(df)
            if deaths:
                report.append("")
                report.append(f"  Total Death Events: {len(deaths)}")
                report.append("  Death Events:")
                for death in deaths[:5]:
                    report.append(f"    Step {death['step']}: {death['reward_drop']:.2f} reward drop")
            report.append("")

        # 5. Final Score
        if df is not None:
            report.append("-" * 80)
            report.append("5. FINAL PERFORMANCE")
            report.append("-" * 80)
            final_reward = df.iloc[-1]['cumulative_reward']
            report.append(f"  Final Cumulative Reward: {final_reward:.2f}")
            report.append("")

        # Write report
        report_text = "\n".join(report)
        output_path = self.results_dir / output_file
        with open(output_path, 'w') as f:
            f.write(report_text)

        print(report_text)
        print(f"\nReport saved to: {output_path}")

        return report_text


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyze_spatial_reasoning.py <results_directory>")
        print("\nExample:")
        print("  python analyze_spatial_reasoning.py ./comparison_results/vision_only_seed42/Pong_bedrock_direct_frame")
        print("  python analyze_spatial_reasoning.py ./comparison_results/vision_symbol_seed42/Pong_bedrock_symbolic_only")
        sys.exit(1)

    results_dir = sys.argv[1]

    analyzer = SpatialReasoningAnalyzer(results_dir)
    analyzer.generate_report()


if __name__ == "__main__":
    main()
