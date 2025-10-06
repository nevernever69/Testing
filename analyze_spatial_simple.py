#!/usr/bin/env python3
"""
Simple Spatial Reasoning Benchmark Analyzer (no dependencies)
"""

import json
import csv
from pathlib import Path
from collections import defaultdict

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
            try:
                with open(response_file, 'r') as f:
                    responses.append(json.load(f))
            except Exception as e:
                print(f"Error loading {response_file}: {e}")

        return responses

    def load_actions_rewards(self):
        """Load actions and rewards CSV"""
        csv_path = self.results_dir / 'actions_rewards.csv'
        if not csv_path.exists():
            csv_path = self.results_dir / 'Results' / 'checkpoint_final' / 'actions_rewards.csv'

        if not csv_path.exists():
            print(f"Warning: Could not find actions_rewards.csv")
            return None

        data = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for step_num, row in enumerate(reader):
                data.append({
                    'step': step_num,
                    'action': int(row['action']),
                    'cumulative_reward': float(row['cumulative_reward'])
                })
        return data

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
        percentages = {k: (v / total_responses * 100) if total_responses > 0 else 0
                      for k, v in stats.items()}

        return stats, percentages

    def analyze_pong_spatial_reasoning(self, responses):
        """Specific analysis for Pong spatial reasoning"""
        analysis = {
            'total_frames': len(responses),
            'paddle_mentions': 0,
            'ball_mentions': 0,
            'correct_side_awareness': 0,
            'trajectory_predictions': 0,
            'noop_actions': 0,
            'up_actions': 0,
            'down_actions': 0,
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
                analysis['up_actions'] += 1
            elif action in [3, 5]:  # DOWN
                analysis['down_actions'] += 1

            # Detect spatial errors - only flag if explicitly claims to control wrong paddle
            # Check for phrases like "I am controlling the left paddle" or "my left paddle"
            if ('controlling' in reasoning and 'left paddle' in reasoning and
                not ('right paddle' in reasoning or 'green paddle' in reasoning.lower())):
                analysis['spatial_errors'].append(f"Step {resp['step']}: Confused about paddle control")
            elif 'my left paddle' in reasoning.lower() or 'i control the left' in reasoning.lower():
                analysis['spatial_errors'].append(f"Step {resp['step']}: Confused about paddle control")

        return analysis

    def detect_hits_and_misses(self, data, responses):
        """Detect hits and misses based on reward changes"""
        if data is None or len(data) < 2:
            return None

        hits = []
        misses = []

        for i in range(1, len(data)):
            reward_change = data[i]['cumulative_reward'] - data[i-1]['cumulative_reward']

            if reward_change > 0.1:  # Hit/score
                reasoning = responses[i].get('reasoning', '')[:100] if i < len(responses) else ''
                hits.append({
                    'step': i,
                    'reward': reward_change,
                    'reasoning': reasoning
                })
            elif reward_change < -0.1:  # Miss/death
                reasoning = responses[i].get('reasoning', '')[:100] if i < len(responses) else ''
                misses.append({
                    'step': i,
                    'reward': reward_change,
                    'reasoning': reasoning
                })

        hit_rate = len(hits) / (len(hits) + len(misses)) if (len(hits) + len(misses)) > 0 else 0

        return {
            'hits': hits,
            'misses': misses,
            'hit_rate': hit_rate,
        }

    def generate_report(self, output_file='spatial_reasoning_report.txt'):
        """Generate comprehensive spatial reasoning report"""
        print("Loading data...")
        responses = self.load_responses()
        data = self.load_actions_rewards()

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
            count = stats[key]
            report.append(f"  {key.replace('_', ' ').title()}: {pct:.1f}% ({count} frames)")
        report.append("")

        # 2. Game-Specific Analysis
        if self.game_type == 'pong':
            report.append("-" * 80)
            report.append("2. PONG-SPECIFIC SPATIAL ANALYSIS")
            report.append("-" * 80)
            pong_analysis = self.analyze_pong_spatial_reasoning(responses)
            total = pong_analysis['total_frames']

            report.append(f"  Paddle Awareness: {pong_analysis['paddle_mentions']}/{total} frames ({pong_analysis['paddle_mentions']/total*100:.1f}%)")
            report.append(f"  Ball Tracking: {pong_analysis['ball_mentions']}/{total} frames ({pong_analysis['ball_mentions']/total*100:.1f}%)")
            report.append(f"  Correct Side Awareness: {pong_analysis['correct_side_awareness']}/{total} frames ({pong_analysis['correct_side_awareness']/total*100:.1f}%)")
            report.append(f"  Trajectory Predictions: {pong_analysis['trajectory_predictions']}/{total} frames ({pong_analysis['trajectory_predictions']/total*100:.1f}%)")
            report.append("")
            report.append("  Action Distribution:")
            report.append(f"    NOOP (wait): {pong_analysis['noop_actions']} ({pong_analysis['noop_actions']/total*100:.1f}%)")
            report.append(f"    UP moves: {pong_analysis['up_actions']} ({pong_analysis['up_actions']/total*100:.1f}%)")
            report.append(f"    DOWN moves: {pong_analysis['down_actions']} ({pong_analysis['down_actions']/total*100:.1f}%)")

            if pong_analysis['spatial_errors']:
                report.append("")
                report.append("  Spatial Errors Detected:")
                for error in pong_analysis['spatial_errors']:
                    report.append(f"    - {error}")
            report.append("")

        # 3. Hits and Misses
        if data is not None:
            report.append("-" * 80)
            report.append("3. HITS, MISSES, AND PERFORMANCE")
            report.append("-" * 80)

            hits_misses = self.detect_hits_and_misses(data, responses)
            if hits_misses:
                report.append(f"  Total Hits (positive rewards): {len(hits_misses['hits'])}")
                report.append(f"  Total Misses (negative rewards): {len(hits_misses['misses'])}")
                report.append(f"  Hit Rate: {hits_misses['hit_rate']*100:.1f}%")
                report.append("")

                # Show examples
                if hits_misses['hits']:
                    report.append("  Sample Successful Hits:")
                    for hit in hits_misses['hits'][:3]:
                        report.append(f"    Step {hit['step']}: +{hit['reward']:.2f} reward")
                        report.append(f"      → {hit['reasoning']}...")
                    report.append("")

                if hits_misses['misses']:
                    report.append("  Sample Misses/Deaths:")
                    for miss in hits_misses['misses'][:3]:
                        report.append(f"    Step {miss['step']}: {miss['reward']:.2f} reward")
                        report.append(f"      → {miss['reasoning']}...")
                    report.append("")

            # Final Score
            report.append("-" * 80)
            report.append("4. FINAL PERFORMANCE")
            report.append("-" * 80)
            final_reward = data[-1]['cumulative_reward'] if data else 0
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
        print("Usage: python analyze_spatial_simple.py <results_directory>")
        print("\nExample:")
        print("  python analyze_spatial_simple.py ./comparison_results/vision_only_seed42/Pong_bedrock_direct_frame")
        sys.exit(1)

    results_dir = sys.argv[1]

    analyzer = SpatialReasoningAnalyzer(results_dir)
    analyzer.generate_report()


if __name__ == "__main__":
    main()
