#!/usr/bin/env python3
"""
Spatial Reasoning Comparison Analyzer
Compares Vision-Only vs Vision+Symbol pipelines across multiple seeds
"""

import json
import csv
from pathlib import Path
from collections import defaultdict
import sys

class SpatialComparisonAnalyzer:
    def __init__(self, results_base_dir):
        self.results_base_dir = Path(results_base_dir)
        self.game_type = self._detect_game_type()

    def _detect_game_type(self):
        """Detect game type from directory name"""
        dir_str = str(self.results_base_dir).lower()
        if 'pong' in dir_str:
            return 'pong'
        elif 'breakout' in dir_str:
            return 'breakout'
        elif 'space' in dir_str or 'invaders' in dir_str:
            return 'space_invaders'
        return 'unknown'

    def find_all_seeds(self):
        """Find all seeds with both vision_only and vision_symbol results"""
        seeds = set()

        # Find vision_only seeds
        for dir_path in self.results_base_dir.glob('vision_only_seed*'):
            seed_num = dir_path.name.replace('vision_only_seed', '')
            seeds.add(seed_num)

        # Only keep seeds that have both pipelines
        valid_seeds = []
        for seed in sorted(seeds):
            vision_only_dir = self.results_base_dir / f'vision_only_seed{seed}'
            vision_symbol_dir = self.results_base_dir / f'vision_symbol_seed{seed}'

            if vision_only_dir.exists() and vision_symbol_dir.exists():
                valid_seeds.append(seed)

        return valid_seeds

    def find_results_dir(self, pipeline_dir):
        """Find the actual results directory within pipeline folder"""
        # Look for game-specific subdirectory
        subdirs = list(pipeline_dir.iterdir())
        if subdirs:
            game_dir = subdirs[0]  # Should be like "Pong_bedrock_direct_frame"
            results_dir = game_dir / 'Results'
            if results_dir.exists():
                return results_dir
            return game_dir
        return pipeline_dir

    def load_responses(self, results_dir):
        """Load all response JSON files"""
        response_dir = results_dir / 'responses'
        responses = []

        if not response_dir.exists():
            return responses

        for response_file in sorted(response_dir.glob('response_*.json')):
            try:
                with open(response_file, 'r') as f:
                    responses.append(json.load(f))
            except Exception as e:
                pass

        return responses

    def load_actions_rewards(self, pipeline_dir):
        """Load actions and rewards CSV"""
        # Try multiple possible locations
        possible_paths = [
            pipeline_dir / 'actions_rewards.csv',
            list(pipeline_dir.iterdir())[0] / 'actions_rewards.csv' if list(pipeline_dir.iterdir()) else None,
            list(pipeline_dir.iterdir())[0] / 'Results' / 'checkpoint_final' / 'actions_rewards.csv' if list(pipeline_dir.iterdir()) else None,
        ]

        for csv_path in possible_paths:
            if csv_path and csv_path.exists():
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

        if total_responses == 0:
            return stats, {}

        for response in responses:
            reasoning = response.get('reasoning', '').lower()

            for category, keywords in spatial_keywords.items():
                if any(kw in reasoning for kw in keywords):
                    stats[f'{category}_mentions'] += 1

        # Calculate percentages
        percentages = {k: (v / total_responses * 100) if total_responses > 0 else 0
                      for k, v in stats.items()}

        return stats, percentages

    def analyze_game_specific(self, responses):
        """Game-specific analysis"""
        analysis = {
            'total_frames': len(responses),
            'paddle_mentions': 0,
            'ball_mentions': 0,
            'correct_side_awareness': 0,
            'trajectory_predictions': 0,
            'noop_actions': 0,
            'up_actions': 0,
            'down_actions': 0,
        }

        if len(responses) == 0:
            return analysis

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

        return analysis

    def detect_hits_and_misses(self, data):
        """Detect hits and misses based on reward changes"""
        if data is None or len(data) < 2:
            return {'hits': 0, 'misses': 0, 'hit_rate': 0}

        hits = 0
        misses = 0

        for i in range(1, len(data)):
            reward_change = data[i]['cumulative_reward'] - data[i-1]['cumulative_reward']

            if reward_change > 0.1:  # Hit/score
                hits += 1
            elif reward_change < -0.1:  # Miss/death
                misses += 1

        hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0

        return {
            'hits': hits,
            'misses': misses,
            'hit_rate': hit_rate,
        }

    def analyze_single_pipeline(self, pipeline_dir, pipeline_name):
        """Analyze a single pipeline"""
        results_dir = self.find_results_dir(pipeline_dir)
        responses = self.load_responses(results_dir)
        data = self.load_actions_rewards(pipeline_dir)

        if not responses:
            return None

        # Spatial keyword analysis
        stats, percentages = self.analyze_spatial_keywords(responses)

        # Game-specific analysis
        game_analysis = self.analyze_game_specific(responses)

        # Hits and misses
        hits_misses = self.detect_hits_and_misses(data)

        # Final score
        final_score = data[-1]['cumulative_reward'] if data and len(data) > 0 else 0

        return {
            'pipeline': pipeline_name,
            'total_frames': len(responses),
            'spatial_keywords': percentages,
            'game_specific': game_analysis,
            'hits_misses': hits_misses,
            'final_score': final_score
        }

    def compare_seed(self, seed):
        """Compare both pipelines for a single seed"""
        vision_only_dir = self.results_base_dir / f'vision_only_seed{seed}'
        vision_symbol_dir = self.results_base_dir / f'vision_symbol_seed{seed}'

        vision_only_results = self.analyze_single_pipeline(vision_only_dir, 'Vision-Only')
        vision_symbol_results = self.analyze_single_pipeline(vision_symbol_dir, 'Vision+Symbol')

        return {
            'seed': seed,
            'vision_only': vision_only_results,
            'vision_symbol': vision_symbol_results
        }

    def generate_comparison_report(self, output_file='spatial_comparison_report.txt'):
        """Generate comprehensive comparison report across all seeds"""
        print(f"Analyzing {self.game_type.upper()} results in: {self.results_base_dir}")

        seeds = self.find_all_seeds()
        if not seeds:
            print("No valid seed pairs found!")
            return

        print(f"Found {len(seeds)} seeds with both pipelines: {seeds}")

        # Analyze all seeds
        all_results = []
        for seed in seeds:
            print(f"Analyzing seed {seed}...")
            result = self.compare_seed(seed)
            all_results.append(result)

        # Generate report
        report = []
        report.append("=" * 100)
        report.append("SPATIAL REASONING COMPARISON REPORT")
        report.append("=" * 100)
        report.append(f"Game: {self.game_type.upper()}")
        report.append(f"Results Directory: {self.results_base_dir}")
        report.append(f"Seeds Analyzed: {', '.join(seeds)}")
        report.append("")

        # Per-seed comparison
        for result in all_results:
            seed = result['seed']
            vo = result['vision_only']
            vs = result['vision_symbol']

            if not vo or not vs:
                report.append(f"⚠️  Seed {seed}: Incomplete data")
                report.append("")
                continue

            report.append("=" * 100)
            report.append(f"SEED {seed} COMPARISON")
            report.append("=" * 100)
            report.append("")

            # Performance comparison
            report.append("-" * 100)
            report.append("1. PERFORMANCE METRICS")
            report.append("-" * 100)
            report.append(f"{'Metric':<40} {'Vision-Only':<25} {'Vision+Symbol':<25}")
            report.append("-" * 100)

            report.append(f"{'Final Score':<40} {vo['final_score']:<25.2f} {vs['final_score']:<25.2f}")
            report.append(f"{'Total Hits':<40} {vo['hits_misses']['hits']:<25} {vs['hits_misses']['hits']:<25}")
            report.append(f"{'Total Misses':<40} {vo['hits_misses']['misses']:<25} {vs['hits_misses']['misses']:<25}")
            report.append(f"{'Hit Rate':<40} {vo['hits_misses']['hit_rate']*100:<25.1f}% {vs['hits_misses']['hit_rate']*100:<25.1f}%")

            # Winner
            if vo['final_score'] > vs['final_score']:
                report.append(f"\n🏆 Winner: Vision-Only (+{vo['final_score'] - vs['final_score']:.2f} points)")
            elif vs['final_score'] > vo['final_score']:
                report.append(f"\n🏆 Winner: Vision+Symbol (+{vs['final_score'] - vo['final_score']:.2f} points)")
            else:
                report.append(f"\n🤝 Tie: Both scored {vo['final_score']:.2f}")

            report.append("")

            # Spatial reasoning quality
            report.append("-" * 100)
            report.append("2. SPATIAL REASONING QUALITY")
            report.append("-" * 100)
            report.append(f"{'Keyword Category':<40} {'Vision-Only':<25} {'Vision+Symbol':<25}")
            report.append("-" * 100)

            for category in ['position_mentions', 'direction_mentions', 'distance_mentions', 'relative_mentions']:
                vo_pct = vo['spatial_keywords'].get(category, 0)
                vs_pct = vs['spatial_keywords'].get(category, 0)
                cat_name = category.replace('_mentions', '').title()
                report.append(f"{cat_name + ' Awareness':<40} {vo_pct:<25.1f}% {vs_pct:<25.1f}%")

            report.append("")

            # Game-specific metrics
            report.append("-" * 100)
            report.append("3. GAME-SPECIFIC ANALYSIS")
            report.append("-" * 100)
            report.append(f"{'Metric':<40} {'Vision-Only':<25} {'Vision+Symbol':<25}")
            report.append("-" * 100)

            vo_game = vo['game_specific']
            vs_game = vs['game_specific']
            total = vo_game['total_frames']

            report.append(f"{'Paddle Awareness':<40} {vo_game['paddle_mentions']/total*100:<25.1f}% {vs_game['paddle_mentions']/total*100:<25.1f}%")
            report.append(f"{'Ball Tracking':<40} {vo_game['ball_mentions']/total*100:<25.1f}% {vs_game['ball_mentions']/total*100:<25.1f}%")
            report.append(f"{'Correct Side Awareness':<40} {vo_game['correct_side_awareness']/total*100:<25.1f}% {vs_game['correct_side_awareness']/total*100:<25.1f}%")
            report.append(f"{'Trajectory Predictions':<40} {vo_game['trajectory_predictions']/total*100:<25.1f}% {vs_game['trajectory_predictions']/total*100:<25.1f}%")

            report.append("")

            # Action distribution
            report.append("-" * 100)
            report.append("4. ACTION DISTRIBUTION")
            report.append("-" * 100)
            report.append(f"{'Action Type':<40} {'Vision-Only':<25} {'Vision+Symbol':<25}")
            report.append("-" * 100)

            report.append(f"{'NOOP (wait)':<40} {vo_game['noop_actions']/total*100:<25.1f}% {vs_game['noop_actions']/total*100:<25.1f}%")
            report.append(f"{'UP moves':<40} {vo_game['up_actions']/total*100:<25.1f}% {vs_game['up_actions']/total*100:<25.1f}%")
            report.append(f"{'DOWN moves':<40} {vo_game['down_actions']/total*100:<25.1f}% {vs_game['down_actions']/total*100:<25.1f}%")

            report.append("")
            report.append("")

        # Overall summary across all seeds
        if len(all_results) > 1:
            report.append("=" * 100)
            report.append("OVERALL SUMMARY (ALL SEEDS)")
            report.append("=" * 100)
            report.append("")

            vo_wins = 0
            vs_wins = 0
            ties = 0
            vo_total_score = 0
            vs_total_score = 0
            vo_total_hits = 0
            vs_total_hits = 0
            vo_total_misses = 0
            vs_total_misses = 0

            for result in all_results:
                vo = result['vision_only']
                vs = result['vision_symbol']

                if not vo or not vs:
                    continue

                if vo['final_score'] > vs['final_score']:
                    vo_wins += 1
                elif vs['final_score'] > vo['final_score']:
                    vs_wins += 1
                else:
                    ties += 1

                vo_total_score += vo['final_score']
                vs_total_score += vs['final_score']
                vo_total_hits += vo['hits_misses']['hits']
                vs_total_hits += vs['hits_misses']['hits']
                vo_total_misses += vo['hits_misses']['misses']
                vs_total_misses += vs['hits_misses']['misses']

            report.append(f"Seeds Won by Vision-Only: {vo_wins}")
            report.append(f"Seeds Won by Vision+Symbol: {vs_wins}")
            report.append(f"Ties: {ties}")
            report.append("")
            report.append(f"Average Score - Vision-Only: {vo_total_score/len(all_results):.2f}")
            report.append(f"Average Score - Vision+Symbol: {vs_total_score/len(all_results):.2f}")
            report.append("")
            report.append(f"Total Hits - Vision-Only: {vo_total_hits}")
            report.append(f"Total Hits - Vision+Symbol: {vs_total_hits}")
            report.append(f"Total Misses - Vision-Only: {vo_total_misses}")
            report.append(f"Total Misses - Vision+Symbol: {vs_total_misses}")
            report.append("")

            # Overall winner
            if vo_wins > vs_wins:
                report.append(f"🏆 OVERALL WINNER: Vision-Only (won {vo_wins}/{len(all_results)} seeds)")
            elif vs_wins > vo_wins:
                report.append(f"🏆 OVERALL WINNER: Vision+Symbol (won {vs_wins}/{len(all_results)} seeds)")
            else:
                report.append(f"🤝 OVERALL TIE: Both won {vo_wins} seeds each")

        # Write report
        report_text = "\n".join(report)
        output_path = self.results_base_dir / output_file
        with open(output_path, 'w') as f:
            f.write(report_text)

        print("\n" + report_text)
        print(f"\n📊 Comparison report saved to: {output_path}")

        return report_text


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_spatial_reasoning.py <results_base_directory>")
        print("\nExample:")
        print("  python compare_spatial_reasoning.py ./comparison_results/pong_bedrock_results/")
        sys.exit(1)

    results_base_dir = sys.argv[1]

    analyzer = SpatialComparisonAnalyzer(results_base_dir)
    analyzer.generate_comparison_report()


if __name__ == "__main__":
    main()
