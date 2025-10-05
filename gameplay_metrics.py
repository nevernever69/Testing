"""
Gameplay Metrics Tracker - Real performance measurement for Atari games.

Tracks:
- Score/reward over time
- Survival time (frames before game over)
- Balls caught (for Breakout/Pong)
- Actions taken
- Response times
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from scipy import stats


class GameplayMetrics:
    """Track gameplay performance metrics."""

    def __init__(self, game_name: str, pipeline_name: str, provider: str = None,
                 model_id: str = None, seed: int = None):
        self.game_name = game_name
        self.pipeline_name = pipeline_name
        self.provider = provider  # ✅ ADD THIS
        self.model_id = model_id  # ✅ ADD THIS
        self.seed = seed
        self.start_time = datetime.now()
        
        # Core metrics
        self.frames = []
        self.rewards = []
        self.cumulative_rewards = []
        self.actions = []
        self.lives_history = []
        self.response_times = []
        
        # Game-specific metrics
        self.ball_positions = []  # For Breakout/Pong
        self.paddle_positions = []
        self.catch_events = []  # When paddle successfully caught ball
        self.miss_events = []   # When ball was missed
        
    def record_step(self, frame_num: int, reward: float, action: int, 
                   lives: int, response_time: float = 0.0,
                   ball_pos: tuple = None, paddle_pos: tuple = None):
        """Record metrics for one gameplay step."""
        self.frames.append(frame_num)
        self.rewards.append(reward)
        self.cumulative_rewards.append(sum(self.rewards))
        self.actions.append(action)
        self.lives_history.append(lives)
        self.response_times.append(response_time)
        
        if ball_pos is not None:
            self.ball_positions.append(ball_pos)
        if paddle_pos is not None:
            self.paddle_positions.append(paddle_pos)
            
        # Detect catch/miss events
        if len(self.lives_history) > 1:
            if self.lives_history[-1] < self.lives_history[-2]:
                self.miss_events.append(frame_num)
            elif reward > 0 and self.game_name.lower() in ['pong', 'breakout']:
                self.catch_events.append(frame_num)
    
    def compute_statistics(self) -> Dict:
        """Compute final statistics."""
        if not self.frames:
            return {}
        
        total_frames = len(self.frames)
        final_score = self.cumulative_rewards[-1] if self.cumulative_rewards else 0
        
        # Survival metrics
        initial_lives = self.lives_history[0] if self.lives_history else 0
        final_lives = self.lives_history[-1] if self.lives_history else 0
        lives_lost = initial_lives - final_lives
        
        # Ball catch rate (for Breakout/Pong)
        catch_rate = len(self.catch_events) / (len(self.catch_events) + len(self.miss_events)) if (len(self.catch_events) + len(self.miss_events)) > 0 else 0
        
        # Action distribution
        action_counts = {}
        for action in self.actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Response time stats
        avg_response_time = np.mean(self.response_times) if self.response_times else 0
        
        stats = {
            'game': self.game_name,
            'pipeline': self.pipeline_name,
            'provider': self.provider,  # ✅ ADD THIS
            'model_id': self.model_id,  # ✅ ADD THIS
            'seed': self.seed,
            'timestamp': self.start_time.isoformat(),

            # Core metrics
            'total_frames': total_frames,
            'final_score': float(final_score),
            'final_reward': float(final_score),  # Alias
            'mean_reward_per_frame': float(np.mean(self.rewards)) if self.rewards else 0,
            
            # Survival metrics
            'initial_lives': initial_lives,
            'final_lives': final_lives,
            'lives_lost': lives_lost,
            'survived_full_episode': final_lives > 0,
            
            # Performance metrics
            'balls_caught': len(self.catch_events),
            'balls_missed': len(self.miss_events),
            'catch_rate': float(catch_rate),
            
            # Timing
            'avg_response_time_ms': float(avg_response_time * 1000),
            
            # Action distribution
            'action_distribution': action_counts,
            'action_entropy': self._compute_entropy(self.actions),
            
            # Reward progression
            'max_reward': float(max(self.cumulative_rewards)) if self.cumulative_rewards else 0,
            'reward_at_100': float(self.cumulative_rewards[99]) if len(self.cumulative_rewards) > 99 else None,
            'reward_at_300': float(self.cumulative_rewards[299]) if len(self.cumulative_rewards) > 299 else None,
            'reward_at_600': float(self.cumulative_rewards[-1]) if len(self.cumulative_rewards) >= 600 else None,
        }
        
        return stats
    
    def _compute_entropy(self, actions: List[int]) -> float:
        """Compute action entropy (measure of action diversity)."""
        if not actions:
            return 0.0
        unique, counts = np.unique(actions, return_counts=True)
        probs = counts / len(actions)
        return float(-np.sum(probs * np.log2(probs + 1e-10)))
    
    def save(self, output_dir: Path):
        """Save metrics to file."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stats = self.compute_statistics()
        
        # Save summary statistics
        summary_file = output_dir / f"gameplay_metrics_{self.pipeline_name.lower().replace(' ', '_').replace('+', '')}_{self.seed}.json"
        with open(summary_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Save detailed time series
        detailed_file = output_dir / f"gameplay_timeseries_{self.pipeline_name.lower().replace(' ', '_').replace('+', '')}_{self.seed}.json"
        with open(detailed_file, 'w') as f:
            json.dump({
                'game': self.game_name,
                'pipeline': self.pipeline_name,
                'seed': self.seed,
                'frames': self.frames,
                'rewards': [float(r) for r in self.rewards],
                'cumulative_rewards': [float(r) for r in self.cumulative_rewards],
                'actions': self.actions,
                'lives': self.lives_history,
                'catch_events': self.catch_events,
                'miss_events': self.miss_events
            }, f, indent=2)
        
        return summary_file


def aggregate_seeds(metrics_files: List[Path]) -> Dict:
    """Aggregate metrics across multiple seeds."""
    all_metrics = []
    for f in metrics_files:
        with open(f) as file:
            all_metrics.append(json.load(file))
    
    if not all_metrics:
        return {}
    
    # Aggregate statistics
    final_scores = [m['final_score'] for m in all_metrics]
    catch_rates = [m['catch_rate'] for m in all_metrics]
    total_frames = [m['total_frames'] for m in all_metrics]
    lives_lost = [m['lives_lost'] for m in all_metrics]
    
    aggregated = {
        'game': all_metrics[0]['game'],
        'pipeline': all_metrics[0]['pipeline'],
        'num_seeds': len(all_metrics),
        'seeds': [m['seed'] for m in all_metrics],
        
        # Score statistics
        'final_score': {
            'mean': float(np.mean(final_scores)),
            'std': float(np.std(final_scores)),
            'min': float(np.min(final_scores)),
            'max': float(np.max(final_scores)),
            'all_values': final_scores
        },
        
        # Catch rate statistics
        'catch_rate': {
            'mean': float(np.mean(catch_rates)),
            'std': float(np.std(catch_rates)),
            'min': float(np.min(catch_rates)),
            'max': float(np.max(catch_rates)),
            'all_values': catch_rates
        },
        
        # Survival statistics
        'survival_frames': {
            'mean': float(np.mean(total_frames)),
            'std': float(np.std(total_frames)),
            'all_values': total_frames
        },
        
        'lives_lost': {
            'mean': float(np.mean(lives_lost)),
            'std': float(np.std(lives_lost)),
            'all_values': lives_lost
        },
        
        # Individual run details
        'individual_runs': all_metrics
    }
    
    return aggregated


def compare_pipelines(baseline_file: Path, comparison_file: Path) -> Dict:
    """Compare two pipelines with statistical tests."""
    with open(baseline_file) as f:
        baseline = json.load(f)
    with open(comparison_file) as f:
        comparison = json.load(f)
    
    # Extract scores
    baseline_scores = baseline['final_score']['all_values']
    comparison_scores = comparison['final_score']['all_values']
    
    # Perform t-test
    t_stat, p_value = stats.ttest_ind(baseline_scores, comparison_scores)
    
    # Cohen's d effect size
    pooled_std = np.sqrt((np.var(baseline_scores) + np.var(comparison_scores)) / 2)
    cohens_d = (np.mean(comparison_scores) - np.mean(baseline_scores)) / pooled_std if pooled_std > 0 else 0
    
    # Improvement metrics
    mean_improvement = np.mean(comparison_scores) - np.mean(baseline_scores)
    pct_improvement = (mean_improvement / abs(np.mean(baseline_scores)) * 100) if np.mean(baseline_scores) != 0 else float('inf')
    
    comparison_result = {
        'baseline': {
            'name': baseline['pipeline'],
            'mean_score': baseline['final_score']['mean'],
            'std': baseline['final_score']['std'],
            'num_seeds': baseline['num_seeds']
        },
        'comparison': {
            'name': comparison['pipeline'],
            'mean_score': comparison['final_score']['mean'],
            'std': comparison['final_score']['std'],
            'num_seeds': comparison['num_seeds']
        },
        'statistical_tests': {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant_at_0.05': p_value < 0.05,
            'significant_at_0.01': p_value < 0.01,
            'cohens_d': float(cohens_d),
            'effect_size_interpretation': _interpret_cohens_d(cohens_d)
        },
        'improvement': {
            'absolute': float(mean_improvement),
            'percentage': float(pct_improvement),
            'direction': 'better' if mean_improvement > 0 else 'worse'
        }
    }
    
    return comparison_result


def _interpret_cohens_d(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def print_comparison_report(comparison: Dict):
    """Pretty print comparison results."""
    print("\n" + "="*70)
    print("GAMEPLAY PERFORMANCE COMPARISON")
    print("="*70)
    
    baseline = comparison['baseline']
    comp = comparison['comparison']
    stats_tests = comparison['statistical_tests']
    improvement = comparison['improvement']
    
    print(f"\n{baseline['name']:25} → Score: {baseline['mean_score']:.2f} ± {baseline['std']:.2f} (n={baseline['num_seeds']})")
    print(f"{comp['name']:25} → Score: {comp['mean_score']:.2f} ± {comp['std']:.2f} (n={comp['num_seeds']})")
    
    print(f"\n{'Improvement:':25} {improvement['absolute']:+.2f} ({improvement['percentage']:+.1f}%) [{improvement['direction']}]")
    
    print(f"\n{'Statistical Significance:':25}")
    print(f"  {'t-statistic:':23} {stats_tests['t_statistic']:.3f}")
    print(f"  {'p-value:':23} {stats_tests['p_value']:.4f}")
    print(f"  {'Significant (p<0.05):':23} {'YES ✓' if stats_tests['significant_at_0.05'] else 'NO'}")
    print(f"  {'Significant (p<0.01):':23} {'YES ✓✓' if stats_tests['significant_at_0.01'] else 'NO'}")
    
    print(f"\n{'Effect Size:':25}")
    print(f"  {'Cohen\'s d:':23} {stats_tests['cohens_d']:.3f}")
    print(f"  {'Interpretation:':23} {stats_tests['effect_size_interpretation'].upper()}")
    
    print("\n" + "="*70)
    
    if stats_tests['significant_at_0.05'] and abs(stats_tests['cohens_d']) > 0.5:
        print("✓ RESULT: Statistically significant improvement with meaningful effect size!")
    elif stats_tests['significant_at_0.05']:
        print("⚠ RESULT: Statistically significant but small effect size")
    else:
        print("✗ RESULT: No statistically significant difference")
    
    print("="*70 + "\n")
