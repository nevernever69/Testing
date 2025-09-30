"""
Detailed Match Logger for Spatial Reasoning Benchmark
This module provides comprehensive logging of exactly what was matched, scored,
and compared during the benchmark evaluation process.
"""

import json
import os
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np

class DetailedMatchLogger:
    """
    Comprehensive logging system that captures:
    1. Exact keyword matches for each scoring component
    2. Ground truth vs predicted values
    3. Scoring breakdowns with reasoning
    4. Pattern matches and spatial vocabulary detection
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.log_dir = os.path.join(output_dir, "detailed_logs")
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize log structures
        self.match_logs = []
        self.scoring_logs = []
        self.comparison_logs = []

        # Load vocabulary for pattern matching
        self.spatial_vocabulary = {
            'coordinate_patterns': [
                r'\b\d+\s*,\s*\d+\b',  # "120, 80"
                r'\(?\d+\s*,\s*\d+\)?',  # "(120,80)" or "120,80"
                r'at\s+\d+',  # "at 120"
                r'position\s+\d+',  # "position 120"
            ],
            'spatial_terms': [
                'left', 'right', 'center', 'middle', 'top', 'bottom',
                'above', 'below', 'near', 'far', 'close', 'distance',
                'horizontal', 'vertical', 'diagonal', 'corner',
                'upper', 'lower', 'side', 'edge', 'boundary'
            ],
            'object_terms': [
                'paddle', 'ball', 'player', 'enemy', 'brick', 'block',
                'bullet', 'missile', 'ship', 'alien', 'invader'
            ]
        }

    def log_diagnostic_scoring(self,
                              response: str,
                              task_type: str,
                              scoring_result: Any,
                              ground_truth: Optional[Dict] = None,
                              frame_id: str = "",
                              pipeline_type: str = "") -> str:
        """
        Log detailed breakdown of diagnostic task scoring.
        Returns: unique log ID for this scoring event
        """
        log_id = f"{pipeline_type}_{task_type}_{frame_id}_{int(datetime.now().timestamp()*1000)}"

        log_entry = {
            'log_id': log_id,
            'timestamp': datetime.now().isoformat(),
            'frame_id': frame_id,
            'pipeline_type': pipeline_type,
            'task_type': task_type,
            'response': response,
            'ground_truth': ground_truth,
            'scoring_breakdown': {},
            'keyword_matches': {},
            'pattern_matches': {},
            'spatial_analysis': {},
            'final_score': scoring_result.score if hasattr(scoring_result, 'score') else 0.0,
            'confidence': scoring_result.confidence if hasattr(scoring_result, 'confidence') else 0.0,
            'issues': scoring_result.issues if hasattr(scoring_result, 'issues') else []
        }

        response_lower = response.lower()

        if task_type == 'identification':
            log_entry.update(self._log_identification_matches(response_lower, scoring_result))

        elif task_type == 'visual':
            log_entry.update(self._log_visual_matches(response_lower, scoring_result))

        elif task_type == 'spatial':
            log_entry.update(self._log_spatial_matches(response_lower, scoring_result, ground_truth))

        elif task_type == 'strategy':
            log_entry.update(self._log_strategy_matches(response_lower, scoring_result))

        # Save individual log
        log_file = os.path.join(self.log_dir, f"diagnostic_score_{log_id}.json")
        with open(log_file, 'w') as f:
            json.dump(log_entry, f, indent=2, default=str)

        self.scoring_logs.append(log_entry)
        return log_id

    def _log_identification_matches(self, response_lower: str, scoring_result: Any) -> Dict:
        """Log exact matches for game identification."""
        game_names = {
            'pong': ['pong', 'table tennis', 'ping pong'],
            'breakout': ['breakout', 'brick breaker', 'arkanoid'],
            'spaceinvaders': ['space invaders', 'invaders', 'space inv']
        }

        matches_found = {}
        for game, names in game_names.items():
            for name in names:
                if name in response_lower:
                    matches_found[name] = {
                        'position': response_lower.find(name),
                        'context': self._extract_context(response_lower, response_lower.find(name))
                    }

        return {
            'identification_matches': matches_found,
            'expected_names': game_names,
            'match_success': len(matches_found) > 0
        }

    def _log_visual_matches(self, response_lower: str, scoring_result: Any) -> Dict:
        """Log visual object detection matches."""
        object_keywords = ['paddle', 'ball', 'player', 'brick', 'block', 'enemy', 'bullet', 'alien']
        color_keywords = ['white', 'black', 'red', 'green', 'blue', 'yellow', 'orange']
        size_keywords = ['small', 'large', 'big', 'tiny', 'rectangular', 'square', 'round']

        object_matches = {}
        color_matches = {}
        size_matches = {}

        for keyword in object_keywords:
            if keyword in response_lower:
                object_matches[keyword] = {
                    'count': response_lower.count(keyword),
                    'positions': [m.start() for m in re.finditer(keyword, response_lower)],
                    'contexts': [self._extract_context(response_lower, pos) for pos in [m.start() for m in re.finditer(keyword, response_lower)]]
                }

        for color in color_keywords:
            if color in response_lower:
                color_matches[color] = {
                    'count': response_lower.count(color),
                    'contexts': [self._extract_context(response_lower, m.start()) for m in re.finditer(color, response_lower)]
                }

        for size in size_keywords:
            if size in response_lower:
                size_matches[size] = {
                    'count': response_lower.count(size),
                    'contexts': [self._extract_context(response_lower, m.start()) for m in re.finditer(size, response_lower)]
                }

        return {
            'visual_matches': {
                'objects': object_matches,
                'colors': color_matches,
                'sizes': size_matches
            },
            'object_detection_score': len(object_matches),
            'property_detection_score': len(color_matches) + len(size_matches)
        }

    def _log_spatial_matches(self, response_lower: str, scoring_result: Any, ground_truth: Optional[Dict]) -> Dict:
        """Log spatial reasoning matches and ground truth comparisons."""
        # Pattern matching for coordinates
        coordinate_matches = {}
        for i, pattern in enumerate(self.spatial_vocabulary['coordinate_patterns']):
            matches = re.findall(pattern, response_lower)
            if matches:
                coordinate_matches[f'pattern_{i}'] = {
                    'pattern': pattern,
                    'matches': matches,
                    'count': len(matches)
                }

        # Spatial term matching
        spatial_term_matches = {}
        for term in self.spatial_vocabulary['spatial_terms']:
            if term in response_lower:
                spatial_term_matches[term] = {
                    'count': response_lower.count(term),
                    'contexts': [self._extract_context(response_lower, m.start()) for m in re.finditer(term, response_lower)]
                }

        # Ground truth comparison (if available)
        ground_truth_comparison = {}
        if ground_truth and 'objects' in ground_truth:
            predicted_positions = self._extract_positions_from_response(response_lower)
            actual_positions = [(obj.get('x', 0), obj.get('y', 0)) for obj in ground_truth['objects']]

            ground_truth_comparison = {
                'predicted_positions': predicted_positions,
                'actual_positions': actual_positions,
                'position_errors': self._calculate_position_errors(predicted_positions, actual_positions),
                'accuracy_score': self._calculate_position_accuracy(predicted_positions, actual_positions)
            }

        return {
            'spatial_matches': {
                'coordinates': coordinate_matches,
                'spatial_terms': spatial_term_matches,
                'ground_truth_comparison': ground_truth_comparison
            }
        }

    def _log_strategy_matches(self, response_lower: str, scoring_result: Any) -> Dict:
        """Log strategy understanding matches."""
        strategy_keywords = [
            'intercept', 'block', 'avoid', 'hit', 'catch', 'defend', 'attack',
            'angle', 'direction', 'speed', 'timing', 'prediction', 'trajectory'
        ]

        strategy_matches = {}
        for keyword in strategy_keywords:
            if keyword in response_lower:
                strategy_matches[keyword] = {
                    'count': response_lower.count(keyword),
                    'contexts': [self._extract_context(response_lower, m.start()) for m in re.finditer(keyword, response_lower)]
                }

        return {
            'strategy_matches': strategy_matches,
            'strategy_complexity_score': len(strategy_matches)
        }

    def log_trajectory_prediction(self,
                                 response: str,
                                 predicted_positions: List[Tuple[float, float]],
                                 actual_positions: List[Tuple[float, float]],
                                 frame_id: str,
                                 pipeline_type: str) -> str:
        """Log detailed trajectory prediction analysis."""
        log_id = f"{pipeline_type}_trajectory_{frame_id}_{int(datetime.now().timestamp()*1000)}"

        # Calculate detailed error metrics
        l2_errors = []
        direction_errors = []

        for i, (pred, actual) in enumerate(zip(predicted_positions, actual_positions)):
            l2_error = np.sqrt((pred[0] - actual[0])**2 + (pred[1] - actual[1])**2)
            l2_errors.append(l2_error)

            # Direction error (if we have at least 2 points)
            if i > 0:
                pred_direction = np.arctan2(pred[1] - predicted_positions[i-1][1],
                                          pred[0] - predicted_positions[i-1][0])
                actual_direction = np.arctan2(actual[1] - actual_positions[i-1][1],
                                            actual[0] - actual_positions[i-1][0])
                direction_error = abs(pred_direction - actual_direction)
                direction_errors.append(direction_error)

        log_entry = {
            'log_id': log_id,
            'timestamp': datetime.now().isoformat(),
            'frame_id': frame_id,
            'pipeline_type': pipeline_type,
            'response': response,
            'predictions': {
                'raw_response': response,
                'extracted_positions': predicted_positions,
                'position_extraction_method': 'coordinate_pattern_matching'
            },
            'ground_truth': {
                'actual_positions': actual_positions
            },
            'error_analysis': {
                'l2_errors': l2_errors,
                'mean_l2_error': np.mean(l2_errors) if l2_errors else float('inf'),
                'max_l2_error': np.max(l2_errors) if l2_errors else float('inf'),
                'direction_errors': direction_errors,
                'mean_direction_error': np.mean(direction_errors) if direction_errors else float('inf')
            },
            'accuracy_metrics': {
                'positions_within_10px': sum(1 for e in l2_errors if e <= 10),
                'positions_within_20px': sum(1 for e in l2_errors if e <= 20),
                'positions_within_50px': sum(1 for e in l2_errors if e <= 50),
                'success_rate_10px': sum(1 for e in l2_errors if e <= 10) / len(l2_errors) if l2_errors else 0.0
            }
        }

        # Save individual log
        log_file = os.path.join(self.log_dir, f"trajectory_{log_id}.json")
        with open(log_file, 'w') as f:
            json.dump(log_entry, f, indent=2, default=str)

        self.match_logs.append(log_entry)
        return log_id

    def log_pipeline_comparison(self,
                               baseline_results: Dict,
                               comparison_results: Dict,
                               baseline_name: str = "Vision-only",
                               comparison_name: str = "Vision+Symbol") -> str:
        """Log detailed comparison between pipelines."""
        log_id = f"comparison_{int(datetime.now().timestamp()*1000)}"

        comparison_entry = {
            'log_id': log_id,
            'timestamp': datetime.now().isoformat(),
            'pipelines': {
                'baseline': baseline_name,
                'comparison': comparison_name
            },
            'detailed_comparison': {},
            'improvement_analysis': {},
            'statistical_significance': {}
        }

        # Compare each category in detail
        categories = ['visual', 'spatial', 'strategy', 'identification']
        for category in categories:
            if (category in baseline_results.get('overall_statistics', {}).get('category_breakdown', {}) and
                category in comparison_results.get('overall_statistics', {}).get('category_breakdown', {})):

                baseline_stats = baseline_results['overall_statistics']['category_breakdown'][category]
                comparison_stats = comparison_results['overall_statistics']['category_breakdown'][category]

                improvement = comparison_stats['mean_score'] - baseline_stats['mean_score']
                improvement_pct = (improvement / baseline_stats['mean_score'] * 100) if baseline_stats['mean_score'] > 0 else 0

                comparison_entry['detailed_comparison'][category] = {
                    'baseline': {
                        'mean_score': baseline_stats['mean_score'],
                        'std_score': baseline_stats.get('std_score', 0),
                        'sample_count': baseline_stats.get('sample_count', 0)
                    },
                    'comparison': {
                        'mean_score': comparison_stats['mean_score'],
                        'std_score': comparison_stats.get('std_score', 0),
                        'sample_count': comparison_stats.get('sample_count', 0)
                    },
                    'improvement': {
                        'absolute': improvement,
                        'percentage': improvement_pct,
                        'significant': abs(improvement) >= 0.05,
                        'interpretation': self._interpret_improvement(improvement, category)
                    }
                }

        # Save comparison log
        log_file = os.path.join(self.log_dir, f"comparison_{log_id}.json")
        with open(log_file, 'w') as f:
            json.dump(comparison_entry, f, indent=2, default=str)

        self.comparison_logs.append(comparison_entry)
        return log_id

    def generate_summary_report(self) -> str:
        """Generate comprehensive summary of all logged matches and scores."""
        report_file = os.path.join(self.log_dir, "DETAILED_MATCH_SUMMARY.txt")

        with open(report_file, 'w') as f:
            f.write("SPATIAL REASONING BENCHMARK - DETAILED MATCH ANALYSIS\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Total Scoring Logs: {len(self.scoring_logs)}\n")
            f.write(f"Total Match Logs: {len(self.match_logs)}\n")
            f.write(f"Total Comparison Logs: {len(self.comparison_logs)}\n\n")

            # Diagnostic scoring summary
            f.write("DIAGNOSTIC TASK SCORING SUMMARY\n")
            f.write("-" * 40 + "\n")

            for task_type in ['visual', 'spatial', 'strategy', 'identification']:
                task_logs = [log for log in self.scoring_logs if log['task_type'] == task_type]
                if task_logs:
                    f.write(f"\n{task_type.upper()} TASK:\n")
                    f.write(f"  Total evaluations: {len(task_logs)}\n")

                    # Pipeline breakdown
                    vision_only_logs = [log for log in task_logs if 'vision_only' in log['pipeline_type']]
                    vision_symbol_logs = [log for log in task_logs if 'vision_symbol' in log['pipeline_type']]

                    if vision_only_logs:
                        avg_score_vo = np.mean([log['final_score'] for log in vision_only_logs])
                        f.write(f"  Vision-only average score: {avg_score_vo:.3f}\n")

                    if vision_symbol_logs:
                        avg_score_vs = np.mean([log['final_score'] for log in vision_symbol_logs])
                        f.write(f"  Vision+Symbol average score: {avg_score_vs:.3f}\n")

                        if vision_only_logs:
                            improvement = avg_score_vs - avg_score_vo
                            f.write(f"  Improvement: {improvement:+.3f} ({improvement/avg_score_vo*100:+.1f}%)\n")

                    # Top keyword matches
                    self._write_keyword_summary(f, task_logs, task_type)

            # Trajectory prediction summary
            if self.match_logs:
                f.write("\n\nTRAJECTORY PREDICTION SUMMARY\n")
                f.write("-" * 40 + "\n")

                trajectory_logs = [log for log in self.match_logs if 'trajectory' in log['log_id']]
                if trajectory_logs:
                    mean_errors = [log['error_analysis']['mean_l2_error'] for log in trajectory_logs
                                 if log['error_analysis']['mean_l2_error'] != float('inf')]
                    if mean_errors:
                        f.write(f"  Mean L2 Error: {np.mean(mean_errors):.2f} pixels\n")
                        f.write(f"  Best L2 Error: {np.min(mean_errors):.2f} pixels\n")
                        f.write(f"  Worst L2 Error: {np.max(mean_errors):.2f} pixels\n")

                    success_rates = [log['accuracy_metrics']['success_rate_10px'] for log in trajectory_logs]
                    if success_rates:
                        f.write(f"  Average success rate (10px): {np.mean(success_rates):.1%}\n")

            f.write(f"\n\nDETAILED LOGS LOCATION: {self.log_dir}\n")
            f.write("Individual log files contain complete match breakdowns.\n")

        return report_file

    def _write_keyword_summary(self, f, task_logs: List[Dict], task_type: str):
        """Write keyword match summary for a task type."""
        if task_type == 'visual':
            all_object_matches = {}
            for log in task_logs:
                if 'visual_matches' in log and 'objects' in log['visual_matches']:
                    for obj, data in log['visual_matches']['objects'].items():
                        if obj not in all_object_matches:
                            all_object_matches[obj] = 0
                        all_object_matches[obj] += data['count']

            if all_object_matches:
                f.write("  Top object detections:\n")
                sorted_objects = sorted(all_object_matches.items(), key=lambda x: x[1], reverse=True)
                for obj, count in sorted_objects[:5]:
                    f.write(f"    {obj}: {count} mentions\n")

        elif task_type == 'spatial':
            all_spatial_terms = {}
            for log in task_logs:
                if 'spatial_matches' in log and 'spatial_terms' in log['spatial_matches']:
                    for term, data in log['spatial_matches']['spatial_terms'].items():
                        if term not in all_spatial_terms:
                            all_spatial_terms[term] = 0
                        all_spatial_terms[term] += data['count']

            if all_spatial_terms:
                f.write("  Top spatial terms:\n")
                sorted_terms = sorted(all_spatial_terms.items(), key=lambda x: x[1], reverse=True)
                for term, count in sorted_terms[:5]:
                    f.write(f"    {term}: {count} mentions\n")

    def _extract_context(self, text: str, position: int, window: int = 30) -> str:
        """Extract context around a match position."""
        start = max(0, position - window)
        end = min(len(text), position + window)
        return text[start:end].strip()

    def _extract_positions_from_response(self, response: str) -> List[Tuple[float, float]]:
        """Extract coordinate positions from response text."""
        positions = []
        for pattern in self.spatial_vocabulary['coordinate_patterns']:
            matches = re.findall(pattern, response)
            for match in matches:
                # Parse coordinate pairs
                coords = re.findall(r'\d+', match)
                if len(coords) >= 2:
                    positions.append((float(coords[0]), float(coords[1])))
        return positions

    def _calculate_position_errors(self, predicted: List[Tuple], actual: List[Tuple]) -> List[float]:
        """Calculate L2 errors between predicted and actual positions."""
        if not predicted or not actual:
            return []

        errors = []
        for pred, act in zip(predicted, actual):
            error = np.sqrt((pred[0] - act[0])**2 + (pred[1] - act[1])**2)
            errors.append(error)
        return errors

    def _calculate_position_accuracy(self, predicted: List[Tuple], actual: List[Tuple]) -> float:
        """Calculate position accuracy score."""
        errors = self._calculate_position_errors(predicted, actual)
        if not errors:
            return 0.0

        # Score based on errors within reasonable thresholds
        accurate_predictions = sum(1 for error in errors if error <= 20)  # 20 pixel threshold
        return accurate_predictions / len(errors)

    def _interpret_improvement(self, improvement: float, category: str) -> str:
        """Interpret the significance of score improvements."""
        if improvement >= 0.15:
            return f"Major improvement in {category}"
        elif improvement >= 0.05:
            return f"Moderate improvement in {category}"
        elif improvement >= 0.01:
            return f"Slight improvement in {category}"
        elif improvement >= -0.01:
            return f"No significant change in {category}"
        else:
            return f"Performance degradation in {category}"