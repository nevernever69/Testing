"""
Trajectory Prediction Benchmark

This module implements comprehensive trajectory prediction evaluation with L2 distance
metrics and detailed error analysis. Tests VLM ability to predict object movement
patterns, which is crucial for spatial reasoning in Atari games.

Usage:
    benchmark = TrajectoryPredictionBenchmark()
    results = benchmark.evaluate_pipeline(pipeline_func, selected_frames)
    benchmark.generate_analysis_report(results, 'trajectory_results.json')
"""

import numpy as np
import json
import math
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass
import re
from collections import defaultdict

from ocatari_ground_truth import OCAtariGroundTruth, calculate_trajectory_error
from automated_scoring_rubric import AutomatedScoringRubric

# Import detailed logger if available
try:
    from detailed_match_logger import DetailedMatchLogger
    DETAILED_LOGGING_AVAILABLE = True
except ImportError:
    DetailedMatchLogger = None
    DETAILED_LOGGING_AVAILABLE = False


@dataclass
class TrajectoryPrediction:
    """Single trajectory prediction result."""
    frame_id: str
    object_id: int
    object_category: str
    predicted_positions: List[Tuple[int, int]]
    actual_positions: List[Tuple[int, int]]
    l2_error: float
    position_errors: List[float]  # Error per prediction step
    direction_accuracy: float
    prediction_horizon: int
    confidence: Optional[float] = None


@dataclass
class TrajectoryBenchmarkResult:
    """Complete trajectory benchmark results."""
    predictions: List[TrajectoryPrediction]
    aggregate_metrics: Dict[str, float]
    error_analysis: Dict[str, Any]
    game_breakdown: Dict[str, Dict[str, float]]
    horizon_analysis: Dict[int, Dict[str, float]]
    category_analysis: Dict[str, Dict[str, float]]


class TrajectoryPredictionBenchmark:
    """
    Comprehensive trajectory prediction benchmark for spatial reasoning evaluation.

    Tests VLM ability to predict future object positions using:
    - L2 distance error metrics
    - Direction accuracy analysis
    - Multi-horizon prediction evaluation
    - Game and object-specific analysis
    """

    def __init__(self, prediction_horizons: List[int] = None, enable_detailed_logging: bool = False, log_output_dir: str = None):
        """
        Initialize trajectory prediction benchmark.

        Args:
            prediction_horizons: List of prediction steps to evaluate [1, 3, 5]
            enable_detailed_logging: Enable detailed match logging
            log_output_dir: Directory for detailed log files
        """
        self.prediction_horizons = prediction_horizons or [1, 3, 5]
        self.scorer = AutomatedScoringRubric(enable_detailed_logging, log_output_dir)

        # Initialize detailed logger if requested
        self.enable_detailed_logging = enable_detailed_logging and DETAILED_LOGGING_AVAILABLE
        self.logger = None
        if self.enable_detailed_logging and log_output_dir:
            self.logger = DetailedMatchLogger(log_output_dir)

        # Trajectory prediction prompts for different horizons
        self.trajectory_prompts = {
            1: "Given this frame, predict where the {object} will be in the next frame. Provide coordinates as (x, y).",
            3: "Analyze the {object} in this frame sequence. Predict its position in the next 3 frames as: (x1, y1), (x2, y2), (x3, y3).",
            5: "Study the {object} movement pattern. Predict its trajectory for the next 5 frames: (x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5)."
        }

        # Error thresholds for scoring
        self.error_thresholds = {
            'excellent': 5.0,    # < 5 pixels = excellent
            'good': 10.0,        # < 10 pixels = good
            'acceptable': 20.0,  # < 20 pixels = acceptable
            'poor': 40.0         # >= 40 pixels = poor
        }

    def evaluate_pipeline(self,
                         pipeline_func: Callable[[np.ndarray, str], str],
                         test_frames: List[Dict[str, Any]],
                         max_predictions_per_frame: int = 2) -> TrajectoryBenchmarkResult:
        """
        Evaluate pipeline on trajectory prediction tasks.

        Args:
            pipeline_func: Function that takes (frame, prompt) and returns text response
            test_frames: List of frame data with ground truth
            max_predictions_per_frame: Maximum objects to test per frame

        Returns:
            Complete benchmark results with detailed analysis
        """
        print(f"Evaluating trajectory prediction on {len(test_frames)} frames...")

        all_predictions = []

        for frame_idx, frame_data in enumerate(test_frames):
            print(f"  Processing frame {frame_idx + 1}/{len(test_frames)}")

            frame_predictions = self._evaluate_frame(
                frame_data, pipeline_func, max_predictions_per_frame
            )
            all_predictions.extend(frame_predictions)

        # Aggregate and analyze results
        return self._analyze_results(all_predictions)

    def _evaluate_frame(self,
                       frame_data: Dict[str, Any],
                       pipeline_func: Callable[[np.ndarray, str], str],
                       max_predictions: int) -> List[TrajectoryPrediction]:
        """Evaluate trajectory prediction for a single frame."""
        frame = np.array(frame_data['frame'])
        objects = frame_data['objects']
        frame_id = frame_data.get('frame_id', 'unknown')
        game_name = frame_data.get('game', 'unknown')

        predictions = []

        # Select objects with interesting trajectories
        moving_objects = [
            (i, obj) for i, obj in enumerate(objects)
            if self._is_object_suitable_for_trajectory(obj)
        ]

        # Limit number of objects to test
        if len(moving_objects) > max_predictions:
            # Prefer objects with higher velocity
            moving_objects.sort(key=lambda x: abs(x[1]['velocity'][0]) + abs(x[1]['velocity'][1]), reverse=True)
            moving_objects = moving_objects[:max_predictions]

        for obj_id, obj in moving_objects:
            for horizon in self.prediction_horizons:
                prediction = self._predict_object_trajectory(
                    frame, obj, obj_id, horizon, pipeline_func, frame_id, game_name
                )
                if prediction:
                    predictions.append(prediction)

        return predictions

    def _is_object_suitable_for_trajectory(self, obj: Dict[str, Any]) -> bool:
        """Check if object is suitable for trajectory prediction."""
        velocity = obj.get('velocity', [0, 0])
        speed = math.sqrt(velocity[0]**2 + velocity[1]**2)

        # Only test objects that are moving
        if speed < 0.5:
            return False

        # Prefer certain object types
        category = obj.get('category', '').lower()
        preferred_categories = ['ball', 'bullet', 'alien', 'invader']

        return any(cat in category for cat in preferred_categories) or speed > 2.0

    def _predict_object_trajectory(self,
                                  frame: np.ndarray,
                                  obj: Dict[str, Any],
                                  obj_id: int,
                                  horizon: int,
                                  pipeline_func: Callable[[np.ndarray, str], str],
                                  frame_id: str,
                                  game_name: str) -> Optional[TrajectoryPrediction]:
        """Predict trajectory for a specific object."""
        category = obj.get('category', 'object')

        # Create trajectory prediction prompt
        prompt = self.trajectory_prompts[horizon].format(object=category.lower())

        try:
            # Get prediction from pipeline
            response = pipeline_func(frame, prompt)

            # Extract predicted coordinates
            predicted_coords = self._extract_coordinates_from_response(response)

            if len(predicted_coords) != horizon:
                return None  # Invalid prediction format

            # Generate ground truth trajectory using physics
            current_pos = tuple(obj['position'])
            velocity = tuple(obj['velocity'])
            actual_coords = self._generate_ground_truth_trajectory(current_pos, velocity, horizon)

            # Calculate metrics
            l2_error = self._calculate_l2_error(predicted_coords, actual_coords)
            position_errors = [
                math.sqrt((pred[0] - actual[0])**2 + (pred[1] - actual[1])**2)
                for pred, actual in zip(predicted_coords, actual_coords)
            ]
            direction_accuracy = self._calculate_direction_accuracy(predicted_coords, actual_coords)

            trajectory_result = TrajectoryPrediction(
                frame_id=frame_id,
                object_id=obj_id,
                object_category=category,
                predicted_positions=predicted_coords,
                actual_positions=actual_coords,
                l2_error=l2_error,
                position_errors=position_errors,
                direction_accuracy=direction_accuracy,
                prediction_horizon=horizon
            )

            # Log detailed trajectory prediction if enabled
            if self.enable_detailed_logging and self.logger:
                pipeline_type = getattr(pipeline_func, '_pipeline_type', 'unknown')
                self.logger.log_trajectory_prediction(
                    response=response,
                    predicted_positions=predicted_coords,
                    actual_positions=actual_coords,
                    frame_id=frame_id,
                    pipeline_type=pipeline_type
                )

            return trajectory_result

        except Exception as e:
            print(f"    Error predicting trajectory: {e}")
            return None

    def _extract_coordinates_from_response(self, response: str) -> List[Tuple[int, int]]:
        """Extract coordinate pairs from VLM response."""
        coords = []

        # Pattern for (x, y) coordinates
        coord_pattern = r'\(\s*(\d+)\s*,\s*(\d+)\s*\)'
        matches = re.findall(coord_pattern, response)

        for match in matches:
            try:
                x, y = int(match[0]), int(match[1])
                # Validate coordinates are within reasonable bounds
                if 0 <= x <= 160 and 0 <= y <= 210:  # Atari frame dimensions
                    coords.append((x, y))
            except ValueError:
                continue

        # Alternative patterns if parentheses format fails
        if not coords:
            # Pattern: x1=123, y1=456, x2=789, y2=012, etc.
            x_pattern = r'x\d*[=:]\s*(\d+)'
            y_pattern = r'y\d*[=:]\s*(\d+)'

            x_matches = [int(m) for m in re.findall(x_pattern, response.lower())]
            y_matches = [int(m) for m in re.findall(y_pattern, response.lower())]

            for x, y in zip(x_matches, y_matches):
                if 0 <= x <= 160 and 0 <= y <= 210:
                    coords.append((x, y))

        return coords

    def _generate_ground_truth_trajectory(self,
                                        start_pos: Tuple[int, int],
                                        velocity: Tuple[float, float],
                                        steps: int) -> List[Tuple[int, int]]:
        """Generate ground truth trajectory using simple physics."""
        trajectory = []
        x, y = start_pos
        dx, dy = velocity

        for step in range(1, steps + 1):
            # Simple linear motion (could be enhanced with bounce physics)
            new_x = x + dx * step
            new_y = y + dy * step

            # Clamp to screen boundaries
            new_x = max(0, min(160, int(round(new_x))))
            new_y = max(0, min(210, int(round(new_y))))

            trajectory.append((new_x, new_y))

        return trajectory

    def _calculate_l2_error(self, predicted: List[Tuple[int, int]], actual: List[Tuple[int, int]]) -> float:
        """Calculate average L2 distance error across all predictions."""
        if len(predicted) != len(actual) or not predicted:
            return float('inf')

        total_error = 0.0
        for pred, act in zip(predicted, actual):
            error = math.sqrt((pred[0] - act[0])**2 + (pred[1] - act[1])**2)
            total_error += error

        return total_error / len(predicted)

    def _calculate_direction_accuracy(self, predicted: List[Tuple[int, int]], actual: List[Tuple[int, int]]) -> float:
        """Calculate accuracy of predicted movement direction."""
        if len(predicted) < 2 or len(actual) < 2:
            return 0.0

        # Calculate predicted direction vector (from first to last position)
        pred_start, pred_end = predicted[0], predicted[-1]
        pred_direction = (pred_end[0] - pred_start[0], pred_end[1] - pred_start[1])

        # Calculate actual direction vector
        actual_start, actual_end = actual[0], actual[-1]
        actual_direction = (actual_end[0] - actual_start[0], actual_end[1] - actual_start[1])

        # Calculate cosine similarity
        pred_mag = math.sqrt(pred_direction[0]**2 + pred_direction[1]**2)
        actual_mag = math.sqrt(actual_direction[0]**2 + actual_direction[1]**2)

        if pred_mag == 0 or actual_mag == 0:
            return 0.0

        dot_product = pred_direction[0] * actual_direction[0] + pred_direction[1] * actual_direction[1]
        cosine_similarity = dot_product / (pred_mag * actual_mag)

        # Convert to accuracy (0.0 to 1.0)
        return max(0.0, cosine_similarity)

    def _analyze_results(self, predictions: List[TrajectoryPrediction]) -> TrajectoryBenchmarkResult:
        """Analyze trajectory prediction results and generate comprehensive metrics."""
        if not predictions:
            empty_metrics = {
                'total_predictions': 0,
                'valid_predictions': 0,
                'mean_l2_error': float('inf'),
                'median_l2_error': float('inf'),
                'std_l2_error': 0.0,
                'mean_direction_accuracy': 0.0,
                'accuracy_distribution': {},
                'success_rate': 0.0
            }
            return TrajectoryBenchmarkResult(
                predictions=[], aggregate_metrics=empty_metrics, error_analysis={},
                game_breakdown={}, horizon_analysis={}, category_analysis={}
            )

        # Aggregate metrics
        l2_errors = [p.l2_error for p in predictions if p.l2_error != float('inf')]
        direction_accuracies = [p.direction_accuracy for p in predictions]

        aggregate_metrics = {
            'total_predictions': len(predictions),
            'valid_predictions': len(l2_errors),
            'mean_l2_error': np.mean(l2_errors) if l2_errors else float('inf'),
            'median_l2_error': np.median(l2_errors) if l2_errors else float('inf'),
            'std_l2_error': np.std(l2_errors) if l2_errors else 0.0,
            'mean_direction_accuracy': np.mean(direction_accuracies),
            'accuracy_distribution': self._calculate_accuracy_distribution(l2_errors),
            'success_rate': self._calculate_success_rate(l2_errors)
        }

        # Error analysis
        error_analysis = {
            'error_categories': self._categorize_errors(l2_errors),
            'improvement_over_baseline': self._calculate_baseline_comparison(predictions),
            'failure_modes': self._analyze_failure_modes(predictions)
        }

        # Game-specific analysis
        game_breakdown = self._analyze_by_game(predictions)

        # Horizon-specific analysis
        horizon_analysis = self._analyze_by_horizon(predictions)

        # Object category analysis
        category_analysis = self._analyze_by_category(predictions)

        return TrajectoryBenchmarkResult(
            predictions=predictions,
            aggregate_metrics=aggregate_metrics,
            error_analysis=error_analysis,
            game_breakdown=game_breakdown,
            horizon_analysis=horizon_analysis,
            category_analysis=category_analysis
        )

    def _calculate_accuracy_distribution(self, errors: List[float]) -> Dict[str, float]:
        """Calculate distribution of prediction accuracy levels."""
        if not errors:
            return {}

        total = len(errors)
        distribution = {}

        for level, threshold in self.error_thresholds.items():
            if level == 'poor':
                count = sum(1 for e in errors if e >= threshold)
            else:
                count = sum(1 for e in errors if e < threshold)

            distribution[level] = count / total

        return distribution

    def _calculate_success_rate(self, errors: List[float]) -> float:
        """Calculate overall success rate (predictions under acceptable threshold)."""
        if not errors:
            return 0.0

        acceptable_threshold = self.error_thresholds['acceptable']
        successful = sum(1 for e in errors if e < acceptable_threshold)
        return successful / len(errors)

    def _categorize_errors(self, errors: List[float]) -> Dict[str, int]:
        """Categorize errors into different ranges."""
        if not errors:
            return {}

        categories = {
            'excellent': sum(1 for e in errors if e < self.error_thresholds['excellent']),
            'good': sum(1 for e in errors if self.error_thresholds['excellent'] <= e < self.error_thresholds['good']),
            'acceptable': sum(1 for e in errors if self.error_thresholds['good'] <= e < self.error_thresholds['acceptable']),
            'poor': sum(1 for e in errors if e >= self.error_thresholds['acceptable'])
        }

        return categories

    def _calculate_baseline_comparison(self, predictions: List[TrajectoryPrediction]) -> Dict[str, float]:
        """Compare against simple baseline predictions (constant velocity)."""
        # This would compare against a simple linear extrapolation baseline
        # For now, return placeholder values
        return {
            'baseline_mean_error': 25.0,  # Placeholder baseline error
            'improvement_percentage': 0.0  # Placeholder improvement
        }

    def _analyze_failure_modes(self, predictions: List[TrajectoryPrediction]) -> Dict[str, Any]:
        """Analyze common failure modes in trajectory predictions."""
        poor_predictions = [p for p in predictions if p.l2_error >= self.error_thresholds['acceptable']]

        failure_analysis = {
            'total_failures': len(poor_predictions),
            'failure_rate': len(poor_predictions) / len(predictions) if predictions else 0.0,
        }

        if poor_predictions:
            # Analyze failure patterns
            failure_analysis.update({
                'worst_error': max(p.l2_error for p in poor_predictions),
                'common_failure_categories': self._get_failure_categories(poor_predictions),
                'horizon_failure_distribution': self._get_horizon_failures(poor_predictions)
            })

        return failure_analysis

    def _get_failure_categories(self, failed_predictions: List[TrajectoryPrediction]) -> Dict[str, int]:
        """Get distribution of object categories in failed predictions."""
        categories = defaultdict(int)
        for pred in failed_predictions:
            categories[pred.object_category] += 1
        return dict(categories)

    def _get_horizon_failures(self, failed_predictions: List[TrajectoryPrediction]) -> Dict[int, int]:
        """Get distribution of prediction horizons in failed predictions."""
        horizons = defaultdict(int)
        for pred in failed_predictions:
            horizons[pred.prediction_horizon] += 1
        return dict(horizons)

    def _analyze_by_game(self, predictions: List[TrajectoryPrediction]) -> Dict[str, Dict[str, float]]:
        """Analyze results broken down by game."""
        games = defaultdict(list)
        for pred in predictions:
            # Extract game name from frame_id
            game = pred.frame_id.split('_')[0] if '_' in pred.frame_id else 'unknown'
            games[game].append(pred)

        game_analysis = {}
        for game, game_predictions in games.items():
            valid_errors = [p.l2_error for p in game_predictions if p.l2_error != float('inf')]

            if valid_errors:
                game_analysis[game] = {
                    'count': len(game_predictions),
                    'mean_error': np.mean(valid_errors),
                    'success_rate': sum(1 for e in valid_errors if e < self.error_thresholds['acceptable']) / len(valid_errors),
                    'mean_direction_accuracy': np.mean([p.direction_accuracy for p in game_predictions])
                }

        return game_analysis

    def _analyze_by_horizon(self, predictions: List[TrajectoryPrediction]) -> Dict[int, Dict[str, float]]:
        """Analyze results broken down by prediction horizon."""
        horizons = defaultdict(list)
        for pred in predictions:
            horizons[pred.prediction_horizon].append(pred)

        horizon_analysis = {}
        for horizon, horizon_predictions in horizons.items():
            valid_errors = [p.l2_error for p in horizon_predictions if p.l2_error != float('inf')]

            if valid_errors:
                horizon_analysis[horizon] = {
                    'count': len(horizon_predictions),
                    'mean_error': np.mean(valid_errors),
                    'success_rate': sum(1 for e in valid_errors if e < self.error_thresholds['acceptable']) / len(valid_errors),
                    'error_growth_rate': np.mean(valid_errors) / horizon  # Error per prediction step
                }

        return horizon_analysis

    def _analyze_by_category(self, predictions: List[TrajectoryPrediction]) -> Dict[str, Dict[str, float]]:
        """Analyze results broken down by object category."""
        categories = defaultdict(list)
        for pred in predictions:
            categories[pred.object_category].append(pred)

        category_analysis = {}
        for category, cat_predictions in categories.items():
            valid_errors = [p.l2_error for p in cat_predictions if p.l2_error != float('inf')]

            if valid_errors:
                category_analysis[category] = {
                    'count': len(cat_predictions),
                    'mean_error': np.mean(valid_errors),
                    'success_rate': sum(1 for e in valid_errors if e < self.error_thresholds['acceptable']) / len(valid_errors),
                    'mean_direction_accuracy': np.mean([p.direction_accuracy for p in cat_predictions])
                }

        return category_analysis

    def generate_analysis_report(self, results: TrajectoryBenchmarkResult, output_file: str = None) -> str:
        """Generate comprehensive analysis report."""
        report = f"""
Trajectory Prediction Benchmark Report
=====================================

OVERALL PERFORMANCE
------------------
Total Predictions: {results.aggregate_metrics['total_predictions']}
Valid Predictions: {results.aggregate_metrics['valid_predictions']}
Mean L2 Error: {results.aggregate_metrics['mean_l2_error']:.2f} pixels
Median L2 Error: {results.aggregate_metrics['median_l2_error']:.2f} pixels
Direction Accuracy: {results.aggregate_metrics['mean_direction_accuracy']:.1%}
Success Rate: {results.aggregate_metrics['success_rate']:.1%}

ACCURACY DISTRIBUTION
--------------------
"""
        for level, percentage in results.aggregate_metrics['accuracy_distribution'].items():
            report += f"{level.capitalize()}: {percentage:.1%}\n"

        report += f"""
GAME-SPECIFIC PERFORMANCE
------------------------
"""
        for game, metrics in results.game_breakdown.items():
            report += f"{game}: {metrics['mean_error']:.1f}px error, {metrics['success_rate']:.1%} success\n"

        report += f"""
HORIZON ANALYSIS
---------------
"""
        for horizon, metrics in results.horizon_analysis.items():
            report += f"{horizon}-step prediction: {metrics['mean_error']:.1f}px error, {metrics['error_growth_rate']:.1f}px/step\n"

        report += f"""
FAILURE ANALYSIS
---------------
"""
        if results.error_analysis and 'failure_modes' in results.error_analysis:
            report += f"Failure Rate: {results.error_analysis['failure_modes']['failure_rate']:.1%}\n"
            report += f"Failed Categories: {results.error_analysis['failure_modes'].get('common_failure_categories', {})}\n"
        else:
            report += "No failure analysis data available\n"

        if output_file:
            # Export full results to JSON
            export_data = {
                'report': report,
                'detailed_results': {
                    'aggregate_metrics': results.aggregate_metrics,
                    'error_analysis': results.error_analysis,
                    'game_breakdown': results.game_breakdown,
                    'horizon_analysis': results.horizon_analysis,
                    'category_analysis': results.category_analysis
                },
                'predictions': [
                    {
                        'frame_id': p.frame_id,
                        'object_category': p.object_category,
                        'predicted_positions': p.predicted_positions,
                        'actual_positions': p.actual_positions,
                        'l2_error': p.l2_error,
                        'direction_accuracy': p.direction_accuracy,
                        'prediction_horizon': p.prediction_horizon
                    }
                    for p in results.predictions
                ]
            }

            with open(output_file, 'w') as f:
                json.dump(export_data, f, indent=2)

            print(f"Detailed results exported to {output_file}")

        return report

    def create_visualization(self, results: TrajectoryBenchmarkResult, output_path: str):
        """Create visualization plots for trajectory prediction results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Error distribution histogram
        valid_errors = [p.l2_error for p in results.predictions if p.l2_error != float('inf')]
        axes[0, 0].hist(valid_errors, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('L2 Error Distribution')
        axes[0, 0].set_xlabel('L2 Error (pixels)')
        axes[0, 0].set_ylabel('Frequency')

        # Error by prediction horizon
        horizon_means = [results.horizon_analysis[h]['mean_error'] for h in sorted(results.horizon_analysis.keys())]
        horizon_labels = sorted(results.horizon_analysis.keys())
        axes[0, 1].plot(horizon_labels, horizon_means, 'o-')
        axes[0, 1].set_title('Error vs Prediction Horizon')
        axes[0, 1].set_xlabel('Prediction Steps')
        axes[0, 1].set_ylabel('Mean L2 Error (pixels)')

        # Success rate by game
        if results.game_breakdown:
            games = list(results.game_breakdown.keys())
            success_rates = [results.game_breakdown[g]['success_rate'] * 100 for g in games]
            axes[1, 0].bar(games, success_rates)
            axes[1, 0].set_title('Success Rate by Game')
            axes[1, 0].set_ylabel('Success Rate (%)')

        # Direction accuracy distribution
        direction_accuracies = [p.direction_accuracy for p in results.predictions]
        axes[1, 1].hist(direction_accuracies, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Direction Accuracy Distribution')
        axes[1, 1].set_xlabel('Direction Accuracy')
        axes[1, 1].set_ylabel('Frequency')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Visualization saved to {output_path}")


def create_mock_trajectory_pipeline() -> Callable[[np.ndarray, str], str]:
    """Create a mock pipeline for testing trajectory prediction."""
    def mock_pipeline(frame: np.ndarray, prompt: str) -> str:
        """Mock trajectory prediction pipeline."""
        # Simulate predictions based on prompt
        if "next frame" in prompt.lower():
            return "The object will be at position (75, 95)"
        elif "next 3 frames" in prompt.lower():
            return "Predicted positions: (76, 94), (77, 93), (78, 92)"
        elif "next 5 frames" in prompt.lower():
            return "Trajectory: (76, 94), (77, 93), (78, 92), (79, 91), (80, 90)"
        return "Unable to predict trajectory"

    return mock_pipeline


if __name__ == "__main__":
    # Test trajectory prediction benchmark
    print("Testing Trajectory Prediction Benchmark")
    print("=" * 50)

    benchmark = TrajectoryPredictionBenchmark()

    # Create mock test data
    test_frames = []
    for i in range(3):
        frame_data = {
            'frame_id': f'Pong_test_{i}',
            'game': 'Pong',
            'frame': np.random.randint(0, 255, (210, 160, 3)),
            'objects': [
                {
                    'position': [75 + i, 100 - i],
                    'velocity': [1.0, -1.0],
                    'category': 'Ball'
                }
            ]
        }
        test_frames.append(frame_data)

    # Test with mock pipeline
    mock_pipeline = create_mock_trajectory_pipeline()
    results = benchmark.evaluate_pipeline(mock_pipeline, test_frames)

    # Generate report
    report = benchmark.generate_analysis_report(results, 'test_trajectory_results.json')
    print(report)

    # Create visualization
    benchmark.create_visualization(results, 'test_trajectory_analysis.png')

    print("Trajectory prediction benchmark test completed!")