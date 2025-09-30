"""
Automated Scoring Rubric for Spatial Reasoning Benchmarks

This module provides comprehensive automated scoring for both:
1. Atari-GPT diagnostic evaluations (Visual, Spatial, Strategy, Identification)
2. New predictive spatial tasks (Trajectory, Collision, Spatial Relationships)

Uses ground truth validation from OCAtari for objective scoring where possible,
and sophisticated NLP analysis for subjective evaluation categories.

Enhanced with detailed match logging for complete transparency.

Usage:
    scorer = AutomatedScoringRubric(enable_detailed_logging=True, log_output_dir="./logs")
    score = scorer.score_response(response, task_type, ground_truth)
    detailed_analysis = scorer.analyze_response_quality(response, task_type)
"""

import re
import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
import math

# Import detailed logger if available
try:
    from detailed_match_logger import DetailedMatchLogger
    DETAILED_LOGGING_AVAILABLE = True
except ImportError:
    DetailedMatchLogger = None
    DETAILED_LOGGING_AVAILABLE = False


@dataclass
class ScoringResult:
    """Detailed scoring result with breakdown."""
    score: float  # 0.0 to 1.0
    max_score: float
    components: Dict[str, float]  # Breakdown by scoring component
    reasoning: str
    confidence: float  # Confidence in the scoring (0.0 to 1.0)
    issues: List[str]  # Any detected issues or concerns


class AutomatedScoringRubric:
    """
    Comprehensive automated scoring system for spatial reasoning benchmarks.

    Provides objective and subjective scoring across multiple task types with
    detailed breakdown and confidence assessment.
    """

    def __init__(self, enable_detailed_logging: bool = False, log_output_dir: str = None):
        """
        Initialize scoring rubric with task-specific criteria.

        Args:
            enable_detailed_logging: Enable detailed match logging
            log_output_dir: Directory for detailed log files
        """
        self.enable_detailed_logging = enable_detailed_logging and DETAILED_LOGGING_AVAILABLE
        self.logger = None

        if self.enable_detailed_logging and log_output_dir:
            self.logger = DetailedMatchLogger(log_output_dir)
        self.task_criteria = {
            # Atari-GPT diagnostic tasks
            'visual': self._init_visual_criteria(),
            'spatial': self._init_spatial_criteria(),
            'strategy': self._init_strategy_criteria(),
            'identification': self._init_identification_criteria(),

            # New predictive spatial tasks
            'trajectory_prediction': self._init_trajectory_criteria(),
            'collision_detection': self._init_collision_criteria(),
            'spatial_relationships': self._init_spatial_relationship_criteria()
        }

        # Common spatial reasoning vocabulary
        self.spatial_vocabulary = {
            'position_words': ['left', 'right', 'top', 'bottom', 'center', 'middle', 'corner', 'edge'],
            'distance_words': ['near', 'far', 'close', 'distant', 'adjacent', 'between'],
            'direction_words': ['up', 'down', 'above', 'below', 'beside', 'next to'],
            'relative_words': ['relative', 'compared', 'relation', 'position', 'location'],
            'coordinate_patterns': [r'\b\d+\s*,\s*\d+\b', r'\bx\s*[=:]\s*\d+\b', r'\by\s*[=:]\s*\d+\b']
        }

    def _init_visual_criteria(self) -> Dict[str, Any]:
        """Initialize visual understanding scoring criteria."""
        return {
            'object_detection': {
                'weight': 0.4,
                'keywords': ['ball', 'paddle', 'player', 'alien', 'invader', 'brick', 'block', 'bullet', 'ship'],
                'required_elements': 2,  # Minimum elements to identify
                'bonus_elements': ['shield', 'enemy', 'opponent']
            },
            'object_properties': {
                'weight': 0.3,
                'color_mentions': ['green', 'red', 'blue', 'yellow', 'white', 'black'],
                'size_mentions': ['small', 'large', 'big', 'tiny', 'wide', 'narrow'],
                'shape_mentions': ['square', 'rectangle', 'round', 'circular']
            },
            'specificity': {
                'weight': 0.3,
                'coordinate_bonus': 0.2,
                'count_bonus': 0.1,  # Bonus for counting objects
                'detail_bonus': 0.1   # Bonus for detailed descriptions
            }
        }

    def _init_spatial_criteria(self) -> Dict[str, Any]:
        """Initialize spatial reasoning scoring criteria."""
        return {
            'position_accuracy': {
                'weight': 0.5,
                'coordinate_accuracy_weight': 0.6,  # If coordinates provided
                'relative_position_weight': 0.4     # If relative descriptions
            },
            'spatial_language': {
                'weight': 0.3,
                'required_spatial_terms': 3,  # Minimum spatial terms expected
                'precision_bonus': 0.2         # Bonus for precise language
            },
            'relationship_understanding': {
                'weight': 0.2,
                'distance_estimation': 0.5,
                'orientation_awareness': 0.5
            }
        }

    def _init_strategy_criteria(self) -> Dict[str, Any]:
        """Initialize strategic planning scoring criteria."""
        return {
            'action_validity': {
                'weight': 0.4,
                'game_appropriate_actions': ['move', 'hit', 'shoot', 'aim', 'avoid', 'block'],
                'specificity_bonus': 0.2
            },
            'strategic_thinking': {
                'weight': 0.4,
                'future_planning': ['next', 'then', 'after', 'following', 'sequence'],
                'goal_awareness': ['win', 'score', 'avoid', 'protect', 'target']
            },
            'contextual_understanding': {
                'weight': 0.2,
                'situation_awareness': 0.6,
                'adaptation_awareness': 0.4
            }
        }

    def _init_identification_criteria(self) -> Dict[str, Any]:
        """Initialize game identification scoring criteria."""
        return {
            'exact_match': {
                'weight': 1.0,
                'game_names': {
                    'pong': ['pong'],
                    'breakout': ['breakout', 'brick breaker'],
                    'spaceinvaders': ['space invaders', 'space-invaders', 'spaceinvaders']
                }
            }
        }

    def _init_trajectory_criteria(self) -> Dict[str, Any]:
        """Initialize trajectory prediction scoring criteria."""
        return {
            'position_accuracy': {
                'weight': 0.7,
                'tolerance_pixels': 10,  # Acceptable error in pixels
                'perfect_threshold': 5   # Perfect score threshold
            },
            'direction_accuracy': {
                'weight': 0.2,
                'angle_tolerance': 15    # Degrees tolerance for direction
            },
            'reasoning_quality': {
                'weight': 0.1,
                'physics_awareness': ['velocity', 'direction', 'bounce', 'trajectory', 'path']
            }
        }

    def _init_collision_criteria(self) -> Dict[str, Any]:
        """Initialize collision detection scoring criteria."""
        return {
            'collision_prediction': {
                'weight': 0.5,
                'binary_accuracy': 1.0  # Will/won't collision occur
            },
            'timing_accuracy': {
                'weight': 0.3,
                'frame_tolerance': 2    # Acceptable error in frames
            },
            'location_accuracy': {
                'weight': 0.2,
                'pixel_tolerance': 15   # Acceptable error in collision location
            }
        }

    def _init_spatial_relationship_criteria(self) -> Dict[str, Any]:
        """Initialize spatial relationship scoring criteria."""
        return {
            'distance_accuracy': {
                'weight': 0.5,
                'relative_error_threshold': 0.1  # 10% relative error acceptable
            },
            'ordering_accuracy': {
                'weight': 0.3,
                'ranking_correctness': 1.0
            },
            'relationship_identification': {
                'weight': 0.2,
                'relationship_terms': ['closest', 'farthest', 'between', 'above', 'below']
            }
        }

    def score_response(self,
                      response: str,
                      task_type: str,
                      ground_truth: Optional[Dict[str, Any]] = None,
                      game_name: Optional[str] = None) -> ScoringResult:
        """
        Score a response using task-specific criteria.

        Args:
            response: Text response to score
            task_type: Type of task being scored
            ground_truth: Optional ground truth data for objective validation
            game_name: Game name for context-specific scoring

        Returns:
            Detailed ScoringResult with breakdown and reasoning
        """
        if task_type not in self.task_criteria:
            return ScoringResult(
                score=0.0, max_score=1.0, components={}, reasoning="Unknown task type",
                confidence=0.0, issues=[f"Unknown task type: {task_type}"]
            )

        if not response or response.strip() == "":
            return ScoringResult(
                score=0.0, max_score=1.0, components={}, reasoning="Empty response",
                confidence=1.0, issues=["Empty or null response"]
            )

        # Route to appropriate scoring method
        if task_type in ['visual', 'spatial', 'strategy', 'identification']:
            result = self._score_diagnostic_task(response, task_type, ground_truth, game_name)
        else:
            result = self._score_predictive_task(response, task_type, ground_truth, game_name)

        # Log detailed scoring if enabled
        if self.enable_detailed_logging and self.logger:
            frame_id = ground_truth.get('frame_id', 'unknown') if ground_truth else 'unknown'
            pipeline_type = ground_truth.get('pipeline_type', 'unknown') if ground_truth else 'unknown'
            self.logger.log_diagnostic_scoring(
                response=response,
                task_type=task_type,
                scoring_result=result,
                ground_truth=ground_truth,
                frame_id=frame_id,
                pipeline_type=pipeline_type
            )

        return result

    def _score_diagnostic_task(self,
                              response: str,
                              task_type: str,
                              ground_truth: Optional[Dict[str, Any]] = None,
                              game_name: Optional[str] = None) -> ScoringResult:
        """Score diagnostic tasks (Visual, Spatial, Strategy, Identification)."""
        response_lower = response.lower()
        criteria = self.task_criteria[task_type]
        components = {}
        issues = []
        reasoning_parts = []

        total_score = 0.0
        confidence_factors = []

        if task_type == 'identification':
            # Special handling for identification - exact match required
            game_key = game_name.lower() if game_name else 'unknown'
            expected_names = criteria['exact_match']['game_names'].get(game_key, [])

            for name in expected_names:
                if name in response_lower:
                    total_score = 1.0
                    components['exact_match'] = 1.0
                    reasoning_parts.append(f"Correctly identified as {name}")
                    confidence_factors.append(1.0)
                    break
            else:
                total_score = 0.0
                components['exact_match'] = 0.0
                reasoning_parts.append("Failed to correctly identify game")
                confidence_factors.append(1.0)
                issues.append("Incorrect or missing game identification")

        elif task_type == 'visual':
            # Object detection scoring
            obj_criteria = criteria['object_detection']
            detected_objects = sum(1 for keyword in obj_criteria['keywords'] if keyword in response_lower)
            object_score = min(1.0, detected_objects / obj_criteria['required_elements'])
            components['object_detection'] = object_score * obj_criteria['weight']

            # Object properties scoring
            prop_criteria = criteria['object_properties']
            color_mentions = sum(1 for color in prop_criteria['color_mentions'] if color in response_lower)
            size_mentions = sum(1 for size in prop_criteria['size_mentions'] if size in response_lower)
            property_score = min(1.0, (color_mentions + size_mentions) / 3)  # Normalize
            components['object_properties'] = property_score * prop_criteria['weight']

            # Specificity scoring
            spec_criteria = criteria['specificity']
            coordinate_bonus = spec_criteria['coordinate_bonus'] if any(
                re.search(pattern, response_lower) for pattern in self.spatial_vocabulary['coordinate_patterns']
            ) else 0.0
            components['specificity'] = coordinate_bonus * spec_criteria['weight']

            total_score = sum(components.values())
            confidence_factors.extend([0.8, 0.7, 0.9])  # Confidence in each component

        elif task_type == 'spatial':
            # Position accuracy (requires ground truth for objective scoring)
            pos_criteria = criteria['position_accuracy']
            if ground_truth and 'objects' in ground_truth:
                position_score = self._score_position_accuracy(response, ground_truth)
                components['position_accuracy'] = position_score * pos_criteria['weight']
                confidence_factors.append(0.95)  # High confidence with ground truth
            else:
                # Subjective scoring based on spatial language quality
                position_score = self._score_spatial_language_quality(response)
                components['position_accuracy'] = position_score * pos_criteria['weight']
                confidence_factors.append(0.6)  # Lower confidence without ground truth

            # Spatial language usage
            lang_criteria = criteria['spatial_language']
            spatial_terms = sum(1 for category in self.spatial_vocabulary.values()
                              if isinstance(category, list)
                              for term in category if term in response_lower)
            language_score = min(1.0, spatial_terms / lang_criteria['required_spatial_terms'])
            components['spatial_language'] = language_score * lang_criteria['weight']
            confidence_factors.append(0.8)

            # Relationship understanding
            rel_criteria = criteria['relationship_understanding']
            relationship_score = self._score_relationship_understanding(response, ground_truth)
            components['relationship_understanding'] = relationship_score * rel_criteria['weight']
            confidence_factors.append(0.7)

            total_score = sum(components.values())

        elif task_type == 'strategy':
            # Action validity
            action_criteria = criteria['action_validity']
            valid_actions = sum(1 for action in action_criteria['game_appropriate_actions']
                              if action in response_lower)
            action_score = min(1.0, valid_actions / 2)  # Expect at least 2 actions
            components['action_validity'] = action_score * action_criteria['weight']

            # Strategic thinking
            strategy_criteria = criteria['strategic_thinking']
            future_terms = sum(1 for term in strategy_criteria['future_planning'] if term in response_lower)
            goal_terms = sum(1 for term in strategy_criteria['goal_awareness'] if term in response_lower)
            strategic_score = min(1.0, (future_terms + goal_terms) / 3)
            components['strategic_thinking'] = strategic_score * strategy_criteria['weight']

            # Contextual understanding
            context_criteria = criteria['contextual_understanding']
            context_score = self._score_contextual_understanding(response, game_name)
            components['contextual_understanding'] = context_score * context_criteria['weight']

            total_score = sum(components.values())
            confidence_factors.extend([0.8, 0.7, 0.6])

        # Calculate overall confidence
        confidence = np.mean(confidence_factors) if confidence_factors else 0.5

        # Generate reasoning
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else f"Scored based on {task_type} criteria"

        return ScoringResult(
            score=total_score,
            max_score=1.0,
            components=components,
            reasoning=reasoning,
            confidence=confidence,
            issues=issues
        )

    def _score_predictive_task(self,
                              response: str,
                              task_type: str,
                              ground_truth: Optional[Dict[str, Any]] = None,
                              game_name: Optional[str] = None) -> ScoringResult:
        """Score predictive spatial tasks (Trajectory, Collision, Spatial Relationships)."""
        if not ground_truth:
            return ScoringResult(
                score=0.0, max_score=1.0, components={}, reasoning="Ground truth required for predictive tasks",
                confidence=0.0, issues=["No ground truth provided for objective scoring"]
            )

        if task_type == 'trajectory_prediction':
            return self._score_trajectory_prediction(response, ground_truth)
        elif task_type == 'collision_detection':
            return self._score_collision_detection(response, ground_truth)
        elif task_type == 'spatial_relationships':
            return self._score_spatial_relationships(response, ground_truth)
        else:
            return ScoringResult(
                score=0.0, max_score=1.0, components={}, reasoning="Unknown predictive task",
                confidence=0.0, issues=[f"Unknown predictive task: {task_type}"]
            )

    def _score_trajectory_prediction(self, response: str, ground_truth: Dict[str, Any]) -> ScoringResult:
        """Score trajectory prediction accuracy."""
        components = {}
        issues = []

        # Extract predicted coordinates from response
        predicted_coords = self._extract_coordinates_from_response(response)
        actual_coords = ground_truth.get('predicted_trajectory', [])

        if not predicted_coords:
            return ScoringResult(
                score=0.0, max_score=1.0, components={}, reasoning="No coordinates found in response",
                confidence=1.0, issues=["Response contains no coordinate predictions"]
            )

        if not actual_coords:
            issues.append("No ground truth trajectory provided")
            return ScoringResult(
                score=0.0, max_score=1.0, components={}, reasoning="No ground truth available",
                confidence=0.0, issues=issues
            )

        # Calculate position accuracy
        min_length = min(len(predicted_coords), len(actual_coords))
        if min_length == 0:
            position_accuracy = 0.0
        else:
            total_error = 0.0
            for i in range(min_length):
                pred_x, pred_y = predicted_coords[i]
                actual_x, actual_y = actual_coords[i]
                error = math.sqrt((pred_x - actual_x)**2 + (pred_y - actual_y)**2)
                total_error += error

            avg_error = total_error / min_length
            criteria = self.task_criteria['trajectory_prediction']
            tolerance = criteria['position_accuracy']['tolerance_pixels']
            perfect_threshold = criteria['position_accuracy']['perfect_threshold']

            if avg_error <= perfect_threshold:
                position_accuracy = 1.0
            elif avg_error <= tolerance:
                position_accuracy = 1.0 - (avg_error - perfect_threshold) / (tolerance - perfect_threshold)
            else:
                position_accuracy = 0.0

        components['position_accuracy'] = position_accuracy * criteria['position_accuracy']['weight']

        # Direction accuracy (if we have enough points)
        if min_length >= 2:
            direction_accuracy = self._calculate_direction_accuracy(predicted_coords, actual_coords)
            components['direction_accuracy'] = direction_accuracy * criteria['direction_accuracy']['weight']
        else:
            components['direction_accuracy'] = 0.0
            issues.append("Insufficient points for direction analysis")

        # Reasoning quality
        reasoning_score = self._score_physics_reasoning(response)
        components['reasoning_quality'] = reasoning_score * criteria['reasoning_quality']['weight']

        total_score = sum(components.values())

        return ScoringResult(
            score=total_score,
            max_score=1.0,
            components=components,
            reasoning=f"Position accuracy: {position_accuracy:.2f}, avg error: {avg_error:.1f}px",
            confidence=0.9,
            issues=issues
        )

    def _extract_coordinates_from_response(self, response: str) -> List[Tuple[int, int]]:
        """Extract coordinate pairs from response text."""
        coords = []

        # Pattern for (x, y) coordinates
        coord_pattern = r'\(\s*(\d+)\s*,\s*(\d+)\s*\)'
        matches = re.findall(coord_pattern, response)

        for match in matches:
            x, y = int(match[0]), int(match[1])
            coords.append((x, y))

        # Alternative pattern: x=123, y=456
        if not coords:
            x_pattern = r'x\s*[=:]\s*(\d+)'
            y_pattern = r'y\s*[=:]\s*(\d+)'

            x_matches = re.findall(x_pattern, response.lower())
            y_matches = re.findall(y_pattern, response.lower())

            for x_val, y_val in zip(x_matches, y_matches):
                coords.append((int(x_val), int(y_val)))

        return coords

    def _score_position_accuracy(self, response: str, ground_truth: Dict[str, Any]) -> float:
        """Score position accuracy against ground truth."""
        if 'objects' not in ground_truth:
            return 0.5  # Neutral score without ground truth

        predicted_coords = self._extract_coordinates_from_response(response)
        actual_objects = ground_truth['objects']

        if not predicted_coords or not actual_objects:
            return 0.0

        # Match predicted coordinates to actual object positions
        total_accuracy = 0.0
        matched_objects = 0

        for pred_coord in predicted_coords:
            best_match_accuracy = 0.0

            for obj in actual_objects:
                if 'position' in obj:
                    actual_pos = obj['position']
                    distance = math.sqrt((pred_coord[0] - actual_pos[0])**2 +
                                       (pred_coord[1] - actual_pos[1])**2)

                    # Accuracy decreases with distance (10 pixel tolerance)
                    if distance <= 10:
                        accuracy = 1.0 - (distance / 10)
                        best_match_accuracy = max(best_match_accuracy, accuracy)

            total_accuracy += best_match_accuracy
            matched_objects += 1

        return total_accuracy / matched_objects if matched_objects > 0 else 0.0

    def _score_spatial_language_quality(self, response: str) -> float:
        """Score quality of spatial language usage."""
        response_lower = response.lower()

        # Count spatial terms
        spatial_term_count = 0
        for category in self.spatial_vocabulary.values():
            if isinstance(category, list):
                spatial_term_count += sum(1 for term in category if term in response_lower)

        # Bonus for coordinate usage
        coordinate_bonus = 0.3 if any(re.search(pattern, response_lower)
                                     for pattern in self.spatial_vocabulary['coordinate_patterns']) else 0.0

        # Base score from spatial terms (normalize to 0-0.7)
        base_score = min(0.7, spatial_term_count / 10)

        return base_score + coordinate_bonus

    def _score_relationship_understanding(self, response: str, ground_truth: Optional[Dict[str, Any]]) -> float:
        """Score understanding of spatial relationships."""
        response_lower = response.lower()

        # Look for relationship descriptions
        relationship_words = ['between', 'near', 'far', 'closest', 'farthest', 'distance']
        relationship_score = sum(1 for word in relationship_words if word in response_lower) / len(relationship_words)

        # Bonus for comparative language
        comparative_words = ['more', 'less', 'closer', 'further', 'nearer', 'compared']
        comparative_bonus = 0.2 if any(word in response_lower for word in comparative_words) else 0.0

        return min(1.0, relationship_score + comparative_bonus)

    def _score_contextual_understanding(self, response: str, game_name: Optional[str]) -> float:
        """Score contextual understanding of game situation."""
        if not game_name:
            return 0.5  # Neutral without context

        response_lower = response.lower()
        game_lower = game_name.lower()

        # Game-specific context scoring
        if 'pong' in game_lower:
            context_words = ['paddle', 'ball', 'hit', 'return', 'serve']
        elif 'breakout' in game_lower:
            context_words = ['paddle', 'ball', 'brick', 'break', 'bounce']
        elif 'spaceinvaders' in game_lower or 'space' in game_lower:
            context_words = ['shoot', 'alien', 'invader', 'bullet', 'dodge']
        else:
            context_words = ['move', 'action', 'game']

        context_score = sum(1 for word in context_words if word in response_lower) / len(context_words)
        return context_score

    def _calculate_direction_accuracy(self, predicted: List[Tuple[int, int]], actual: List[Tuple[int, int]]) -> float:
        """Calculate accuracy of predicted direction/velocity."""
        if len(predicted) < 2 or len(actual) < 2:
            return 0.0

        # Calculate predicted direction vector
        pred_dx = predicted[1][0] - predicted[0][0]
        pred_dy = predicted[1][1] - predicted[0][1]

        # Calculate actual direction vector
        actual_dx = actual[1][0] - actual[0][0]
        actual_dy = actual[1][1] - actual[0][1]

        # Calculate angle difference
        if pred_dx == 0 and pred_dy == 0:
            return 0.0
        if actual_dx == 0 and actual_dy == 0:
            return 0.0

        pred_angle = math.atan2(pred_dy, pred_dx)
        actual_angle = math.atan2(actual_dy, actual_dx)

        angle_diff = abs(pred_angle - actual_angle)
        angle_diff = min(angle_diff, 2 * math.pi - angle_diff)  # Use smaller angle

        # Convert to degrees and normalize
        angle_diff_degrees = math.degrees(angle_diff)
        tolerance = self.task_criteria['trajectory_prediction']['direction_accuracy']['angle_tolerance']

        if angle_diff_degrees <= tolerance:
            return 1.0 - (angle_diff_degrees / tolerance)
        else:
            return 0.0

    def _score_physics_reasoning(self, response: str) -> float:
        """Score quality of physics reasoning in response."""
        response_lower = response.lower()
        physics_terms = ['velocity', 'direction', 'bounce', 'trajectory', 'path', 'momentum', 'speed']

        physics_score = sum(1 for term in physics_terms if term in response_lower) / len(physics_terms)
        return physics_score

    def _score_collision_detection(self, response: str, ground_truth: Dict[str, Any]) -> ScoringResult:
        """Score collision detection predictions."""
        # Simplified implementation - would need actual collision ground truth
        components = {
            'collision_prediction': 0.5,  # Placeholder
            'timing_accuracy': 0.5,       # Placeholder
            'location_accuracy': 0.5      # Placeholder
        }

        return ScoringResult(
            score=0.5, max_score=1.0, components=components,
            reasoning="Collision detection scoring not fully implemented",
            confidence=0.3, issues=["Placeholder implementation"]
        )

    def _score_spatial_relationships(self, response: str, ground_truth: Dict[str, Any]) -> ScoringResult:
        """Score spatial relationship queries."""
        # Simplified implementation - would need actual spatial ground truth
        components = {
            'distance_accuracy': 0.5,    # Placeholder
            'ordering_accuracy': 0.5,    # Placeholder
            'relationship_identification': 0.5  # Placeholder
        }

        return ScoringResult(
            score=0.5, max_score=1.0, components=components,
            reasoning="Spatial relationships scoring not fully implemented",
            confidence=0.3, issues=["Placeholder implementation"]
        )

    def batch_score(self,
                   responses_and_context: List[Tuple[str, str, Optional[Dict[str, Any]], Optional[str]]],
                   ) -> Dict[str, Any]:
        """
        Score multiple responses in batch.

        Args:
            responses_and_context: List of (response, task_type, ground_truth, game_name) tuples

        Returns:
            Batch scoring results with aggregate statistics
        """
        results = []

        for response, task_type, ground_truth, game_name in responses_and_context:
            result = self.score_response(response, task_type, ground_truth, game_name)
            results.append(result)

        # Aggregate statistics
        scores = [r.score for r in results]
        confidences = [r.confidence for r in results]

        # Group by task type
        by_task_type = defaultdict(list)
        for i, (_, task_type, _, _) in enumerate(responses_and_context):
            by_task_type[task_type].append(results[i])

        task_type_stats = {}
        for task_type, task_results in by_task_type.items():
            task_scores = [r.score for r in task_results]
            task_type_stats[task_type] = {
                'count': len(task_scores),
                'mean_score': np.mean(task_scores),
                'std_score': np.std(task_scores),
                'min_score': np.min(task_scores),
                'max_score': np.max(task_scores)
            }

        return {
            'individual_results': results,
            'aggregate_stats': {
                'total_responses': len(results),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'mean_confidence': np.mean(confidences),
                'task_type_breakdown': task_type_stats
            }
        }


if __name__ == "__main__":
    # Test the scoring rubric
    scorer = AutomatedScoringRubric()

    # Test cases
    test_cases = [
        ("I see a ball at position (75, 100) and two paddles", "visual", None, "Pong"),
        ("The ball is on the left side near x=75, paddle on right at x=140", "spatial",
         {"objects": [{"position": [75, 100], "category": "Ball"}, {"position": [140, 120], "category": "Player"}]}, "Pong"),
        ("Move the paddle up to intercept the ball's trajectory", "strategy", None, "Pong"),
        ("This is Pong", "identification", None, "Pong"),
        ("Ball will be at (80, 95), (85, 90), (90, 85)", "trajectory_prediction",
         {"predicted_trajectory": [(80, 95), (85, 90), (90, 85)]}, "Pong")
    ]

    print("Testing Automated Scoring Rubric")
    print("=" * 50)

    for response, task_type, ground_truth, game_name in test_cases:
        result = scorer.score_response(response, task_type, ground_truth, game_name)

        print(f"\nTask: {task_type}")
        print(f"Response: {response}")
        print(f"Score: {result.score:.2f} (confidence: {result.confidence:.2f})")
        print(f"Components: {result.components}")
        print(f"Reasoning: {result.reasoning}")
        if result.issues:
            print(f"Issues: {result.issues}")

    # Test batch scoring
    print(f"\n{'=' * 50}")
    print("Batch Scoring Test")
    print("=" * 50)

    batch_results = scorer.batch_score([
        (case[0], case[1], case[2], case[3]) for case in test_cases
    ])

    print(f"Total responses: {batch_results['aggregate_stats']['total_responses']}")
    print(f"Mean score: {batch_results['aggregate_stats']['mean_score']:.2f}")
    print(f"Mean confidence: {batch_results['aggregate_stats']['mean_confidence']:.2f}")

    for task_type, stats in batch_results['aggregate_stats']['task_type_breakdown'].items():
        print(f"{task_type}: {stats['mean_score']:.2f} ± {stats['std_score']:.2f}")