"""
Atari-GPT Diagnostic Framework

This module implements the exact diagnostic evaluation framework used in the
Atari-GPT paper by Waytowich et al. It enables direct comparison of our
Vision+Symbol approach against their Vision-only baseline.

Based on their 4-category evaluation:
1. Visual Understanding - Object identification
2. Spatial Reasoning - Relative positions
3. Strategic Planning - Next move decisions
4. Environment Identification - Game recognition

Usage:
    evaluator = AtariGPTDiagnostic()
    results = evaluator.evaluate_vision_pipeline(frames, pipeline_func)
    comparison = evaluator.compare_pipelines(vision_only_results, vision_symbol_results)
"""

import numpy as np
import base64
import json
import os
from typing import Dict, List, Tuple, Any, Callable, Optional
from dataclasses import dataclass
from io import BytesIO
from PIL import Image
import re

from ocatari_ground_truth import OCAtariGroundTruth

# Import detailed logging components
try:
    from automated_scoring_rubric import AutomatedScoringRubric
    from ground_truth_comparator import GroundTruthComparator
    AUTOMATED_SCORING_AVAILABLE = True
except ImportError:
    AutomatedScoringRubric = None
    GroundTruthComparator = None
    AUTOMATED_SCORING_AVAILABLE = False


@dataclass
class DiagnosticResult:
    """Single diagnostic evaluation result."""
    category: str  # 'visual', 'spatial', 'strategy', 'identification'
    prompt: str
    response: str
    score: float  # 0.0 to 1.0
    ground_truth: Optional[Dict[str, Any]] = None
    reasoning: Optional[str] = None


class AtariGPTDiagnostic:
    """
    Implements Atari-GPT's diagnostic evaluation framework.

    Provides exact replication of their 4-category evaluation approach
    for direct comparison of pipeline performance.
    """

    def __init__(self, enable_detailed_logging: bool = False, log_output_dir: str = None):
        """
        Initialize diagnostic evaluator with Atari-GPT prompts.

        Args:
            enable_detailed_logging: Enable detailed match logging
            log_output_dir: Directory for detailed log files
        """
        self.enable_detailed_logging = enable_detailed_logging and AUTOMATED_SCORING_AVAILABLE
        self.scorer = None
        self.ground_truth_comparator = None
        if self.enable_detailed_logging and log_output_dir:
            self.scorer = AutomatedScoringRubric(enable_detailed_logging, log_output_dir)
            self.ground_truth_comparator = GroundTruthComparator(log_output_dir)
            print(f"   🔬 Detailed logging enabled for AtariGPTDiagnostic")
        else:
            print(f"   ⚠️  Detailed logging NOT enabled: enable={enable_detailed_logging}, available={AUTOMATED_SCORING_AVAILABLE}, dir={log_output_dir}")
        self.prompts = {
            'visual': "Identify all key elements in this image. Be specific. Use at most 100 words.",
            'spatial': "Where are the key elements located relative to each other? Be specific with respect to their position in the image. Use at most 100 words.",
            'strategy': "Describe the ideal next move if you were playing this game. Be specific. Use at most 100 words.",
            'identification': "Identify the game name. Be specific."
        }

        # Game-specific expected elements for scoring
        self.expected_elements = {
            'Pong': {
                'visual': ['ball', 'paddle', 'player', 'opponent'],
                'spatial': ['left', 'right', 'top', 'bottom', 'center', 'near', 'far'],
                'strategy': ['move', 'hit', 'block', 'return', 'aim'],
                'identification': ['pong']
            },
            'Breakout': {
                'visual': ['ball', 'paddle', 'brick', 'block', 'player'],
                'spatial': ['above', 'below', 'left', 'right', 'top', 'bottom'],
                'strategy': ['move', 'hit', 'aim', 'break', 'bounce'],
                'identification': ['breakout']
            },
            'SpaceInvaders': {
                'visual': ['alien', 'invader', 'ship', 'bullet', 'player', 'shield'],
                'spatial': ['above', 'below', 'left', 'right', 'formation', 'row'],
                'strategy': ['shoot', 'move', 'avoid', 'dodge', 'aim'],
                'identification': ['space invaders', 'space-invaders', 'spaceinvaders']
            }
        }

    def evaluate_single_frame(self,
                            frame: np.ndarray,
                            game_name: str,
                            pipeline_func: Callable[[np.ndarray, str], str],
                            ground_truth: Optional[Dict[str, Any]] = None) -> List[DiagnosticResult]:
        """
        Evaluate a single frame using all 4 diagnostic categories.

        Args:
            frame: RGB frame array (210, 160, 3)
            game_name: Name of the game for context
            pipeline_func: Function that takes (frame, prompt) and returns response string
            ground_truth: Optional OCAtari ground truth data for validation

        Returns:
            List of DiagnosticResult objects for each category
        """
        results = []

        for category, prompt in self.prompts.items():
            try:
                # Get response from pipeline
                response = pipeline_func(frame, prompt)

                # Score the response with detailed logging if available
                if self.enable_detailed_logging and self.scorer:
                    print(f"   🔍 Using detailed scorer for {category} task")

                    # Debug ground truth construction
                    print(f"   🧪 Original ground_truth type: {type(ground_truth)}")
                    enhanced_ground_truth = ground_truth.copy() if ground_truth else {}

                    # Convert OCAtari format to AutomatedScoringRubric format
                    if 'objects' in enhanced_ground_truth and enhanced_ground_truth['objects']:
                        converted_objects = []
                        for obj in enhanced_ground_truth['objects']:
                            if isinstance(obj, dict) and 'category' in obj and 'position' in obj:
                                # OCAtari format: convert to expected format
                                converted_obj = {
                                    'label': obj['category'].lower(),
                                    'coordinates': [obj['position'][0], obj['position'][1],
                                                  obj['position'][0] + obj.get('size', [4, 4])[0],
                                                  obj['position'][1] + obj.get('size', [4, 4])[1]],
                                    'description': f"{obj['category']} at {obj['position']}"
                                }
                                converted_objects.append(converted_obj)
                        enhanced_ground_truth['objects'] = converted_objects
                        print(f"   ✅ Converted {len(converted_objects)} objects to scorer format")

                    enhanced_ground_truth.update({
                        'frame_id': f"{game_name}_{category}",  # Use category since frame_id not available
                        'pipeline_type': getattr(pipeline_func, '_pipeline_type', 'unknown')
                    })
                    print(f"   🧪 Enhanced ground_truth objects: {len(enhanced_ground_truth.get('objects', []))} objects")

                    print(f"   🚀 About to call detailed scorer...")
                    try:
                        scoring_result = self.scorer.score_response(
                            response=response,
                            task_type=category,
                            ground_truth=enhanced_ground_truth,
                            game_name=game_name
                        )
                        score = scoring_result.score
                        print(f"   ✅ Detailed scorer returned: {score} (type: {type(score)})")
                    except Exception as e:
                        print(f"   ❌ Detailed scorer error: {e}")
                        import traceback
                        print(f"   📍 Traceback: {traceback.format_exc()}")
                        score = self._score_response(response, category, game_name, ground_truth)
                        print(f"   🔄 Fallback scorer returned: {score}")

                    # Also do detailed ground truth comparison
                    print(f"   🧪 Debug: category={category}, comparator={self.ground_truth_comparator is not None}")
                    if self.ground_truth_comparator:
                        if category == 'visual':
                            comparison = self.ground_truth_comparator.compare_visual_task(
                                response=response,
                                ground_truth=enhanced_ground_truth,
                                pipeline_type=enhanced_ground_truth['pipeline_type'],
                                frame_id=enhanced_ground_truth['frame_id']
                            )
                        elif category == 'spatial':
                            comparison = self.ground_truth_comparator.compare_spatial_task(
                                response=response,
                                ground_truth=enhanced_ground_truth,
                                pipeline_type=enhanced_ground_truth['pipeline_type'],
                                frame_id=enhanced_ground_truth['frame_id']
                            )
                        else:
                            comparison = None

                        if comparison:
                            comparison_file = self.ground_truth_comparator.save_comparison(comparison)
                            print(f"   📋 Ground truth comparison saved: {os.path.basename(comparison_file)}")

                else:
                    # Use original scoring method
                    score = self._score_response(response, category, game_name, ground_truth)

                # Create diagnostic result
                result = DiagnosticResult(
                    category=category,
                    prompt=prompt,
                    response=response,
                    score=score,
                    ground_truth=ground_truth,
                    reasoning=self._generate_scoring_reasoning(response, category, game_name, score)
                )

                print(f"   📊 Final DiagnosticResult: {category}={score}")
                results.append(result)

            except Exception as e:
                # Handle pipeline errors gracefully
                error_result = DiagnosticResult(
                    category=category,
                    prompt=prompt,
                    response=f"ERROR: {str(e)}",
                    score=0.0,
                    reasoning=f"Pipeline error: {str(e)}"
                )
                results.append(error_result)

        return results

    def _score_response(self,
                       response: str,
                       category: str,
                       game_name: str,
                       ground_truth: Optional[Dict[str, Any]] = None) -> float:
        """
        Score a response based on Atari-GPT evaluation criteria.

        Uses keyword matching and content analysis to assign scores from 0.0 to 1.0.
        """
        if not response or response.startswith("ERROR"):
            return 0.0

        response_lower = response.lower()
        expected = self.expected_elements.get(game_name, {}).get(category, [])

        if category == 'identification':
            # Simple exact match for game identification
            for expected_name in expected:
                if expected_name in response_lower:
                    return 1.0
            return 0.0

        elif category == 'visual':
            # Count how many expected visual elements are mentioned
            mentioned_count = sum(1 for element in expected if element in response_lower)

            # Additional scoring based on specificity
            specificity_bonus = 0.0
            if ground_truth and 'objects' in ground_truth:
                # Check if response mentions actual object categories from ground truth
                for obj in ground_truth['objects']:
                    obj_category = obj['category'].lower()
                    if obj_category in response_lower:
                        specificity_bonus += 0.1

            base_score = mentioned_count / len(expected) if expected else 0.5
            return min(1.0, base_score + specificity_bonus)

        elif category == 'spatial':
            # Score based on spatial language and accuracy
            spatial_terms = sum(1 for term in expected if term in response_lower)

            # Bonus for specific position mentions
            position_bonus = 0.0
            position_patterns = [
                r'\b\d+\s*pixels?\b', r'\bx\s*[=:]\s*\d+\b', r'\by\s*[=:]\s*\d+\b',
                r'\btop\s+\w+\b', r'\bbottom\s+\w+\b', r'\bleft\s+\w+\b', r'\bright\s+\w+\b'
            ]

            for pattern in position_patterns:
                if re.search(pattern, response_lower):
                    position_bonus += 0.1

            base_score = spatial_terms / len(expected) if expected else 0.0
            return min(1.0, base_score + position_bonus)

        elif category == 'strategy':
            # Score based on action words and game-relevant strategy
            action_words = sum(1 for action in expected if action in response_lower)

            # Bonus for specific, actionable advice
            actionable_bonus = 0.0
            actionable_patterns = [
                r'\bmove\s+\w+\b', r'\baim\s+\w+\b', r'\bhit\s+\w+\b',
                r'\bavoid\s+\w+\b', r'\bshoot\s+\w+\b'
            ]

            for pattern in actionable_patterns:
                if re.search(pattern, response_lower):
                    actionable_bonus += 0.15

            base_score = action_words / len(expected) if expected else 0.0
            return min(1.0, base_score + actionable_bonus)

        return 0.5  # Default neutral score

    def _generate_scoring_reasoning(self,
                                  response: str,
                                  category: str,
                                  game_name: str,
                                  score: float) -> str:
        """Generate human-readable reasoning for the assigned score."""
        if score == 0.0:
            return f"No relevant {category} content detected"
        elif score >= 0.8:
            return f"Excellent {category} understanding demonstrated"
        elif score >= 0.6:
            return f"Good {category} understanding with minor gaps"
        elif score >= 0.4:
            return f"Basic {category} understanding, needs improvement"
        else:
            return f"Limited {category} understanding"

    def evaluate_frame_set(self,
                          frames_and_ground_truth: List[Tuple[np.ndarray, str, Dict[str, Any]]],
                          pipeline_func: Callable[[np.ndarray, str], str]) -> Dict[str, Any]:
        """
        Evaluate a set of frames and compute aggregate statistics.

        Args:
            frames_and_ground_truth: List of (frame, game_name, ground_truth) tuples
            pipeline_func: Pipeline function to evaluate

        Returns:
            Comprehensive evaluation results with category breakdowns
        """
        all_results = []

        print(f"Evaluating {len(frames_and_ground_truth)} frames...")

        for i, (frame, game_name, ground_truth) in enumerate(frames_and_ground_truth):
            print(f"  Evaluating frame {i+1}/{len(frames_and_ground_truth)} ({game_name})")

            frame_results = self.evaluate_single_frame(frame, game_name, pipeline_func, ground_truth)
            all_results.extend(frame_results)

        # Aggregate results by category
        category_stats = {}
        for category in ['visual', 'spatial', 'strategy', 'identification']:
            category_results = [r for r in all_results if r.category == category]

            if category_results:
                scores = [r.score for r in category_results]
                category_stats[category] = {
                    'count': len(scores),
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores),
                    'min_score': np.min(scores),
                    'max_score': np.max(scores),
                    'scores': scores
                }

        # Overall statistics
        all_scores = [r.score for r in all_results]
        overall_stats = {
            'total_evaluations': len(all_results),
            'mean_score': np.mean(all_scores),
            'std_score': np.std(all_scores),
            'category_breakdown': category_stats
        }

        return {
            'overall_statistics': overall_stats,
            'detailed_results': all_results,
            'evaluation_summary': self._generate_evaluation_summary(category_stats)
        }

    def _generate_evaluation_summary(self, category_stats: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
        """Generate summary interpretations of evaluation results."""
        summary = {}

        for category, stats in category_stats.items():
            mean_score = stats['mean_score']

            if mean_score >= 0.8:
                performance = "Excellent"
            elif mean_score >= 0.6:
                performance = "Good"
            elif mean_score >= 0.4:
                performance = "Fair"
            else:
                performance = "Poor"

            summary[category] = f"{performance} ({mean_score:.1%} accuracy, n={stats['count']})"

        return summary

    def compare_pipelines(self,
                         baseline_results: Dict[str, Any],
                         comparison_results: Dict[str, Any],
                         baseline_name: str = "Vision-only",
                         comparison_name: str = "Vision+Symbol") -> Dict[str, Any]:
        """
        Compare two pipeline evaluation results.

        Args:
            baseline_results: Results from baseline pipeline (typically Vision-only)
            comparison_results: Results from comparison pipeline (typically Vision+Symbol)
            baseline_name: Name for baseline pipeline
            comparison_name: Name for comparison pipeline

        Returns:
            Detailed comparison analysis showing improvements and performance gaps
        """
        comparison = {
            'pipelines': {
                'baseline': baseline_name,
                'comparison': comparison_name
            },
            'category_comparison': {},
            'overall_comparison': {},
            'statistical_significance': {}
        }

        baseline_categories = baseline_results['overall_statistics']['category_breakdown']
        comparison_categories = comparison_results['overall_statistics']['category_breakdown']

        # Compare each category
        for category in ['visual', 'spatial', 'strategy', 'identification']:
            if category in baseline_categories and category in comparison_categories:
                baseline_score = baseline_categories[category]['mean_score']
                comparison_score = comparison_categories[category]['mean_score']

                improvement = comparison_score - baseline_score
                improvement_pct = (improvement / baseline_score * 100) if baseline_score > 0 else 0

                comparison['category_comparison'][category] = {
                    'baseline_score': baseline_score,
                    'comparison_score': comparison_score,
                    'absolute_improvement': improvement,
                    'percent_improvement': improvement_pct,
                    'interpretation': self._interpret_improvement(improvement, category)
                }

        # Overall comparison
        baseline_overall = baseline_results['overall_statistics']['mean_score']
        comparison_overall = comparison_results['overall_statistics']['mean_score']

        overall_improvement = comparison_overall - baseline_overall
        overall_improvement_pct = (overall_improvement / baseline_overall * 100) if baseline_overall > 0 else 0

        comparison['overall_comparison'] = {
            'baseline_score': baseline_overall,
            'comparison_score': comparison_overall,
            'absolute_improvement': overall_improvement,
            'percent_improvement': overall_improvement_pct,
            'interpretation': self._interpret_improvement(overall_improvement, 'overall')
        }

        return comparison

    def _interpret_improvement(self, improvement: float, category: str) -> str:
        """Interpret the significance of score improvements."""
        if improvement >= 0.15:
            return f"Major improvement in {category} reasoning"
        elif improvement >= 0.05:
            return f"Moderate improvement in {category} reasoning"
        elif improvement >= 0.01:
            return f"Slight improvement in {category} reasoning"
        elif improvement >= -0.01:
            return f"No significant change in {category} reasoning"
        else:
            return f"Performance degradation in {category} reasoning"

    def export_results(self,
                      results: Dict[str, Any],
                      filename: str,
                      include_responses: bool = False):
        """
        Export evaluation results to JSON.

        Args:
            results: Evaluation results dictionary
            filename: Output filename
            include_responses: Whether to include full text responses (can be large)
        """
        export_data = results.copy()

        if not include_responses and 'detailed_results' in export_data:
            # Remove response text to reduce file size
            for result in export_data['detailed_results']:
                if hasattr(result, 'response'):
                    result.response = "[Response text excluded from export]"

        # Convert DiagnosticResult objects to dictionaries
        if 'detailed_results' in export_data:
            detailed_results = []
            for result in export_data['detailed_results']:
                if isinstance(result, DiagnosticResult):
                    detailed_results.append({
                        'category': result.category,
                        'prompt': result.prompt,
                        'response': result.response if include_responses else "[Excluded]",
                        'score': result.score,
                        'reasoning': result.reasoning
                    })
                else:
                    detailed_results.append(result)
            export_data['detailed_results'] = detailed_results

        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"Results exported to {filename}")


def create_mock_pipeline_func(pipeline_type: str = "vision_only") -> Callable[[np.ndarray, str], str]:
    """
    Create a mock pipeline function for testing.

    In practice, this would be replaced with your actual Vision-only, Vision+Symbol,
    or Privileged-Symbol pipelines.
    """
    def mock_vision_only(frame: np.ndarray, prompt: str) -> str:
        """Mock vision-only pipeline - simulates poor spatial reasoning."""
        if "identify all key elements" in prompt.lower():
            return "I can see some objects in the image including what appears to be game elements."
        elif "where are the key elements located" in prompt.lower():
            return "The elements are positioned in various locations across the screen."
        elif "ideal next move" in prompt.lower():
            return "Move in a strategic direction to achieve the game objective."
        elif "identify the game name" in prompt.lower():
            return "This appears to be an Atari game."
        return "Unable to process this request."

    def mock_vision_symbol(frame: np.ndarray, prompt: str) -> str:
        """Mock vision+symbol pipeline - simulates better spatial reasoning."""
        if "identify all key elements" in prompt.lower():
            return "I can identify a player paddle at coordinates (140, 120), a ball at (75, 100), and an opponent paddle at (20, 115)."
        elif "where are the key elements located" in prompt.lower():
            return "The player paddle is on the right side at x=140, the ball is in the center-left at (75, 100), and the opponent paddle is on the far left at x=20."
        elif "ideal next move" in prompt.lower():
            return "Move the paddle upward to intercept the ball's trajectory and return it toward the opponent's weak side."
        elif "identify the game name" in prompt.lower():
            return "This is Pong, a classic paddle-based game."
        return "Processing with symbolic grounding enabled."

    if pipeline_type == "vision_symbol":
        return mock_vision_symbol
    else:
        return mock_vision_only


if __name__ == "__main__":
    # Test the diagnostic framework
    evaluator = AtariGPTDiagnostic()

    # Create mock frame data
    extractor = OCAtariGroundTruth('Pong')

    # Generate test frames
    test_frames = []
    for i in range(3):
        for _ in range(10):
            action = extractor.env.action_space.sample()
            extractor.step(action)

        frame, objects = extractor.get_frame_and_objects()
        ground_truth = {
            'objects': [obj.to_dict() for obj in objects],
            'spatial_relationships': extractor.get_spatial_relationships()
        }
        test_frames.append((frame, 'Pong', ground_truth))

    extractor.close()

    # Test both pipeline types
    vision_only_func = create_mock_pipeline_func("vision_only")
    vision_symbol_func = create_mock_pipeline_func("vision_symbol")

    print("Testing Vision-only pipeline...")
    vision_only_results = evaluator.evaluate_frame_set(test_frames, vision_only_func)

    print("\nTesting Vision+Symbol pipeline...")
    vision_symbol_results = evaluator.evaluate_frame_set(test_frames, vision_symbol_func)

    # Compare results
    print("\nComparing pipelines...")
    comparison = evaluator.compare_pipelines(vision_only_results, vision_symbol_results)

    # Print summary
    print("\n" + "="*60)
    print("DIAGNOSTIC EVALUATION SUMMARY")
    print("="*60)

    for category, comp in comparison['category_comparison'].items():
        improvement = comp['percent_improvement']
        print(f"{category.capitalize():15}: {comp['baseline_score']:.1%} → {comp['comparison_score']:.1%} ({improvement:+.1f}%)")

    overall = comparison['overall_comparison']
    print(f"{'Overall':15}: {overall['baseline_score']:.1%} → {overall['comparison_score']:.1%} ({overall['percent_improvement']:+.1f}%)")

    # Export results
    evaluator.export_results(vision_only_results, "vision_only_diagnostic.json")
    evaluator.export_results(vision_symbol_results, "vision_symbol_diagnostic.json")

    with open("pipeline_comparison.json", 'w') as f:
        json.dump(comparison, f, indent=2)

    print("\nResults exported to JSON files.")