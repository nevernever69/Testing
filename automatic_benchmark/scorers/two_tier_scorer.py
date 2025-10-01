"""
Two-tier evaluation scorer for fair object detection evaluation.

Separates core objects (critical) from secondary objects (bonus) and weights them appropriately.
"""

from typing import Dict, Any, List
from automatic_benchmark.config.object_importance import (
    get_object_tier,
    calculate_tiered_score,
    get_core_objects,
    get_secondary_objects
)


class TwoTierScorer:
    """
    Evaluates object detection with two-tier weighting:
    - Core objects: 70% weight (critical for gameplay)
    - Secondary objects: 30% weight (bonus for completeness)
    """

    def __init__(self, core_weight: float = 0.7, secondary_weight: float = 0.3):
        """
        Initialize two-tier scorer.

        Args:
            core_weight: Weight for core objects (default 0.7 = 70%)
            secondary_weight: Weight for secondary objects (default 0.3 = 30%)
        """
        self.core_weight = core_weight
        self.secondary_weight = secondary_weight

    def evaluate(self,
                response: str,
                ground_truth: Dict[str, Any],
                game_name: str,
                task_type: str = 'visual') -> Dict[str, Any]:
        """
        Evaluate response with two-tier object importance.

        Args:
            response: VLM response text
            ground_truth: Ground truth data with annotated objects
            game_name: Name of the game
            task_type: Type of task being evaluated

        Returns:
            {
                'core_score': float,
                'secondary_score': float,
                'final_score': float,
                'core_details': {...},
                'secondary_details': {...},
                'reasoning': str
            }
        """
        # Get objects from ground truth
        objects = ground_truth.get('ocatari_data', {}).get('objects', [])
        if not objects:
            return {
                'core_score': 0.0,
                'secondary_score': 0.0,
                'final_score': 0.0,
                'reasoning': 'No ground truth objects available'
            }

        # Separate objects by tier
        core_objs = [obj for obj in objects if obj.get('tier') == 'core']
        secondary_objs = [obj for obj in objects if obj.get('tier') in ['secondary', 'unknown']]

        # Evaluate core objects
        core_result = self._evaluate_object_tier(response, core_objs, 'core', task_type)

        # Evaluate secondary objects
        secondary_result = self._evaluate_object_tier(response, secondary_objs, 'secondary', task_type)

        # Calculate weighted final score
        final_score = calculate_tiered_score(
            core_result['score'],
            secondary_result['score'],
            self.core_weight,
            self.secondary_weight
        )

        return {
            'core_score': core_result['score'],
            'secondary_score': secondary_result['score'],
            'final_score': final_score,
            'core_details': core_result,
            'secondary_details': secondary_result,
            'reasoning': self._generate_reasoning(core_result, secondary_result, final_score),
            'breakdown': {
                'core_objects': len(core_objs),
                'secondary_objects': len(secondary_objs),
                'core_detected': core_result.get('detected_count', 0),
                'secondary_detected': secondary_result.get('detected_count', 0)
            }
        }

    def _evaluate_object_tier(self,
                              response: str,
                              objects: List[Dict],
                              tier: str,
                              task_type: str) -> Dict[str, Any]:
        """
        Evaluate detection for a specific tier of objects.

        Args:
            response: VLM response
            objects: List of objects in this tier
            tier: 'core' or 'secondary'
            task_type: Type of task

        Returns:
            {
                'score': float,
                'detected': List[str],
                'missed': List[str],
                'detected_count': int
            }
        """
        if not objects:
            return {
                'score': 1.0,  # Perfect score if no objects in this tier
                'detected': [],
                'missed': [],
                'detected_count': 0
            }

        response_lower = response.lower()
        detected = []
        missed = []

        for obj in objects:
            category = obj['category']

            # Check if object category is mentioned in response
            # Use flexible matching: partial string match
            category_variations = self._get_category_variations(category)

            found = any(var in response_lower for var in category_variations)

            if found:
                detected.append(category)
            else:
                missed.append(category)

        # Calculate score based on detection rate
        detection_rate = len(detected) / len(objects) if objects else 0.0

        return {
            'score': detection_rate,
            'detected': detected,
            'missed': missed,
            'detected_count': len(detected),
            'total_count': len(objects)
        }

    def _get_category_variations(self, category: str) -> List[str]:
        """
        Generate variations of an object category for flexible matching.

        Examples:
            'Player' → ['player', 'paddle', 'user', 'your']
            'Enemy' → ['enemy', 'opponent', 'ai', 'computer']
            'Ball' → ['ball', 'projectile']
        """
        category_lower = category.lower()
        variations = [category_lower]

        # Common variations
        variation_map = {
            'player': ['paddle', 'user', 'your', 'right_paddle', 'player_paddle'],
            'enemy': ['opponent', 'ai', 'computer', 'left_paddle', 'enemy_paddle'],
            'ball': ['projectile', 'sphere'],
            'alien': ['enemy', 'invader', 'ufo'],
            'bullet': ['projectile', 'shot', 'laser'],
            'brick': ['block', 'tile'],
            'score': ['points', 'score_display'],
            'lives': ['life', 'remaining']
        }

        for key, vars in variation_map.items():
            if key in category_lower:
                variations.extend(vars)

        return variations

    def _generate_reasoning(self,
                           core_result: Dict,
                           secondary_result: Dict,
                           final_score: float) -> str:
        """Generate human-readable reasoning for the score."""
        reasoning_parts = []

        # Core objects
        core_detected = core_result.get('detected_count', 0)
        core_total = core_result.get('total_count', 0)
        if core_total > 0:
            reasoning_parts.append(
                f"Core objects: {core_detected}/{core_total} detected ({core_result['score']:.1%})"
            )
            if core_result.get('missed'):
                reasoning_parts.append(f"  Missed core: {', '.join(core_result['missed'])}")

        # Secondary objects
        sec_detected = secondary_result.get('detected_count', 0)
        sec_total = secondary_result.get('total_count', 0)
        if sec_total > 0:
            reasoning_parts.append(
                f"Secondary objects: {sec_detected}/{sec_total} detected ({secondary_result['score']:.1%})"
            )

        # Final weighted score
        reasoning_parts.append(
            f"Weighted score: {final_score:.1%} "
            f"(core={self.core_weight:.0%}, secondary={self.secondary_weight:.0%})"
        )

        return "; ".join(reasoning_parts)
