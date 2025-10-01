"""
Rule-based scorer.
Fast, deterministic validation using keyword matching and coordinate checking.
"""

import re
from typing import Dict, Any, List
from .base_scorer import BaseScorer, ScoringResult
from ..config import (
    OBJECT_SYNONYMS,
    INVALID_ACTIONS,
    COORDINATE_TOLERANCE_PIXELS,
    RULE_BASED_WEIGHTS
)
from ..utils.helpers import extract_coordinates, check_coordinate_accuracy


class RuleBasedScorer(BaseScorer):
    """
    Rule-based scorer using keyword matching and heuristics.
    Fast and deterministic, but limited to pattern matching.
    """

    def score(
        self,
        response: str,
        task_type: str,
        ground_truth: Dict[str, Any],
        game_name: str
    ) -> ScoringResult:
        """Score response using rule-based methods."""
        self._validate_task_type(task_type)

        if task_type == 'visual':
            return self._score_visual(response, ground_truth)
        elif task_type == 'spatial':
            return self._score_spatial(response, ground_truth)
        elif task_type == 'strategy':
            return self._score_strategy(response, ground_truth, game_name)
        elif task_type == 'identification':
            return self._score_identification(response, game_name)

    def _score_visual(
        self,
        response: str,
        ground_truth: Dict[str, Any]
    ) -> ScoringResult:
        """
        Score visual understanding (object detection).

        Scoring:
        - Correctly identified objects: +points
        - Hallucinated objects: -points
        - Specificity (coordinates, colors): +bonus
        """
        objects = self._extract_objects(ground_truth)
        response_lower = response.lower()

        if not response or response.startswith("ERROR"):
            return ScoringResult(
                score=0.0,
                confidence=1.0,
                reasoning="No valid response provided",
                details={'detected': 0, 'expected': len(objects), 'hallucinations': 0}
            )

        # Count detected objects
        detected_count = 0
        detected_objects = []

        for obj in objects:
            category = obj.get('category', '')
            synonyms = OBJECT_SYNONYMS.get(category, [category.lower()])

            if any(syn in response_lower for syn in synonyms):
                detected_count += 1
                detected_objects.append(category)

        # Check for false positives (hallucinations)
        hallucinations = self._count_hallucinations(response_lower, objects)

        # Base score: detection rate
        detection_rate = detected_count / len(objects) if objects else 0.5
        hallucination_penalty = 0.5 * hallucinations / max(len(objects), 1)

        # Bonus for specificity (mentioning coordinates, colors)
        specificity_bonus = self._calculate_specificity_bonus(response, objects)

        # Combine scores
        weights = RULE_BASED_WEIGHTS['visual']
        final_score = (
            weights['object_detection'] * detection_rate -
            weights['false_positive_penalty'] * hallucination_penalty +
            weights['specificity_bonus'] * specificity_bonus
        )

        final_score = self._clamp_score(final_score)

        # Generate reasoning
        reasoning = f"Detected {detected_count}/{len(objects)} objects"
        if hallucinations > 0:
            reasoning += f", {hallucinations} hallucinations"
        if specificity_bonus > 0:
            reasoning += f", specificity bonus: {specificity_bonus:.2f}"

        return ScoringResult(
            score=final_score,
            confidence=0.8,  # Rule-based has medium confidence
            reasoning=reasoning,
            details={
                'detected': detected_count,
                'expected': len(objects),
                'detected_objects': detected_objects,
                'hallucinations': hallucinations,
                'specificity_bonus': specificity_bonus
            }
        )

    def _count_hallucinations(
        self,
        response_lower: str,
        objects: List[Dict[str, Any]]
    ) -> int:
        """Count hallucinated objects (mentioned but not in ground truth)."""
        # Get all valid object categories
        valid_categories = set()
        for obj in objects:
            category = obj.get('category', '')
            valid_categories.add(category.lower())
            valid_categories.update(OBJECT_SYNONYMS.get(category, []))

        # Common object terms that might be mentioned
        common_objects = [
            'paddle', 'ball', 'brick', 'alien', 'ship', 'bullet',
            'player', 'enemy', 'opponent', 'shield', 'barrier'
        ]

        hallucinations = 0
        for term in common_objects:
            if term in response_lower and term not in valid_categories:
                # Check if any valid synonym matches
                is_hallucination = True
                for valid in valid_categories:
                    if term in valid or valid in term:
                        is_hallucination = False
                        break

                if is_hallucination:
                    hallucinations += 1

        return hallucinations

    def _calculate_specificity_bonus(
        self,
        response: str,
        objects: List[Dict[str, Any]]
    ) -> float:
        """Calculate bonus for specific details (coordinates, colors)."""
        bonus = 0.0

        # Check for coordinate mentions
        coords = extract_coordinates(response)
        if coords:
            accuracy = check_coordinate_accuracy(
                coords,
                objects,
                tolerance=COORDINATE_TOLERANCE_PIXELS
            )
            bonus += 0.3 * accuracy

        # Check for color mentions
        color_terms = ['red', 'blue', 'green', 'white', 'orange', 'yellow', 'purple']
        if any(color in response.lower() for color in color_terms):
            bonus += 0.2

        # Check for size mentions
        size_terms = ['large', 'small', 'big', 'tiny', 'wide', 'tall']
        if any(size in response.lower() for size in size_terms):
            bonus += 0.1

        return min(1.0, bonus)

    def _score_spatial(
        self,
        response: str,
        ground_truth: Dict[str, Any]
    ) -> ScoringResult:
        """
        Score spatial reasoning (relative positions).

        Scoring:
        - Correct relative positions (left/right/above/below): 40%
        - Coordinate accuracy: 30%
        - Distance awareness: 30%
        """
        relationships = self._extract_spatial_relationships(ground_truth)
        objects = self._extract_objects(ground_truth)
        response_lower = response.lower()

        if not response or response.startswith("ERROR"):
            return ScoringResult(
                score=0.0,
                confidence=1.0,
                reasoning="No valid response provided"
            )

        weights = RULE_BASED_WEIGHTS['spatial']

        # 1. Relative position accuracy (40%)
        relative_score = self._check_relative_positions(response_lower, relationships)

        # 2. Coordinate accuracy (30%)
        coords = extract_coordinates(response)
        if coords:
            coordinate_score = check_coordinate_accuracy(coords, objects, COORDINATE_TOLERANCE_PIXELS)
        else:
            coordinate_score = 0.0

        # 3. Distance awareness (30%)
        distance_score = self._check_distance_awareness(response_lower, relationships)

        # Combine with weights
        final_score = (
            weights['relative_position'] * relative_score +
            weights['coordinate_accuracy'] * coordinate_score +
            weights['distance_awareness'] * distance_score
        )

        reasoning = f"Relative: {relative_score:.2f}, Coord: {coordinate_score:.2f}, Distance: {distance_score:.2f}"

        return ScoringResult(
            score=self._clamp_score(final_score),
            confidence=0.75,
            reasoning=reasoning,
            details={
                'relative_position_score': relative_score,
                'coordinate_score': coordinate_score,
                'distance_score': distance_score
            }
        )

    def _check_relative_positions(
        self,
        response_lower: str,
        relationships: Dict[str, Any]
    ) -> float:
        """Check if relative positions (left/right/above/below) are correct."""
        score = 0.0
        checks = 0

        categories = relationships.get('object_categories', [])
        if not categories:
            return 0.0

        # Check leftmost/rightmost
        if relationships.get('leftmost_object') is not None:
            leftmost_idx = relationships['leftmost_object']
            leftmost_cat = categories[leftmost_idx].lower()

            if leftmost_cat in response_lower and 'left' in response_lower:
                score += 1.0
            checks += 1

        if relationships.get('rightmost_object') is not None:
            rightmost_idx = relationships['rightmost_object']
            rightmost_cat = categories[rightmost_idx].lower()

            if rightmost_cat in response_lower and 'right' in response_lower:
                score += 1.0
            checks += 1

        # Check topmost/bottommost
        if relationships.get('topmost_object') is not None:
            topmost_idx = relationships['topmost_object']
            topmost_cat = categories[topmost_idx].lower()

            if topmost_cat in response_lower and ('top' in response_lower or 'above' in response_lower):
                score += 1.0
            checks += 1

        if relationships.get('bottommost_object') is not None:
            bottommost_idx = relationships['bottommost_object']
            bottommost_cat = categories[bottommost_idx].lower()

            if bottommost_cat in response_lower and ('bottom' in response_lower or 'below' in response_lower):
                score += 1.0
            checks += 1

        return score / checks if checks > 0 else 0.0

    def _check_distance_awareness(
        self,
        response_lower: str,
        relationships: Dict[str, Any]
    ) -> float:
        """Check if distance awareness (near/far) is mentioned."""
        distance_terms = [
            'near', 'far', 'close', 'distant', 'approach', 'away',
            'distance', 'pixels', 'apart', 'between'
        ]

        if any(term in response_lower for term in distance_terms):
            return 1.0

        # Partial credit for spatial descriptions
        spatial_terms = ['center', 'middle', 'edge', 'side', 'corner']
        if any(term in response_lower for term in spatial_terms):
            return 0.5

        return 0.0

    def _score_strategy(
        self,
        response: str,
        ground_truth: Dict[str, Any],
        game_name: str
    ) -> ScoringResult:
        """
        Score strategic reasoning (next action).

        Scoring:
        - Action validity: 30%
        - Action optimality: 40%
        - Justification: 30%
        """
        response_lower = response.lower()

        if not response or response.startswith("ERROR"):
            return ScoringResult(
                score=0.0,
                confidence=1.0,
                reasoning="No valid response provided"
            )

        # Check for invalid actions first
        invalid_actions = INVALID_ACTIONS.get(game_name, [])
        for invalid in invalid_actions:
            if invalid in response_lower:
                return ScoringResult(
                    score=0.0,
                    confidence=1.0,
                    reasoning=f"Invalid action for {game_name}: {invalid}",
                    details={'invalid_action': invalid}
                )

        # Calculate optimal action
        optimal_action = self._calculate_optimal_action(ground_truth, game_name)

        weights = RULE_BASED_WEIGHTS['strategy']

        # Check if mentions valid action
        action_space = self._extract_action_space(ground_truth)
        valid_actions = [a.lower() for a in action_space.get('actions', [])]

        action_validity_score = 0.0
        mentioned_action = None
        for action in valid_actions:
            if action.lower() in response_lower:
                action_validity_score = 1.0
                mentioned_action = action
                break

        # Check optimality
        if optimal_action and mentioned_action:
            if optimal_action.lower() in mentioned_action.lower():
                optimality_score = 1.0
            else:
                optimality_score = 0.6  # Valid but not optimal
        else:
            optimality_score = 0.3  # Has some strategy mentioned

        # Check justification quality
        justification_score = self._check_justification(response_lower)

        # Combine scores
        final_score = (
            weights['action_validity'] * action_validity_score +
            weights['action_optimality'] * optimality_score +
            weights['justification'] * justification_score
        )

        reasoning = f"Validity: {action_validity_score:.2f}, Optimality: {optimality_score:.2f}, Justification: {justification_score:.2f}"

        return ScoringResult(
            score=self._clamp_score(final_score),
            confidence=0.6,  # Lower confidence for strategy (hard to validate automatically)
            reasoning=reasoning,
            details={
                'optimal_action': optimal_action,
                'mentioned_action': mentioned_action,
                'validity_score': action_validity_score,
                'optimality_score': optimality_score,
                'justification_score': justification_score
            }
        )

    def _calculate_optimal_action(
        self,
        ground_truth: Dict[str, Any],
        game_name: str
    ) -> str:
        """Calculate optimal action based on game state (simplified heuristic)."""
        objects = self._extract_objects(ground_truth)

        if game_name == 'Pong':
            # Simple heuristic: move paddle toward ball
            player = next((o for o in objects if o['category'] == 'Player'), None)
            ball = next((o for o in objects if o['category'] == 'Ball'), None)

            if player and ball:
                ball_y = ball['position'][1]
                player_y = player['position'][1]

                if ball_y < player_y:
                    return 'UP'
                elif ball_y > player_y:
                    return 'DOWN'

        # Default: return generic action
        return 'OPTIMAL_ACTION'

    def _check_justification(self, response_lower: str) -> float:
        """Check quality of justification/reasoning."""
        # Look for reasoning keywords
        reasoning_terms = [
            'because', 'since', 'to', 'in order to',
            'intercept', 'avoid', 'aim', 'position', 'defend'
        ]

        score = 0.0
        for term in reasoning_terms:
            if term in response_lower:
                score += 0.2

        return min(1.0, score)

    def _score_identification(
        self,
        response: str,
        game_name: str
    ) -> ScoringResult:
        """Score game identification (simple string matching)."""
        response_lower = response.lower()
        game_lower = game_name.lower()

        # Exact or close match
        if game_lower in response_lower:
            return ScoringResult(
                score=1.0,
                confidence=1.0,
                reasoning=f"Correctly identified as {game_name}"
            )

        # Check for variants
        variants = {
            'Pong': ['pong', 'ping pong'],
            'Breakout': ['breakout', 'break out', 'brick breaker'],
            'SpaceInvaders': ['space invaders', 'spaceinvaders', 'invaders']
        }

        for variant in variants.get(game_name, []):
            if variant in response_lower:
                return ScoringResult(
                    score=1.0,
                    confidence=1.0,
                    reasoning=f"Correctly identified variant: {variant}"
                )

        return ScoringResult(
            score=0.0,
            confidence=1.0,
            reasoning=f"Did not identify {game_name}"
        )
