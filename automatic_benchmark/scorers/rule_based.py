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
    RULE_BASED_WEIGHTS,
    STRICTER_PENALTIES,
    MULTIPART_WEIGHTS
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
        Score visual understanding (object detection) with multi-part evaluation.

        Multi-part scoring:
        - Part 1 (Objects 40%): Identify all objects present
        - Part 2 (Counts 35%): Accurate counts for each object type
        - Part 3 (Properties 25%): Describe visual properties

        Stricter penalties applied for errors.
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

        # Multi-part weights
        weights = MULTIPART_WEIGHTS['visual']
        penalties = STRICTER_PENALTIES['visual']

        # Part 1: Object Detection (40%)
        detected_count = 0
        detected_objects = []
        core_objects = [obj for obj in objects if obj.get('tier') == 'core']

        for obj in objects:
            category = obj.get('category', '')
            synonyms = OBJECT_SYNONYMS.get(category, [category.lower()])

            if any(syn in response_lower for syn in synonyms):
                detected_count += 1
                detected_objects.append(category)

        # Calculate detection score with stricter penalties
        objects_score = 0.0
        if detected_count == len(objects):
            objects_score = weights['objects']
        else:
            # Missing core objects penalty
            missing_core = len([obj for obj in core_objects if obj['category'] not in detected_objects])
            objects_score = weights['objects'] * (detected_count / len(objects))
            objects_score += missing_core * penalties['missing_core_object']

        # Part 2: Hallucinations (reduce from Part 1)
        hallucinations = self._count_hallucinations(response_lower, objects)
        hallucination_penalty = hallucinations * penalties['hallucinated_object']
        objects_score += hallucination_penalty

        # Part 3: Counts (35%) - Check if counts mentioned
        counts_score = 0.0
        if self._has_count_information(response_lower):
            counts_score = weights['counts']
        else:
            counts_score = penalties['wrong_count']  # No counts = wrong counts

        # Part 4: Properties (25%) - Check if visual properties described
        properties_score = 0.0
        if self._has_visual_properties(response_lower):
            properties_score = weights['properties']
        else:
            properties_score = penalties['missing_properties']

        # Check for incomplete answer (skipped sections)
        incomplete_penalty = 0.0
        if len(response.split()) < 10:  # Very short response likely incomplete
            incomplete_penalty = penalties['incomplete_answer']

        # Combine all parts
        final_score = objects_score + counts_score + properties_score + incomplete_penalty
        final_score = self._clamp_score(final_score)

        # Generate reasoning
        reasoning = f"Objects: {detected_count}/{len(objects)}"
        if hallucinations > 0:
            reasoning += f", {hallucinations} hallucinations"
        reasoning += f", Counts: {'✓' if counts_score > 0 else '✗'}"
        reasoning += f", Properties: {'✓' if properties_score > 0 else '✗'}"

        return ScoringResult(
            score=final_score,
            confidence=0.8,
            reasoning=reasoning,
            details={
                'detected': detected_count,
                'expected': len(objects),
                'detected_objects': detected_objects,
                'hallucinations': hallucinations,
                'objects_score': objects_score,
                'counts_score': counts_score,
                'properties_score': properties_score
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

    def _has_count_information(self, response_lower: str) -> bool:
        """Check if response mentions object counts."""
        # Look for number words or digits followed by object terms
        count_patterns = [
            r'\d+\s+(object|paddle|ball|brick|alien|ship|bullet|shield)',
            r'(one|two|three|four|five|six|seven|eight|nine|ten)\s+(object|paddle|ball|brick|alien|ship|bullet|shield)',
            r'(multiple|several|many|few)\s+(object|paddle|ball|brick|alien|ship|bullet|shield)',
            r'count',
            r'how many',
            r'number of'
        ]
        return any(re.search(pattern, response_lower) for pattern in count_patterns)

    def _has_visual_properties(self, response_lower: str) -> bool:
        """Check if response describes visual properties."""
        # Look for color, size, shape, state descriptions
        property_terms = [
            'red', 'blue', 'green', 'white', 'orange', 'yellow', 'purple', 'black', 'color',
            'large', 'small', 'big', 'tiny', 'wide', 'tall', 'size',
            'square', 'rectangular', 'circular', 'round', 'shape',
            'bright', 'dark', 'moving', 'static', 'flashing'
        ]
        return any(term in response_lower for term in property_terms)

    def _score_spatial(
        self,
        response: str,
        ground_truth: Dict[str, Any]
    ) -> ScoringResult:
        """
        Score spatial reasoning with multi-part evaluation.

        Multi-part scoring:
        - Part 1 (Absolute Positions 25%): Where objects are on screen
        - Part 2 (Relative Positions 35%): Object relationships
        - Part 3 (Distances 20%): Near/far assessments
        - Part 4 (Alignment 20%): Aligned objects

        Stricter penalties applied for errors.
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

        # Multi-part weights
        weights = MULTIPART_WEIGHTS['spatial']
        penalties = STRICTER_PENALTIES['spatial']

        # Part 1: Absolute Positions (25%)
        absolute_score = 0.0
        if self._has_absolute_positions(response_lower):
            absolute_score = weights['absolute_positions']
        else:
            absolute_score = penalties['no_absolute_positions']

        # Part 2: Relative Positions (35%)
        relative_score = 0.0
        has_relative = self._has_relative_positions(response_lower)
        if has_relative:
            # Check if relationships are correct
            correct_relationships = self._check_relative_positions(response_lower, relationships)
            relative_score = weights['relative_positions'] * correct_relationships

            # Apply penalty for wrong directions (detected separately)
            wrong_directions = self._count_wrong_directions(response_lower, relationships)
            relative_score += wrong_directions * penalties['wrong_direction']
        else:
            relative_score = penalties['no_relative_positions']

        # Part 3: Distances (20%)
        distance_score = 0.0
        if self._has_distance_information(response_lower):
            distance_score = weights['distances']
        else:
            # No penalty for missing distance, but no points
            distance_score = 0.0

        # Part 4: Alignment (20%)
        alignment_score = 0.0
        if self._has_alignment_information(response_lower):
            alignment_score = weights['alignment']
        # No penalty for missing alignment

        # Check for incomplete answer
        incomplete_penalty = 0.0
        if len(response.split()) < 15:  # Very short response
            incomplete_penalty = penalties['incomplete_answer']

        # Combine all parts
        final_score = absolute_score + relative_score + distance_score + alignment_score + incomplete_penalty
        final_score = self._clamp_score(final_score)

        reasoning = f"Abs: {'✓' if absolute_score > 0 else '✗'}, Rel: {'✓' if relative_score > 0 else '✗'}, Dist: {'✓' if distance_score > 0 else '✗'}, Align: {'✓' if alignment_score > 0 else '✗'}"

        return ScoringResult(
            score=final_score,
            confidence=0.75,
            reasoning=reasoning,
            details={
                'absolute_score': absolute_score,
                'relative_score': relative_score,
                'distance_score': distance_score,
                'alignment_score': alignment_score
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

    def _has_absolute_positions(self, response_lower: str) -> bool:
        """Check if response mentions absolute positions on screen."""
        position_terms = [
            'top', 'bottom', 'left', 'right', 'center', 'middle',
            'upper', 'lower', 'side', 'corner', 'edge'
        ]
        return any(term in response_lower for term in position_terms)

    def _has_relative_positions(self, response_lower: str) -> bool:
        """Check if response mentions relative positions between objects."""
        relative_terms = [
            'above', 'below', 'left of', 'right of', 'between',
            'next to', 'beside', 'near', 'close to', 'far from'
        ]
        return any(term in response_lower for term in relative_terms)

    def _count_wrong_directions(self, response_lower: str, relationships: Dict[str, Any]) -> int:
        """Count wrong directional claims (simplified heuristic)."""
        # This is a simplified version - the LLM judge will do better
        # For now, return 0 (no penalty from rule-based)
        return 0

    def _has_distance_information(self, response_lower: str) -> bool:
        """Check if response mentions distance information."""
        distance_terms = [
            'near', 'far', 'close', 'distant', 'distance',
            'apart', 'between', 'away from', 'pixels'
        ]
        return any(term in response_lower for term in distance_terms)

    def _has_alignment_information(self, response_lower: str) -> bool:
        """Check if response mentions alignment."""
        alignment_terms = [
            'aligned', 'vertical', 'horizontal', 'same line',
            'same row', 'same column', 'in line'
        ]
        return any(term in response_lower for term in alignment_terms)

    def _score_strategy(
        self,
        response: str,
        ground_truth: Dict[str, Any],
        game_name: str
    ) -> ScoringResult:
        """
        Score strategic reasoning with multi-part evaluation.

        Multi-part scoring:
        - Part 1 (Situation Analysis 30%): Did they analyze game state?
        - Part 2 (Action 40%): Did they give specific action?
        - Part 3 (Justification 30%): Did they explain why?

        Stricter penalties applied for errors.
        """
        response_lower = response.lower()

        if not response or response.startswith("ERROR"):
            return ScoringResult(
                score=0.0,
                confidence=1.0,
                reasoning="No valid response provided"
            )

        # Multi-part weights
        weights = MULTIPART_WEIGHTS['strategy']
        penalties = STRICTER_PENALTIES['strategy']

        # Check for invalid actions first (AUTO-FAIL)
        invalid_actions = INVALID_ACTIONS.get(game_name, [])
        for invalid in invalid_actions:
            if invalid in response_lower:
                return ScoringResult(
                    score=0.0,
                    confidence=1.0,
                    reasoning=f"Invalid action for {game_name}: {invalid}",
                    details={'invalid_action': invalid}
                )

        # Part 1: Situation Analysis (30%)
        situation_score = 0.0
        if self._has_situation_analysis(response_lower):
            situation_score = weights['situation_analysis']
        else:
            situation_score = penalties['no_situation_analysis']

        # Part 2: Action (40%)
        action_score = 0.0
        action_space = self._extract_action_space(ground_truth)
        valid_actions = [a.lower() for a in action_space.get('actions', [])]

        mentioned_action = None
        for action in valid_actions:
            if action.lower() in response_lower:
                mentioned_action = action
                break

        if mentioned_action:
            # Check if optimal
            optimal_action = self._calculate_optimal_action(ground_truth, game_name)
            if optimal_action and optimal_action.lower() in mentioned_action.lower():
                action_score = weights['action']  # Optimal
            else:
                action_score = 0.15  # Suboptimal but valid
                action_score += penalties['suboptimal_action']
        else:
            # No specific action mentioned
            action_score = penalties['no_justification']

        # Check for hedging ("move up or down")
        hedging_penalty = 0.0
        if ' or ' in response_lower:
            hedging_penalty = penalties['hedging']

        # Part 3: Justification (30%)
        justification_score = 0.0
        if self._has_justification(response_lower):
            justification_score = weights['justification']
        else:
            justification_score = penalties['no_justification']

        # Check for incomplete answer
        incomplete_penalty = 0.0
        if len(response.split()) < 10:  # Very short response
            incomplete_penalty = penalties['incomplete_answer']

        # Combine all parts
        final_score = situation_score + action_score + justification_score + hedging_penalty + incomplete_penalty
        final_score = self._clamp_score(final_score)

        reasoning = f"Situation: {'✓' if situation_score > 0 else '✗'}, Action: {'✓' if mentioned_action else '✗'}, Justif: {'✓' if justification_score > 0 else '✗'}"
        if hedging_penalty < 0:
            reasoning += ", Hedging detected"

        return ScoringResult(
            score=final_score,
            confidence=0.6,
            reasoning=reasoning,
            details={
                'situation_score': situation_score,
                'action_score': action_score,
                'justification_score': justification_score,
                'mentioned_action': mentioned_action
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

    def _has_situation_analysis(self, response_lower: str) -> bool:
        """Check if response analyzes the current game situation."""
        analysis_terms = [
            'situation', 'state', 'position', 'threat', 'opportunity',
            'coming', 'moving', 'approaching', 'heading', 'trajectory',
            'above', 'below', 'near', 'far', 'close'
        ]
        # Require at least 2 analysis terms for a real situation analysis
        count = sum(1 for term in analysis_terms if term in response_lower)
        return count >= 2

    def _has_justification(self, response_lower: str) -> bool:
        """Check if response includes justification for action."""
        justification_terms = [
            'because', 'since', 'to', 'in order to', 'so that',
            'intercept', 'avoid', 'prevent', 'hit', 'catch',
            'defend', 'attack', 'score', 'win', 'survive'
        ]
        return any(term in response_lower for term in justification_terms)

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
